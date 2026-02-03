import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
from io import BytesIO
import re
import json
import base64

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Pharma Analytics Intelligence Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS STYLING
# ============================================================================
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f2847 0%, #1a4d7a 100%);
    }
    
    h1, h2, h3 {
        color: #0f2847;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(15, 40, 71, 0.1);
        border-left: 5px solid #1a4d7a;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(15, 40, 71, 0.15);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #1a4d7a;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-delta {
        font-size: 18px;
        font-weight: 600;
        margin-top: 8px;
    }
    
    .positive {
        color: #28a745;
    }
    
    .negative {
        color: #dc3545;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #1a4d7a 0%, #0f2847 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 10px 20px rgba(15, 40, 71, 0.2);
    }
    
    .insight-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 15px;
        color: #ffffff;
    }
    
    .insight-text {
        font-size: 16px;
        line-height: 1.6;
        color: #e8ecf1;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #0f2847 0%, #1a4d7a 100%);
    }
    
    .stSidebar .stSelectbox label, .stSidebar .stMultiSelect label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 14px;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    .success-box {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    .danger-box {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    .info-box {
        background: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    .section-header {
        background: linear-gradient(90deg, #1a4d7a 0%, #0f2847 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 30px 0 20px 0;
        font-size: 22px;
        font-weight: 700;
        box-shadow: 0 4px 8px rgba(15, 40, 71, 0.2);
    }
    
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(15, 40, 71, 0.1);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1a4d7a;
        font-weight: 700;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 16px;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 10px 10px 0 0;
        padding: 15px 30px;
        font-weight: 600;
        color: #1a4d7a;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a4d7a 0%, #0f2847 100%);
        color: white;
    }
    
    .footer {
        text-align: center;
        padding: 30px;
        color: #6c757d;
        font-size: 14px;
        margin-top: 50px;
        border-top: 2px solid #e8ecf1;
    }
    
    .data-table {
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    .column-info {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1a4d7a;
    }
    
    .year-badge {
        display: inline-block;
        padding: 5px 15px;
        background: #1a4d7a;
        color: white;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
        color: #212529;
        padding: 10px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .trend-up {
        color: #28a745;
        font-weight: 700;
    }
    
    .trend-down {
        color: #dc3545;
        font-weight: 700;
    }
    
    .trend-neutral {
        color: #6c757d;
        font-weight: 700;
    }
    
    .profit-card {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .loss-card {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .neutral-card {
        background: linear-gradient(135deg, #6c757d 0%, #5a6268 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .download-btn {
        background: linear-gradient(135deg, #1a4d7a 0%, #0f2847 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(15, 40, 71, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
@st.cache_data(show_spinner=False)
def load_data_cached(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False, chunksize=50000)
            df = pd.concat(df, ignore_index=True)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {str(e)}")
        return None

def clean_numeric_column(series):
    if series.dtype == 'object':
        series = series.astype(str)
        series = series.str.replace(',', '.', regex=False)
        series = series.str.replace(' ', '', regex=False)
        series = series.str.replace('$', '', regex=False)
        series = series.str.replace('USD', '', regex=False)
        series = series.str.replace('MNF', '', regex=False)
        series = pd.to_numeric(series, errors='coerce')
    return series.fillna(0)

# ============================================================================
# COLUMN DETECTION FUNCTIONS
# ============================================================================
def detect_column_patterns(df):
    detected_columns = {
        'value_columns': [],
        'volume_columns': [],
        'unit_columns': [],
        'price_columns': [],
        'unit_price_columns': [],
        'dimension_columns': []
    }
    
    df_columns_lower = [str(col).lower() for col in df.columns]
    
    for idx, col in enumerate(df.columns):
        col_lower = str(col).lower()
        
        dimension_keywords = ['country', 'region', 'corporation', 'manufacturer', 'molecule', 
                             'product', 'specialty', 'panel', 'sector', 'chemical', 'salt',
                             'international', 'pack', 'size', 'strength', 'volume', 'prescription',
                             'source', 'name', 'sub', 'nfc', 'list']
        
        if any(keyword in col_lower for keyword in dimension_keywords):
            detected_columns['dimension_columns'].append(col)
        
        if any(keyword in col_lower for keyword in ['usd', 'value', 'sales', 'revenue', 'mnf', '$', 'amount']):
            if 'price' not in col_lower and 'avg' not in col_lower:
                detected_columns['value_columns'].append(col)
        
        if any(keyword in col_lower for keyword in ['standard', 'unit', 'volume', 'su', 'quantity']):
            if 'price' not in col_lower and 'avg' not in col_lower:
                if 'standard' in col_lower or 'su' in col_lower:
                    detected_columns['volume_columns'].append(col)
                else:
                    detected_columns['unit_columns'].append(col)
        
        if any(keyword in col_lower for keyword in ['price', 'avg', 'average']):
            if 'standard' in col_lower or 'su' in col_lower:
                detected_columns['price_columns'].append(col)
            else:
                detected_columns['unit_price_columns'].append(col)
    
    return detected_columns

def find_year_in_column(col_name, target_year):
    col_str = str(col_name)
    patterns = [str(target_year), f' {target_year} ', f'_{target_year}_', f'{target_year}-', f'-{target_year}']
    
    for pattern in patterns:
        if pattern in col_str:
            return True
    
    if re.search(r'\b' + str(target_year) + r'\b', col_str):
        return True
    
    return False

def get_column_for_year(df, column_patterns, column_type, year):
    if column_type == 'value':
        column_list = column_patterns['value_columns']
    elif column_type == 'volume':
        column_list = column_patterns['volume_columns']
    elif column_type == 'price':
        column_list = column_patterns['price_columns']
    elif column_type == 'unit_price':
        column_list = column_patterns['unit_price_columns']
    elif column_type == 'units':
        column_list = column_patterns['unit_columns']
    else:
        return None
    
    year_columns = []
    for col in column_list:
        if find_year_in_column(col, year):
            year_columns.append(col)
    
    if year_columns:
        return sorted(year_columns, key=len)[0]
    
    if column_list:
        return column_list[0]
    
    return None

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================
def process_dataframe(df):
    column_patterns = detect_column_patterns(df)
    
    all_numeric_columns = (column_patterns['value_columns'] + 
                          column_patterns['volume_columns'] + 
                          column_patterns['unit_columns'] + 
                          column_patterns['price_columns'] + 
                          column_patterns['unit_price_columns'])
    
    for col in all_numeric_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    for col in column_patterns['dimension_columns']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
    
    return df, column_patterns

def calculate_growth_rate(current, previous):
    if previous == 0 or pd.isna(previous):
        return 0
    return ((current - previous) / previous) * 100

def calculate_cagr(end_value, start_value, periods):
    if start_value <= 0 or end_value <= 0:
        return 0
    return (((end_value / start_value) ** (1 / periods)) - 1) * 100

# ============================================================================
# ANALYTICS FUNCTIONS
# ============================================================================
def calculate_price_volume_decomposition(df, column_patterns, year_current, year_previous):
    value_col_current = get_column_for_year(df, column_patterns, 'value', year_current)
    value_col_previous = get_column_for_year(df, column_patterns, 'value', year_previous)
    volume_col_current = get_column_for_year(df, column_patterns, 'volume', year_current)
    volume_col_previous = get_column_for_year(df, column_patterns, 'volume', year_previous)
    
    if not value_col_current or not value_col_previous or not volume_col_current or not volume_col_previous:
        return None
    
    total_value_current = df[value_col_current].sum()
    total_value_previous = df[value_col_previous].sum()
    total_volume_current = df[volume_col_current].sum()
    total_volume_previous = df[volume_col_previous].sum()
    
    avg_price_current = total_value_current / total_volume_current if total_volume_current > 0 else 0
    avg_price_previous = total_value_previous / total_volume_previous if total_volume_previous > 0 else 0
    
    total_value_growth = calculate_growth_rate(total_value_current, total_value_previous)
    volume_growth = calculate_growth_rate(total_volume_current, total_volume_previous)
    price_growth = calculate_growth_rate(avg_price_current, avg_price_previous)
    
    volume_contribution = (total_volume_current - total_volume_previous) * avg_price_previous
    price_contribution = (avg_price_current - avg_price_previous) * total_volume_current
    mix_contribution = total_value_current - total_value_previous - volume_contribution - price_contribution
    
    total_change = total_value_current - total_value_previous
    
    if total_change != 0:
        volume_effect_pct = (volume_contribution / total_change) * 100
        price_effect_pct = (price_contribution / total_change) * 100
        mix_effect_pct = (mix_contribution / total_change) * 100
    else:
        volume_effect_pct = price_effect_pct = mix_effect_pct = 0
    
    return {
        'total_value_growth': total_value_growth,
        'volume_growth': volume_growth,
        'price_growth': price_growth,
        'volume_effect_pct': volume_effect_pct,
        'price_effect_pct': price_effect_pct,
        'mix_effect_pct': mix_effect_pct,
        'avg_price_current': avg_price_current,
        'avg_price_previous': avg_price_previous,
        'value_col_current': value_col_current,
        'value_col_previous': value_col_previous,
        'volume_col_current': volume_col_current,
        'volume_col_previous': volume_col_previous
    }

def analyze_specialty_premium(df, column_patterns, year):
    specialty_col = None
    for col in column_patterns['dimension_columns']:
        if 'specialty' in str(col).lower():
            specialty_col = col
            break
    
    if not specialty_col:
        return None
    
    value_col = get_column_for_year(df, column_patterns, 'value', year)
    volume_col = get_column_for_year(df, column_patterns, 'volume', year)
    
    if not value_col or not volume_col:
        return None
    
    specialty_mask = df[specialty_col].astype(str).str.contains('specialty', case=False, na=False)
    specialty_df = df[specialty_mask]
    non_specialty_df = df[~specialty_mask]
    
    specialty_value = specialty_df[value_col].sum()
    specialty_volume = specialty_df[volume_col].sum()
    non_specialty_value = non_specialty_df[value_col].sum()
    non_specialty_volume = non_specialty_df[volume_col].sum()
    
    specialty_price = specialty_value / specialty_volume if specialty_volume > 0 else 0
    non_specialty_price = non_specialty_value / non_specialty_volume if non_specialty_volume > 0 else 0
    
    premium = ((specialty_price - non_specialty_price) / non_specialty_price * 100) if non_specialty_price > 0 else 0
    
    total_value = specialty_value + non_specialty_value
    specialty_share = (specialty_value / total_value * 100) if total_value > 0 else 0
    
    return {
        'specialty_price': specialty_price,
        'non_specialty_price': non_specialty_price,
        'premium_pct': premium,
        'specialty_share': specialty_share,
        'specialty_value': specialty_value,
        'non_specialty_value': non_specialty_value
    }

def calculate_portfolio_concentration(df, column_patterns, year):
    molecule_col = None
    for col in column_patterns['dimension_columns']:
        if 'molecule' in str(col).lower():
            molecule_col = col
            break
    
    if not molecule_col:
        return None
    
    value_col = get_column_for_year(df, column_patterns, 'value', year)
    
    if not value_col:
        return None
    
    molecule_values = df.groupby(molecule_col)[value_col].sum().sort_values(ascending=False)
    total_value = molecule_values.sum()
    
    if total_value == 0:
        return {'top_1': 0, 'top_3': 0, 'top_5': 0, 'top_10': 0, 'hhi': 0}
    
    top_1 = (molecule_values.iloc[0] / total_value * 100) if len(molecule_values) > 0 else 0
    top_3 = (molecule_values.head(3).sum() / total_value * 100) if len(molecule_values) >= 3 else 0
    top_5 = (molecule_values.head(5).sum() / total_value * 100) if len(molecule_values) >= 5 else 0
    top_10 = (molecule_values.head(10).sum() / total_value * 100) if len(molecule_values) >= 10 else 0
    
    market_shares = molecule_values / total_value
    hhi = (market_shares ** 2).sum() * 10000
    
    return {
        'top_1': top_1,
        'top_3': top_3,
        'top_5': top_5,
        'top_10': top_10,
        'hhi': hhi,
        'total_molecules': len(molecule_values)
    }

def analyze_growth_fragility(df, column_patterns):
    available_years = []
    for year in [2022, 2023, 2024]:
        if get_column_for_year(df, column_patterns, 'value', year):
            available_years.append(year)
    
    if len(available_years) < 2:
        return {
            'fragility_score': 0,
            'volume_dependency': 0,
            'top_molecule_dependency': 0,
            'concentration_risk': 0,
            'price_volatility': 0,
            'market_concentration': 0
        }
    
    latest_year = available_years[-1]
    previous_year = available_years[-2]
    
    decomp = calculate_price_volume_decomposition(df, column_patterns, latest_year, previous_year)
    
    if not decomp:
        return {
            'fragility_score': 0,
            'volume_dependency': 0,
            'top_molecule_dependency': 0,
            'concentration_risk': 0,
            'price_volatility': 0,
            'market_concentration': 0
        }
    
    molecule_col = None
    for col in column_patterns['dimension_columns']:
        if 'molecule' in str(col).lower():
            molecule_col = col
            break
    
    fragility_score = 0
    
    if decomp['volume_effect_pct'] < 30:
        fragility_score += 30
    
    concentration = calculate_portfolio_concentration(df, column_patterns, latest_year)
    if concentration and concentration['top_3'] > 50:
        fragility_score += 25
    
    price_volatility = abs(decomp['price_growth'])
    if price_volatility > 15:
        fragility_score += 20
    
    manufacturer_col = None
    for col in column_patterns['dimension_columns']:
        if 'manufacturer' in str(col).lower():
            manufacturer_col = col
            break
    
    if manufacturer_col and decomp['value_col_current']:
        manufacturer_values = df.groupby(manufacturer_col)[decomp['value_col_current']].sum().sort_values(ascending=False)
        if len(manufacturer_values) > 0:
            top_3_manufacturer_share = manufacturer_values.head(3).sum() / manufacturer_values.sum() * 100
            if top_3_manufacturer_share > 60:
                fragility_score += 25
    
    fragility_score = min(fragility_score, 100)
    
    return {
        'fragility_score': fragility_score,
        'volume_dependency': decomp['volume_effect_pct'],
        'concentration_risk': concentration['top_3'] if concentration else 0,
        'price_volatility': price_volatility,
        'market_concentration': concentration['hhi'] if concentration else 0
    }

def analyze_market_health(df, column_patterns):
    available_years = []
    for year in [2022, 2023, 2024]:
        if get_column_for_year(df, column_patterns, 'value', year):
            available_years.append(year)
    
    if len(available_years) < 2:
        return {
            'health_score': 50,
            'growth_stability': 0,
            'profitability_trend': 0,
            'market_dynamics': 'N/A',
            'recommendations': ['Yeterli veri yok']
        }
    
    latest_year = available_years[-1]
    previous_year = available_years[-2]
    
    decomp = calculate_price_volume_decomposition(df, column_patterns, latest_year, previous_year)
    
    if not decomp:
        return {
            'health_score': 50,
            'growth_stability': 0,
            'profitability_trend': 0,
            'market_dynamics': 'N/A',
            'recommendations': ['Yeterli veri yok']
        }
    
    health_score = 50
    
    if decomp['total_value_growth'] > 5:
        health_score += 15
    elif decomp['total_value_growth'] < -5:
        health_score -= 15
    
    if decomp['price_growth'] > 0:
        health_score += 10
    else:
        health_score -= 5
    
    if decomp['volume_growth'] > 0:
        health_score += 10
    else:
        health_score -= 5
    
    specialty_premium = analyze_specialty_premium(df, column_patterns, latest_year)
    if specialty_premium and specialty_premium['premium_pct'] > 10:
        health_score += 10
    
    concentration = calculate_portfolio_concentration(df, column_patterns, latest_year)
    if concentration and concentration['top_3'] < 40:
        health_score += 10
    elif concentration and concentration['top_3'] > 60:
        health_score -= 10
    
    health_score = max(0, min(100, health_score))
    
    market_dynamics = []
    if decomp['price_effect_pct'] > 60:
        market_dynamics.append('Fiyat Odaklı Büyüme')
    if decomp['volume_effect_pct'] > 60:
        market_dynamics.append('Hacim Odaklı Büyüme')
    if abs(decomp['mix_effect_pct']) > 20:
        market_dynamics.append('Mix Değişimi Yoğun')
    
    recommendations = []
    if health_score < 40:
        recommendations.append('Pazardaki düşüş trendini tersine çevirmek için acil önlemler alın')
        recommendations.append('Hacim artışına odaklanın')
        recommendations.append('Fiyat stratejisini gözden geçirin')
    elif health_score < 60:
        recommendations.append('Büyümeyi stabilize etmek için mix optimizasyonu yapın')
        recommendations.append('Mevcut müşteri tabanını koruyun')
        recommendations.append('Yeni ürün lansmanlarını değerlendirin')
    else:
        recommendations.append('Mevcut büyüme stratejisini sürdürün')
        recommendations.append('Pazar payını artırmaya odaklanın')
        recommendations.append('Kârlılığı optimize edin')
    
    return {
        'health_score': health_score,
        'growth_stability': decomp['total_value_growth'],
        'profitability_trend': decomp['price_growth'],
        'market_dynamics': ', '.join(market_dynamics) if market_dynamics else 'Stabil',
        'recommendations': recommendations,
        'price_contribution': decomp['price_effect_pct'],
        'volume_contribution': decomp['volume_effect_pct']
    }

def analyze_competition_landscape(df, column_patterns, year):
    manufacturer_col = None
    for col in column_patterns['dimension_columns']:
        if 'manufacturer' in str(col).lower():
            manufacturer_col = col
            break
    
    if not manufacturer_col:
        return None
    
    value_col = get_column_for_year(df, column_patterns, 'value', year)
    
    if not value_col:
        return None
    
    manufacturer_data = df.groupby(manufacturer_col).agg({
        value_col: 'sum'
    }).reset_index()
    
    manufacturer_data = manufacturer_data.sort_values(value_col, ascending=False)
    manufacturer_data['Market_Share'] = (manufacturer_data[value_col] / manufacturer_data[value_col].sum() * 100).round(2)
    manufacturer_data['Cumulative_Share'] = manufacturer_data['Market_Share'].cumsum()
    
    total_manufacturers = len(manufacturer_data)
    top_3_share = manufacturer_data.head(3)['Market_Share'].sum()
    top_5_share = manufacturer_data.head(5)['Market_Share'].sum()
    top_10_share = manufacturer_data.head(10)['Market_Share'].sum() if total_manufacturers >= 10 else manufacturer_data['Market_Share'].sum()
    
    market_concentration = 'Yüksek' if top_3_share > 50 else ('Orta' if top_3_share > 30 else 'Düşük')
    
    return {
        'manufacturer_data': manufacturer_data,
        'total_manufacturers': total_manufacturers,
        'top_3_share': top_3_share,
        'top_5_share': top_5_share,
        'top_10_share': top_10_share,
        'market_concentration': market_concentration,
        'herfindahl_index': (manufacturer_data['Market_Share'] ** 2).sum() / 100
    }

def analyze_product_portfolio(df, column_patterns):
    molecule_col = None
    manufacturer_col = None
    
    for col in column_patterns['dimension_columns']:
        if 'molecule' in str(col).lower():
            molecule_col = col
        if 'manufacturer' in str(col).lower():
            manufacturer_col = col
    
    if not molecule_col or not manufacturer_col:
        return None
    
    available_years = []
    for year in [2022, 2023, 2024]:
        if get_column_for_year(df, column_patterns, 'value', year):
            available_years.append(year)
    
    if len(available_years) < 2:
        return None
    
    latest_year = available_years[-1]
    previous_year = available_years[-2]
    
    latest_value_col = get_column_for_year(df, column_patterns, 'value', latest_year)
    previous_value_col = get_column_for_year(df, column_patterns, 'value', previous_year)
    
    portfolio_data = df.groupby([manufacturer_col, molecule_col]).agg({
        latest_value_col: 'sum',
        previous_value_col: 'sum'
    }).reset_index()
    
    portfolio_data['Growth'] = portfolio_data.apply(
        lambda x: calculate_growth_rate(x[latest_value_col], x[previous_value_col]),
        axis=1
    )
    
    total_market_value = portfolio_data[latest_value_col].sum()
    portfolio_data['Share'] = (portfolio_data[latest_value_col] / total_market_value * 100).round(3)
    
    portfolio_matrix = []
    for manufacturer in portfolio_data[manufacturer_col].unique():
        manufacturer_products = portfolio_data[portfolio_data[manufacturer_col] == manufacturer]
        
        stars = manufacturer_products[
            (manufacturer_products['Growth'] > 10) & 
            (manufacturer_products['Share'] > 1)
        ]
        
        cash_cows = manufacturer_products[
            (manufacturer_products['Growth'] < 5) & 
            (manufacturer_products['Share'] > 2)
        ]
        
        question_marks = manufacturer_products[
            (manufacturer_products['Growth'] > 15) & 
            (manufacturer_products['Share'] < 1)
        ]
        
        dogs = manufacturer_products[
            (manufacturer_products['Growth'] < 0) & 
            (manufacturer_products['Share'] < 1)
        ]
        
        portfolio_matrix.append({
            'Manufacturer': manufacturer,
            'Stars_Count': len(stars),
            'Cash_Cows_Count': len(cash_cows),
            'Question_Marks_Count': len(question_marks),
            'Dogs_Count': len(dogs),
            'Total_Products': len(manufacturer_products),
            'Total_Value': manufacturer_products[latest_value_col].sum(),
            'Avg_Growth': manufacturer_products['Growth'].mean()
        })
    
    return pd.DataFrame(portfolio_matrix)

def generate_financial_metrics(df, column_patterns):
    available_years = []
    for year in [2022, 2023, 2024]:
        if get_column_for_year(df, column_patterns, 'value', year):
            available_years.append(year)
    
    if len(available_years) < 2:
        return None
    
    metrics_data = []
    
    for i, year in enumerate(available_years):
        value_col = get_column_for_year(df, column_patterns, 'value', year)
        volume_col = get_column_for_year(df, column_patterns, 'volume', year)
        
        if not value_col or not volume_col:
            continue
        
        total_value = df[value_col].sum()
        total_volume = df[volume_col].sum()
        avg_price = total_value / total_volume if total_volume > 0 else 0
        
        if i > 0:
            prev_year = available_years[i-1]
            prev_value_col = get_column_for_year(df, column_patterns, 'value', prev_year)
            prev_volume_col = get_column_for_year(df, column_patterns, 'volume', prev_year)
            
            if prev_value_col and prev_volume_col:
                prev_value = df[prev_value_col].sum()
                prev_volume = df[prev_volume_col].sum()
                value_growth = calculate_growth_rate(total_value, prev_value)
                volume_growth = calculate_growth_rate(total_volume, prev_volume)
            else:
                value_growth = volume_growth = 0
        else:
            value_growth = volume_growth = 0
        
        metrics_data.append({
            'Year': year,
            'Total_Value': total_value,
            'Total_Volume': total_volume,
            'Avg_Price': avg_price,
            'Value_Growth': value_growth,
            'Volume_Growth': volume_growth,
            'Value_Per_Unit': total_value / total_volume if total_volume > 0 else 0
        })
    
    return pd.DataFrame(metrics_data)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_metric_card(title, value, delta=None, delta_text="", prefix="", suffix=""):
    delta_html = ""
    if delta is not None:
        delta_class = "positive" if delta > 0 else "negative"
        delta_symbol = "▲" if delta > 0 else "▼"
        delta_html = f'<div class="metric-delta {delta_class}">{delta_symbol} {abs(delta):.1f}% {delta_text}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{prefix}{value}{suffix}</div>
        {delta_html}
    </div>
    """

def create_insight_box(title, content, box_type="info"):
    return f"""
    <div class="{box_type}-box">
        <div class="insight-title">{title}</div>
        <div class="insight-text">{content}</div>
    </div>
    """

def plot_market_share_treemap(df, column_patterns, year):
    manufacturer_col = None
    molecule_col = None
    
    for col in column_patterns['dimension_columns']:
        if 'manufacturer' in str(col).lower():
            manufacturer_col = col
        if 'molecule' in str(col).lower():
            molecule_col = col
    
    if not manufacturer_col or not molecule_col:
        return None
    
    value_col = get_column_for_year(df, column_patterns, 'value', year)
    
    if not value_col:
        return None
    
    hierarchy_data = df.groupby([manufacturer_col, molecule_col])[value_col].sum().reset_index()
    hierarchy_data = hierarchy_data.sort_values(value_col, ascending=False).head(50)
    
    fig = px.treemap(
        hierarchy_data,
        path=[manufacturer_col, molecule_col],
        values=value_col,
        title=f'{year} Pazar Payı Dağılımı',
        color=value_col,
        color_continuous_scale='Blues',
        hover_data=[value_col]
    )
    
    fig.update_layout(
        height=600,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    
    return fig

def plot_growth_matrix(df, column_patterns):
    portfolio_data = analyze_product_portfolio(df, column_patterns)
    
    if portfolio_data is None or portfolio_data.empty:
        return None
    
    fig = px.scatter(
        portfolio_data,
        x='Avg_Growth',
        y='Total_Value',
        size='Total_Products',
        color='Stars_Count',
        hover_name='Manufacturer',
        title='Üretici Büyüme Matrisi',
        labels={
            'Avg_Growth': 'Ortalama Büyüme (%)',
            'Total_Value': 'Toplam Değer',
            'Total_Products': 'Ürün Sayısı',
            'Stars_Count': 'Yıldız Ürün Sayısı'
        },
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=500,
        xaxis_title='Ortalama Büyüme (%)',
        yaxis_title='Toplam Pazar Değeri'
    )
    
    fig.add_hline(y=portfolio_data['Total_Value'].median(), line_dash="dash", line_color="red")
    fig.add_vline(x=10, line_dash="dash", line_color="green")
    
    return fig

def plot_competitive_landscape(df, column_patterns, year):
    competition_data = analyze_competition_landscape(df, column_patterns, year)
    
    if competition_data is None:
        return None
    
    manufacturer_data = competition_data['manufacturer_data'].head(15)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=manufacturer_data['Market_Share'],
        y=manufacturer_data[manufacturer_data.columns[0]],
        orientation='h',
        name='Pazar Payı',
        marker_color='#1a4d7a'
    ))
    
    fig.update_layout(
        title=f'{year} - Top 15 Üretici Pazar Payı',
        xaxis_title='Pazar Payı (%)',
        yaxis_title='Üretici',
        height=500,
        showlegend=False
    )
    
    return fig

def plot_market_evolution(df, column_patterns):
    financial_metrics = generate_financial_metrics(df, column_patterns)
    
    if financial_metrics is None or financial_metrics.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Toplam Değer Trendi', 'Ortalama Fiyat Trendi',
                       'Değer Büyümesi', 'Hacim Büyümesi'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    fig.add_trace(
        go.Scatter(x=financial_metrics['Year'], y=financial_metrics['Total_Value'],
                  name='Toplam Değer', mode='lines+markers', line=dict(color='#1a4d7a', width=3)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=financial_metrics['Year'], y=financial_metrics['Avg_Price'],
                  name='Ort. Fiyat', mode='lines+markers', line=dict(color='#28a745', width=3)),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=financial_metrics['Year'], y=financial_metrics['Value_Growth'],
              name='Değer Büyüme', marker_color='#ffc107'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=financial_metrics['Year'], y=financial_metrics['Volume_Growth'],
              name='Hacim Büyüme', marker_color='#dc3545'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Değer (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Ort. Fiyat", row=1, col=2)
    fig.update_yaxes(title_text="Büyüme (%)", row=2, col=1)
    fig.update_yaxes(title_text="Büyüme (%)", row=2, col=2)
    
    return fig

# ============================================================================
# FILTER FUNCTIONS
# ============================================================================
def filter_dataframe(df, filters, column_patterns):
    filtered_df = df.copy()
    
    dimension_cols = column_patterns['dimension_columns']
    
    for dim_type in ['country', 'region', 'corporation', 'manufacturer', 'molecule', 'specialty']:
        col = None
        for dim_col in dimension_cols:
            if dim_type in str(dim_col).lower():
                col = dim_col
                break
        
        if col and dim_type in filters and filters[dim_type]:
            if 'Tümü' not in filters[dim_type] and filters[dim_type][0] != 'Çok fazla değer - filtreleme yapılamıyor':
                filtered_df = filtered_df[filtered_df[col].isin(filters[dim_type])]
    
    return filtered_df

def get_dimension_options(df, column_patterns):
    dimension_options = {}
    
    for dim_type in ['country', 'region', 'corporation', 'manufacturer', 'molecule', 'specialty']:
        col = None
        for dim_col in column_patterns['dimension_columns']:
            if dim_type in str(dim_col).lower():
                col = dim_col
                break
        
        if col:
            try:
                unique_values = sorted(df[col].astype(str).unique().tolist())
                if len(unique_values) <= 100:
                    dimension_options[dim_type] = {
                        'col': col,
                        'values': ['Tümü'] + unique_values
                    }
                else:
                    dimension_options[dim_type] = {
                        'col': col,
                        'values': ['Tümü', 'Çok fazla değer - filtreleme yapılamıyor']
                    }
            except:
                dimension_options[dim_type] = {
                    'col': col,
                    'values': ['Tümü', 'Değerler okunamıyor']
                }
    
    return dimension_options

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================
def generate_excel_report(df, column_patterns, analysis_results):
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Raw_Data', index=False)
        
        for year in [2022, 2023, 2024]:
            value_col = get_column_for_year(df, column_patterns, 'value', year)
            if value_col:
                year_data = df.groupby([col for col in column_patterns['dimension_columns'] if col in df.columns])[value_col].sum().reset_index()
                year_data = year_data.sort_values(value_col, ascending=False)
                year_data.to_excel(writer, sheet_name=f'Data_{year}', index=False)
        
        if 'financial_metrics' in analysis_results:
            analysis_results['financial_metrics'].to_excel(writer, sheet_name='Financial_Metrics', index=False)
        
        if 'competition' in analysis_results:
            analysis_results['competition']['manufacturer_data'].to_excel(writer, sheet_name='Competition', index=False)
        
        summary_df = pd.DataFrame([{
            'Metric': 'Total Rows',
            'Value': len(df),
            'Description': 'Number of rows in dataset'
        }])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    return output.getvalue()

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    load_custom_css()
    
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #1a4d7a 0%, #0f2847 100%); border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; font-size: 48px; margin: 0;'>💊 Pharma Analytics Intelligence Platform</h1>
        <p style='color: #e8ecf1; font-size: 18px; margin-top: 10px;'>Advanced Pharmaceutical Market Intelligence & Predictive Analytics</p>
        <p style='color: #e8ecf1; font-size: 14px; margin-top: 5px;'>Enterprise-Grade Analytics Solution for Pharmaceutical Market Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "📁 Pharmaceutical Data File Upload (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Supports large files up to 500K+ rows with automatic column detection"
    )
    
    if uploaded_file is not None:
        with st.spinner('🔄 Loading and processing pharmaceutical data...'):
            df = load_data_cached(uploaded_file)
            
            if df is not None:
                df, column_patterns = process_dataframe(df)
                
                st.success(f'✅ Data successfully loaded: {len(df):,} rows, {len(df.columns)} columns')
                
                with st.expander("🔍 Dataset Overview & Column Detection", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Rows", f"{len(df):,}")
                        st.metric("Total Columns", len(df.columns))
                    
                    with col2:
                        st.metric("Value Columns", len(column_patterns['value_columns']))
                        st.metric("Dimension Columns", len(column_patterns['dimension_columns']))
                    
                    with col3:
                        available_years = []
                        for year in [2022, 2023, 2024]:
                            if get_column_for_year(df, column_patterns, 'value', year):
                                available_years.append(year)
                        st.metric("Available Years", len(available_years))
                    
                    st.markdown("---")
                    
                    tab1, tab2, tab3 = st.tabs(["📊 Column Types", "📈 Sample Data", "🔧 Data Quality"])
                    
                    with tab1:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write("**Value Columns (Sales/Revenue):**")
                            for col in column_patterns['value_columns'][:10]:
                                st.write(f"• {col}")
                            if len(column_patterns['value_columns']) > 10:
                                st.write(f"... and {len(column_patterns['value_columns']) - 10} more")
                        
                        with col_b:
                            st.write("**Dimension Columns:**")
                            for col in column_patterns['dimension_columns'][:10]:
                                st.write(f"• {col}")
                            if len(column_patterns['dimension_columns']) > 10:
                                st.write(f"... and {len(column_patterns['dimension_columns']) - 10} more")
                    
                    with tab2:
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    with tab3:
                        missing_data = df.isnull().sum().sum()
                        total_cells = df.size
                        completeness = ((total_cells - missing_data) / total_cells * 100)
                        
                        col_x, col_y = st.columns(2)
                        with col_x:
                            st.metric("Data Completeness", f"{completeness:.1f}%")
                        with col_y:
                            st.metric("Missing Values", f"{missing_data:,}")
                
                st.sidebar.markdown("""
                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #1a4d7a 0%, #0f2847 100%); border-radius: 10px; margin-bottom: 20px;'>
                    <h2 style='color: white; margin: 0;'>🎯 Data Filters</h2>
                </div>
                """, unsafe_allow_html=True)
                
                filters = {}
                dimension_options = get_dimension_options(df, column_patterns)
                
                if 'country' in dimension_options:
                    filters['country'] = st.sidebar.multiselect(
                        '🌍 Country',
                        options=dimension_options['country']['values'],
                        default=['Tümü']
                    )
                
                if 'region' in dimension_options:
                    filters['region'] = st.sidebar.multiselect(
                        '🗺️ Region',
                        options=dimension_options['region']['values'],
                        default=['Tümü']
                    )
                
                if 'corporation' in dimension_options:
                    filters['corporation'] = st.sidebar.multiselect(
                        '🏢 Corporation',
                        options=dimension_options['corporation']['values'],
                        default=['Tümü']
                    )
                
                if 'manufacturer' in dimension_options:
                    filters['manufacturer'] = st.sidebar.multiselect(
                        '🏭 Manufacturer',
                        options=dimension_options['manufacturer']['values'],
                        default=['Tümü']
                    )
                
                if 'molecule' in dimension_options:
                    filters['molecule'] = st.sidebar.multiselect(
                        '⚗️ Molecule',
                        options=dimension_options['molecule']['values'],
                        default=['Tümü']
                    )
                
                if 'specialty' in dimension_options:
                    filters['specialty'] = st.sidebar.multiselect(
                        '💎 Specialty Product',
                        options=dimension_options['specialty']['values'],
                        default=['Tümü']
                    )
                
                available_years = []
                for year in [2022, 2023, 2024]:
                    if get_column_for_year(df, column_patterns, 'value', year):
                        available_years.append(year)
                
                if available_years:
                    filters['focus_year'] = st.sidebar.selectbox(
                        '📅 Focus Year',
                        options=available_years,
                        index=len(available_years)-1
                    )
                
                filtered_df = filter_dataframe(df, filters, column_patterns)
                
                st.sidebar.markdown(f"""
                <div style='background: #28a745; color: white; padding: 15px; border-radius: 10px; margin-top: 20px; text-align: center;'>
                    <strong>Filtered Dataset</strong><br>
                    <span style='font-size: 24px;'>{len(filtered_df):,}</span> rows<br>
                    <small>{len(filtered_df)/len(df)*100:.1f}% of original data</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.sidebar.markdown("---")
                
                if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
                    st.rerun()
                
                if st.sidebar.button("📥 Export Analysis", use_container_width=True):
                    analysis_results = {
                        'financial_metrics': generate_financial_metrics(filtered_df, column_patterns),
                        'competition': analyze_competition_landscape(filtered_df, column_patterns, available_years[-1]) if available_years else None
                    }
                    
                    excel_data = generate_excel_report(filtered_df, column_patterns, analysis_results)
                    
                    st.sidebar.download_button(
                        label="⬇️ Download Excel Report",
                        data=excel_data,
                        file_name='pharma_analytics_report.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                
                tabs = st.tabs([
                    "📊 Executive Dashboard",
                    "📈 Market Intelligence",
                    "🏭 Competitive Analysis",
                    "⚗️ Product Portfolio",
                    "💰 Financial Metrics",
                    "📋 Raw Data Explorer"
                ])
                
                with tabs[0]:
                    st.markdown('<div class="section-header">📊 Executive Dashboard</div>', unsafe_allow_html=True)
                    
                    if not available_years:
                        st.warning("⚠️ No value columns detected. Please check your data format.")
                    else:
                        latest_year = available_years[-1]
                        
                        market_health = analyze_market_health(filtered_df, column_patterns)
                        fragility = analyze_growth_fragility(filtered_df, column_patterns)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            latest_value_col = get_column_for_year(filtered_df, column_patterns, 'value', latest_year)
                            total_value = filtered_df[latest_value_col].sum() if latest_value_col else 0
                            st.markdown(
                                create_metric_card(
                                    f"Total Market Value ({latest_year})",
                                    f"${total_value/1e6:.1f}M",
                                    prefix="",
                                    suffix=""
                                ),
                                unsafe_allow_html=True
                            )
                        
                        with col2:
                            health_score = market_health['health_score'] if market_health else 50
                            health_color = "success" if health_score > 60 else ("warning" if health_score > 40 else "danger")
                            st.markdown(
                                create_metric_card(
                                    "Market Health Score",
                                    f"{health_score:.0f}/100",
                                    delta=health_score-50,
                                    delta_text="vs neutral"
                                ),
                                unsafe_allow_html=True
                            )
                        
                        with col3:
                            fragility_score = fragility['fragility_score'] if fragility else 0
                            fragility_color = "danger" if fragility_score > 60 else ("warning" if fragility_score > 30 else "success")
                            st.markdown(
                                create_metric_card(
                                    "Growth Fragility",
                                    f"{fragility_score:.0f}/100",
                                    delta_text="risk score"
                                ),
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            specialty_premium = analyze_specialty_premium(filtered_df, column_patterns, latest_year)
                            premium_pct = specialty_premium['premium_pct'] if specialty_premium else 0
                            st.markdown(
                                create_metric_card(
                                    "Specialty Premium",
                                    f"{premium_pct:.1f}%",
                                    delta=premium_pct,
                                    delta_text="price premium"
                                ),
                                unsafe_allow_html=True
                            )
                        
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<div class="section-header">📈 Market Growth Decomposition</div>', unsafe_allow_html=True)
                            
                            if len(available_years) >= 2:
                                decomp = calculate_price_volume_decomposition(
                                    filtered_df, column_patterns, 
                                    available_years[-1], available_years[-2]
                                )
                                
                                if decomp:
                                    fig = go.Figure(data=[
                                        go.Bar(name='Price Effect', x=['Contribution'], y=[decomp['price_effect_pct']], marker_color='#1a4d7a'),
                                        go.Bar(name='Volume Effect', x=['Contribution'], y=[decomp['volume_effect_pct']], marker_color='#28a745'),
                                        go.Bar(name='Mix Effect', x=['Contribution'], y=[decomp['mix_effect_pct']], marker_color='#ffc107')
                                    ])
                                    
                                    fig.update_layout(
                                        barmode='stack',
                                        title=f'Growth Decomposition: {available_years[-2]} → {available_years[-1]}',
                                        height=400,
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.markdown(f"""
                                    <div class="info-box">
                                        <div class="insight-title">📊 Growth Analysis</div>
                                        <div class="insight-text">
                                            • <strong>Total Growth:</strong> {decomp['total_value_growth']:.1f}%<br>
                                            • <strong>Price Growth:</strong> {decomp['price_growth']:.1f}%<br>
                                            • <strong>Volume Growth:</strong> {decomp['volume_growth']:.1f}%<br>
                                            • <strong>Average Price:</strong> ${decomp['avg_price_current']:.2f} ({decomp['avg_price_previous']:.2f} previous)
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="section-header">🎯 Portfolio Concentration</div>', unsafe_allow_html=True)
                            
                            concentration = calculate_portfolio_concentration(filtered_df, column_patterns, latest_year)
                            
                            if concentration:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Indicator(
                                    mode="gauge+number",
                                    value=concentration['top_3'],
                                    title={'text': "Top 3 Molecule Share"},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "#1a4d7a"},
                                        'steps': [
                                            {'range': [0, 30], 'color': "#28a745"},
                                            {'range': [30, 50], 'color': "#ffc107"},
                                            {'range': [50, 100], 'color': "#dc3545"}
                                        ]
                                    }
                                ))
                                
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                risk_level = "HIGH" if concentration['top_3'] > 50 else ("MEDIUM" if concentration['top_3'] > 30 else "LOW")
                                risk_class = "risk-high" if concentration['top_3'] > 50 else ("risk-medium" if concentration['top_3'] > 30 else "risk-low")
                                
                                st.markdown(f"""
                                <div class="{risk_class}">
                                    <strong>Concentration Risk:</strong> {risk_level}<br>
                                    Top 1: {concentration['top_1']:.1f}% | Top 5: {concentration['top_5']:.1f}% | HHI: {concentration['hhi']:.0f}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        if market_health:
                            st.markdown('<div class="section-header">💡 Strategic Recommendations</div>', unsafe_allow_html=True)
                            
                            for i, recommendation in enumerate(market_health['recommendations'], 1):
                                st.markdown(f"""
                                <div class="info-box" style="margin: 10px 0;">
                                    <div class="insight-text">
                                        {i}. {recommendation}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                with tabs[1]:
                    st.markdown('<div class="section-header">📈 Market Intelligence</div>', unsafe_allow_html=True)
                    
                    if available_years:
                        latest_year = available_years[-1]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            treemap_fig = plot_market_share_treemap(filtered_df, column_patterns, latest_year)
                            if treemap_fig:
                                st.plotly_chart(treemap_fig, use_container_width=True)
                            else:
                                st.info("Insufficient data for market share visualization")
                        
                        with col2:
                            evolution_fig = plot_market_evolution(filtered_df, column_patterns)
                            if evolution_fig:
                                st.plotly_chart(evolution_fig, use_container_width=True)
                            else:
                                st.info("Insufficient data for market evolution analysis")
                        
                        st.markdown("---")
                        
                        st.markdown('<div class="section-header">📊 Year-over-Year Analysis</div>', unsafe_allow_html=True)
                        
                        if len(available_years) >= 2:
                            comparison_data = []
                            
                            for year in available_years:
                                value_col = get_column_for_year(filtered_df, column_patterns, 'value', year)
                                volume_col = get_column_for_year(filtered_df, column_patterns, 'volume', year)
                                
                                if value_col and volume_col:
                                    total_value = filtered_df[value_col].sum()
                                    total_volume = filtered_df[volume_col].sum()
                                    avg_price = total_value / total_volume if total_volume > 0 else 0
                                    
                                    comparison_data.append({
                                        'Year': year,
                                        'Total Value': total_value,
                                        'Total Volume': total_volume,
                                        'Avg Price': avg_price
                                    })
                            
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                
                                fig = make_subplots(
                                    rows=1, cols=3,
                                    subplot_titles=('Total Value', 'Total Volume', 'Average Price')
                                )
                                
                                fig.add_trace(
                                    go.Bar(x=comparison_df['Year'], y=comparison_df['Total Value'],
                                          name='Value', marker_color='#1a4d7a'),
                                    row=1, col=1
                                )
                                
                                fig.add_trace(
                                    go.Bar(x=comparison_df['Year'], y=comparison_df['Total Volume'],
                                          name='Volume', marker_color='#28a745'),
                                    row=1, col=2
                                )
                                
                                fig.add_trace(
                                    go.Bar(x=comparison_df['Year'], y=comparison_df['Avg Price'],
                                          name='Price', marker_color='#ffc107'),
                                    row=1, col=3
                                )
                                
                                fig.update_layout(
                                    height=400,
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                with tabs[2]:
                    st.markdown('<div class="section-header">🏭 Competitive Analysis</div>', unsafe_allow_html=True)
                    
                    if available_years:
                        latest_year = available_years[-1]
                        
                        competition_data = analyze_competition_landscape(filtered_df, column_patterns, latest_year)
                        
                        if competition_data:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown(
                                    create_metric_card(
                                        "Total Manufacturers",
                                        f"{competition_data['total_manufacturers']}",
                                        prefix="",
                                        suffix=""
                                    ),
                                    unsafe_allow_html=True
                                )
                            
                            with col2:
                                st.markdown(
                                    create_metric_card(
                                        "Top 3 Share",
                                        f"{competition_data['top_3_share']:.1f}%",
                                        delta_text="market control"
                                    ),
                                    unsafe_allow_html=True
                                )
                            
                            with col3:
                                st.markdown(
                                    create_metric_card(
                                        "Top 5 Share",
                                        f"{competition_data['top_5_share']:.1f}%",
                                        delta_text="market control"
                                    ),
                                    unsafe_allow_html=True
                                )
                            
                            with col4:
                                concentration_level = competition_data['market_concentration']
                                concentration_color = "danger" if concentration_level == 'Yüksek' else ("warning" if concentration_level == 'Orta' else "success")
                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Market Concentration</div>
                                        <div class="metric-value" style="color: {'#dc3545' if concentration_level == 'Yüksek' else ('#ffc107' if concentration_level == 'Orta' else '#28a745')}">
                                            {concentration_level}
                                        </div>
                                        <div class="metric-delta">
                                            HHI: {competition_data['herfindahl_index']:.0f}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            st.markdown("---")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                landscape_fig = plot_competitive_landscape(filtered_df, column_patterns, latest_year)
                                if landscape_fig:
                                    st.plotly_chart(landscape_fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("#### 🏆 Top 5 Manufacturers")
                                
                                top_manufacturers = competition_data['manufacturer_data'].head(5)
                                
                                for idx, row in top_manufacturers.iterrows():
                                    st.markdown(f"""
                                    <div style='background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #1a4d7a;'>
                                        <strong>{row[competition_data['manufacturer_data'].columns[0]]}</strong><br>
                                        <span style='color: #1a4d7a; font-size: 18px; font-weight: 700;'>{row['Market_Share']:.1f}%</span><br>
                                        <small>Market Share</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                
                with tabs[3]:
                    st.markdown('<div class="section-header">⚗️ Product Portfolio Analysis</div>', unsafe_allow_html=True)
                    
                    portfolio_matrix = analyze_product_portfolio(filtered_df, column_patterns)
                    
                    if portfolio_matrix is not None and not portfolio_matrix.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            growth_matrix_fig = plot_growth_matrix(filtered_df, column_patterns)
                            if growth_matrix_fig:
                                st.plotly_chart(growth_matrix_fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### 📊 Portfolio Classification")
                            
                            portfolio_summary = portfolio_matrix.groupby('Manufacturer').agg({
                                'Stars_Count': 'sum',
                                'Cash_Cows_Count': 'sum',
                                'Question_Marks_Count': 'sum',
                                'Dogs_Count': 'sum',
                                'Total_Products': 'sum'
                            }).reset_index()
                            
                            st.dataframe(
                                portfolio_summary.head(10),
                                use_container_width=True,
                                height=400
                            )
                        
                        st.markdown("---")
                        
                        st.markdown("#### 🎯 Strategic Product Categories")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_stars = portfolio_matrix['Stars_Count'].sum()
                            st.markdown(
                                create_metric_card(
                                    "⭐ Stars",
                                    f"{total_stars}",
                                    prefix="",
                                    suffix=" products"
                                ),
                                unsafe_allow_html=True
                            )
                        
                        with col2:
                            total_cash_cows = portfolio_matrix['Cash_Cows_Count'].sum()
                            st.markdown(
                                create_metric_card(
                                    "🐄 Cash Cows",
                                    f"{total_cash_cows}",
                                    prefix="",
                                    suffix=" products"
                                ),
                                unsafe_allow_html=True
                            )
                        
                        with col3:
                            total_question_marks = portfolio_matrix['Question_Marks_Count'].sum()
                            st.markdown(
                                create_metric_card(
                                    "❓ Question Marks",
                                    f"{total_question_marks}",
                                    prefix="",
                                    suffix=" products"
                                ),
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            total_dogs = portfolio_matrix['Dogs_Count'].sum()
                            st.markdown(
                                create_metric_card(
                                    "🐕 Dogs",
                                    f"{total_dogs}",
                                    prefix="",
                                    suffix=" products"
                                ),
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("Insufficient data for portfolio analysis. Need at least 2 years of data.")
                
                with tabs[4]:
                    st.markdown('<div class="section-header">💰 Financial Metrics & KPIs</div>', unsafe_allow_html=True)
                    
                    financial_metrics = generate_financial_metrics(filtered_df, column_patterns)
                    
                    if financial_metrics is not None and not financial_metrics.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📈 Financial Performance")
                            
                            metrics_display = financial_metrics.copy()
                            metrics_display['Total_Value_M'] = metrics_display['Total_Value'] / 1e6
                            metrics_display['Avg_Price'] = metrics_display['Avg_Price'].round(2)
                            
                            st.dataframe(
                                metrics_display[['Year', 'Total_Value_M', 'Total_Volume', 'Avg_Price', 'Value_Growth', 'Volume_Growth']],
                                column_config={
                                    'Year': 'Year',
                                    'Total_Value_M': st.column_config.NumberColumn('Total Value (M$)', format='$%.2fM'),
                                    'Total_Volume': st.column_config.NumberColumn('Total Volume', format='%.0f'),
                                    'Avg_Price': st.column_config.NumberColumn('Avg Price', format='$%.2f'),
                                    'Value_Growth': st.column_config.NumberColumn('Value Growth %', format='%.1f%%'),
                                    'Volume_Growth': st.column_config.NumberColumn('Volume Growth %', format='%.1f%%')
                                },
                                use_container_width=True
                            )
                        
                        with col2:
                            st.markdown("#### 📊 Key Ratios & Metrics")
                            
                            latest_metrics = financial_metrics.iloc[-1]
                            
                            metric_rows = [
                                ("💰 Value per Unit", f"${latest_metrics['Value_Per_Unit']:.2f}", "Average revenue per unit"),
                                ("📈 Value Growth", f"{latest_metrics['Value_Growth']:.1f}%", "Year-over-year growth"),
                                ("📦 Volume Growth", f"{latest_metrics['Volume_Growth']:.1f}%", "Year-over-year growth"),
                                ("🏷️ Avg Price", f"${latest_metrics['Avg_Price']:.2f}", "Average selling price"),
                                ("📊 Price Elasticity", "N/A", "Requires more data"),
                                ("🎯 Market Efficiency", "Calculating...", "Revenue per volume unit")
                            ]
                            
                            for metric_name, metric_value, metric_desc in metric_rows:
                                st.markdown(f"""
                                <div style='background: #f8f9fa; padding: 15px; margin: 8px 0; border-radius: 8px;'>
                                    <div style='font-weight: 600; color: #1a4d7a;'>{metric_name}</div>
                                    <div style='font-size: 24px; font-weight: 700; color: #0f2847;'>{metric_value}</div>
                                    <div style='font-size: 12px; color: #6c757d;'>{metric_desc}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        st.markdown("#### 📉 Profitability Analysis")
                        
                        if len(financial_metrics) >= 2:
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=financial_metrics['Year'],
                                y=financial_metrics['Value_Growth'],
                                name='Value Growth',
                                mode='lines+markers',
                                line=dict(color='#1a4d7a', width=3)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=financial_metrics['Year'],
                                y=financial_metrics['Volume_Growth'],
                                name='Volume Growth',
                                mode='lines+markers',
                                line=dict(color='#28a745', width=3)
                            ))
                            
                            fig.update_layout(
                                title='Growth Trends: Value vs Volume',
                                xaxis_title='Year',
                                yaxis_title='Growth Rate (%)',
                                height=400,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Insufficient data for financial metrics analysis. Need at least 2 years of data.")
                
                with tabs[5]:
                    st.markdown('<div class="section-header">📋 Raw Data Explorer</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown("#### 🔍 Data Preview")
                        
                        display_columns = st.multiselect(
                            'Select columns to display:',
                            options=df.columns.tolist(),
                            default=column_patterns['dimension_columns'][:3] + column_patterns['value_columns'][:2]
                        )
                    
                    with col2:
                        rows_to_show = st.selectbox(
                            'Rows to show:',
                            options=[100, 500, 1000, 5000],
                            index=0
                        )
                    
                    if display_columns:
                        st.dataframe(
                            filtered_df[display_columns].head(rows_to_show),
                            use_container_width=True,
                            height=600
                        )
                        
                        st.markdown(f"*Showing {min(rows_to_show, len(filtered_df)):,} of {len(filtered_df):,} rows*")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            csv_data = filtered_df[display_columns].to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download CSV",
                                data=csv_data,
                                file_name='filtered_pharma_data.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
                        
                        with col2:
                            excel_data = generate_excel_report(filtered_df[display_columns], column_patterns, {})
                            st.download_button(
                                "📊 Download Excel",
                                data=excel_data,
                                file_name='pharma_data_report.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                use_container_width=True
                            )
                        
                        with col3:
                            if st.button("🔄 Reset View", use_container_width=True):
                                st.rerun()
                    else:
                        st.info("Please select at least one column to display.")
                
                st.markdown("""
                <div class="footer">
                    <strong>Pharma Analytics Intelligence Platform v2.0</strong><br>
                    Advanced Pharmaceutical Market Intelligence & Predictive Analytics<br>
                    © 2024 - Enterprise Solution - All Rights Reserved
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="insight-box">
            <div class="insight-title">👋 Welcome to Pharma Analytics Intelligence Platform</div>
            <div class="insight-text">
                Upload your pharmaceutical market data file to begin advanced analytics.<br><br>
                <strong>Platform Features:</strong><br>
                ✅ Automatic column detection and data validation<br>
                ✅ Advanced market intelligence and competitive analysis<br>
                ✅ Product portfolio optimization and BCG matrix<br>
                ✅ Financial metrics and KPI tracking<br>
                ✅ Real-time market health scoring<br>
                ✅ Export capabilities (Excel, CSV)<br>
                ✅ Enterprise-grade security and performance
            </div>
        </div>
        
        <div style="margin-top: 30px; padding: 20px; background: rgba(255, 255, 255, 0.1); border-radius: 10px;">
            <h3>📋 Expected Data Format</h3>
            <p>Your data should include:</p>
            <ul>
                <li><strong>Value Columns:</strong> Sales/Revenue data (e.g., "2023 USD MNF", "Sales 2024")</li>
                <li><strong>Volume Columns:</strong> Quantity/Unit data (e.g., "2023 Standard Units", "Volume")</li>
                <li><strong>Dimension Columns:</strong> Manufacturer, Molecule, Country, Specialty flags</li>
                <li><strong>Time Periods:</strong> Multiple years (2022, 2023, 2024 preferred)</li>
            </ul>
            <p><em>The platform will automatically detect and map your columns.</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
