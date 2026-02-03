"""
================================================================================
PHARMA ANALYTICS INTELLIGENCE PLATFORM - PROFESSIONAL EDITION
================================================================================
Version: 3.0.0
Author: Principal Data Engineer  
License: Proprietary
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
from io import BytesIO
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Pharma Analytics Intelligence Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_custom_css():
    """Load professional CSS styling"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%); }
    
    .stApp { background: linear-gradient(135deg, #0f2847 0%, #1a4d7a 100%); }
    
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
    
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    
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
    
    .stSidebar {
        background: linear-gradient(180deg, #0f2847 0%, #1a4d7a 100%);
    }
    
    .stSidebar .stSelectbox label, .stSidebar .stMultiSelect label {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .footer {
        text-align: center;
        padding: 30px;
        color: #6c757d;
        font-size: 14px;
        margin-top: 50px;
        border-top: 2px solid #e8ecf1;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data_cached(uploaded_file):
    """Load data with robust error handling"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False, chunksize=50000)
            df = pd.concat(df, ignore_index=True)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        df.columns = df.columns.str.strip()
        logger.info(f"Loaded {len(df):,} rows")
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {str(e)}")
        return None

def clean_numeric_column(series):
    """Clean numeric columns"""
    if series.dtype == 'object':
        series = series.astype(str).str.replace(',', '.', regex=False)
        series = series.str.replace(' ', '', regex=False)
        series = pd.to_numeric(series, errors='coerce')
    return series.fillna(0)

def process_dataframe(df):
    """Process and clean dataframe"""
    dimension_cols = [
        'Source.Name', 'Country', 'Sector', 'Panel', 'Region', 'Sub-Region',
        'Corporation', 'Manufacturer', 'Molecule List', 'Molecule',
        'Chemical Salt', 'International Product', 'Specialty Product',
        'NFC123', 'International Pack', 'International Strength',
        'International Size', 'International Volume', 'International Prescription'
    ]
    
    metric_cols = []
    for year in [2022, 2023, 2024]:
        metric_cols.extend([
            f'MAT Q3 {year} USD MNF',
            f'MAT Q3 {year} Standard Units',
            f'MAT Q3 {year} Units',
            f'MAT Q3 {year} SU Avg Price USD MNF',
            f'MAT Q3 {year} Unit Avg Price USD MNF'
        ])
    
    for col in metric_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    for col in dimension_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
    
    return df


def calculate_growth_rate(current, previous):
    """Calculate growth rate"""
    if previous == 0 or pd.isna(previous):
        return 0
    return ((current - previous) / previous) * 100

def calculate_cagr(end_value, start_value, periods):
    """Calculate CAGR"""
    if start_value <= 0 or end_value <= 0:
        return 0
    return (((end_value / start_value) ** (1 / periods)) - 1) * 100

def calculate_price_volume_decomposition(df, year_current, year_previous):
    """Price-volume decomposition analysis"""
    try:
        value_col_current = f'MAT Q3 {year_current} USD MNF'
        value_col_previous = f'MAT Q3 {year_previous} USD MNF'
        volume_col_current = f'MAT Q3 {year_current} Standard Units'
        volume_col_previous = f'MAT Q3 {year_previous} Standard Units'
        
        if not all(col in df.columns for col in [value_col_current, value_col_previous, volume_col_current, volume_col_previous]):
            return {
                'total_value_growth': 0, 'volume_growth': 0, 'price_growth': 0,
                'volume_effect_pct': 0, 'price_effect_pct': 0, 'mix_effect_pct': 0,
                'avg_price_current': 0, 'avg_price_previous': 0
            }
        
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
            'avg_price_previous': avg_price_previous
        }
    except Exception as e:
        logger.error(f"Decomposition error: {str(e)}")
        return {
            'total_value_growth': 0, 'volume_growth': 0, 'price_growth': 0,
            'volume_effect_pct': 0, 'price_effect_pct': 0, 'mix_effect_pct': 0,
            'avg_price_current': 0, 'avg_price_previous': 0
        }


def analyze_specialty_premium(df, year):
    """Analyze specialty premium"""
    try:
        value_col = f'MAT Q3 {year} USD MNF'
        volume_col = f'MAT Q3 {year} Standard Units'
        
        specialty_df = df[df['Specialty Product'] == 'SPECIALTY']
        non_specialty_df = df[df['Specialty Product'] == 'NON SPECIALTY']
        
        specialty_value = specialty_df[value_col].sum()
        specialty_volume = specialty_df[volume_col].sum()
        non_specialty_value = non_specialty_df[value_col].sum()
        non_specialty_volume = non_specialty_df[volume_col].sum()
        
        specialty_price = specialty_value / specialty_volume if specialty_volume > 0 else 0
        non_specialty_price = non_specialty_value / non_specialty_volume if non_specialty_volume > 0 else 0
        
        premium = ((specialty_price - non_specialty_price) / non_specialty_price * 100) if non_specialty_price > 0 else 0
        
        return {
            'specialty_price': specialty_price,
            'non_specialty_price': non_specialty_price,
            'premium_pct': premium,
            'specialty_share': (specialty_value / (specialty_value + non_specialty_value) * 100) if (specialty_value + non_specialty_value) > 0 else 0
        }
    except Exception as e:
        return {'specialty_price': 0, 'non_specialty_price': 0, 'premium_pct': 0, 'specialty_share': 0}

def calculate_portfolio_concentration(df, year):
    """Calculate portfolio concentration"""
    try:
        value_col = f'MAT Q3 {year} USD MNF'
        
        molecule_values = df.groupby('Molecule')[value_col].sum().sort_values(ascending=False)
        total_value = molecule_values.sum()
        
        if total_value == 0:
            return {'top_1': 0, 'top_3': 0, 'top_5': 0, 'top_10': 0, 'hhi': 0}
        
        top_1 = (molecule_values.iloc[0] / total_value * 100) if len(molecule_values) > 0 else 0
        top_3 = (molecule_values.head(3).sum() / total_value * 100) if len(molecule_values) >= 3 else 0
        top_5 = (molecule_values.head(5).sum() / total_value * 100) if len(molecule_values) >= 5 else 0
        top_10 = (molecule_values.head(10).sum() / total_value * 100) if len(molecule_values) >= 10 else 0
        
        market_shares = molecule_values / total_value
        hhi = (market_shares ** 2).sum() * 10000
        
        return {'top_1': top_1, 'top_3': top_3, 'top_5': top_5, 'top_10': top_10, 'hhi': hhi}
    except:
        return {'top_1': 0, 'top_3': 0, 'top_5': 0, 'top_10': 0, 'hhi': 0}

def detect_product_exits(df):
    """Detect exited products"""
    try:
        molecules_2022 = set(df[df['MAT Q3 2022 USD MNF'] > 0]['Molecule'].unique())
        molecules_2024 = set(df[df['MAT Q3 2024 USD MNF'] > 0]['Molecule'].unique())
        
        exited_molecules = molecules_2022 - molecules_2024
        
        exit_analysis = []
        for molecule in exited_molecules:
            molecule_df = df[df['Molecule'] == molecule]
            value_2022 = molecule_df['MAT Q3 2022 USD MNF'].sum()
            manufacturer = molecule_df['Manufacturer'].mode()[0] if len(molecule_df) > 0 else 'Unknown'
            
            exit_analysis.append({
                'Molecule': molecule,
                'Manufacturer': manufacturer,
                'Value_2022': value_2022
            })
        
        return pd.DataFrame(exit_analysis).sort_values('Value_2022', ascending=False)
    except:
        return pd.DataFrame()

def analyze_growth_fragility(df):
    """Growth fragility analysis"""
    try:
        decomp = calculate_price_volume_decomposition(df, 2024, 2023)
        concentration = calculate_portfolio_concentration(df, 2024)
        
        fragility_score = 0
        if decomp['volume_effect_pct'] < 30:
            fragility_score += 30
        if concentration['top_3'] > 50:
            fragility_score += 25
        if abs(decomp['price_growth']) > 15:
            fragility_score += 20
        
        return {
            'fragility_score': fragility_score,
            'volume_dependency': decomp['volume_effect_pct'],
            'concentration_risk': concentration['top_3'],
            'price_volatility': abs(decomp['price_growth'])
        }
    except:
        return {'fragility_score': 0, 'volume_dependency': 0, 'concentration_risk': 0, 'price_volatility': 0}

def detect_growth_engine_reversal(df):
    """Detect growth engine reversal"""
    try:
        decomp_2223 = calculate_price_volume_decomposition(df, 2023, 2022)
        decomp_2324 = calculate_price_volume_decomposition(df, 2024, 2023)
        
        reversal_detected = False
        reversal_type = "Yok"
        
        if decomp_2223['price_effect_pct'] > 60 and decomp_2324['volume_effect_pct'] > 60:
            reversal_detected = True
            reversal_type = "Fiyat Odaklı → Hacim Odaklı"
        elif decomp_2223['volume_effect_pct'] > 60 and decomp_2324['price_effect_pct'] > 60:
            reversal_detected = True
            reversal_type = "Hacim Odaklı → Fiyat Odaklı"
        
        return {
            'reversal_detected': reversal_detected,
            'reversal_type': reversal_type,
            'price_effect_2223': decomp_2223['price_effect_pct'],
            'volume_effect_2223': decomp_2223['volume_effect_pct'],
            'price_effect_2324': decomp_2324['price_effect_pct'],
            'volume_effect_2324': decomp_2324['volume_effect_pct']
        }
    except:
        return {
            'reversal_detected': False,
            'reversal_type': 'Yok',
            'price_effect_2223': 0,
            'volume_effect_2223': 0,
            'price_effect_2324': 0,
            'volume_effect_2324': 0
        }

def analyze_channel_migration(df):
    """Channel migration analysis"""
    try:
        results = []
        for year in [2022, 2023, 2024]:
            value_col = f'MAT Q3 {year} USD MNF'
            specialty_value = df[df['Specialty Product'] == 'SPECIALTY'][value_col].sum()
            non_specialty_value = df[df['Specialty Product'] == 'NON SPECIALTY'][value_col].sum()
            total_value = specialty_value + non_specialty_value
            
            specialty_share = (specialty_value / total_value * 100) if total_value > 0 else 0
            
            results.append({
                'Year': year,
                'Specialty_Share': specialty_share,
                'Non_Specialty_Share': 100 - specialty_share
            })
        
        df_migration = pd.DataFrame(results)
        migration_trend = df_migration['Specialty_Share'].iloc[-1] - df_migration['Specialty_Share'].iloc[0]
        
        return {
            'df_migration': df_migration,
            'migration_trend': migration_trend,
            'specialty_share_2024': df_migration['Specialty_Share'].iloc[-1]
        }
    except:
        return {'df_migration': pd.DataFrame(), 'migration_trend': 0, 'specialty_share_2024': 0}

def analyze_regional_polarization(df):
    """Regional polarization analysis"""
    try:
        regional_data = []
        for year in [2022, 2023, 2024]:
            value_col = f'MAT Q3 {year} USD MNF'
            region_values = df.groupby('Region')[value_col].sum().reset_index()
            region_values['Year'] = year
            region_values.columns = ['Region', 'Value', 'Year']
            regional_data.append(region_values)
        
        df_regional = pd.concat(regional_data, ignore_index=True)
        pivot_df = df_regional.pivot(index='Region', columns='Year', values='Value').fillna(0)
        pivot_df['Growth_2022_2024'] = pivot_df.apply(
            lambda x: calculate_growth_rate(x[2024], x[2022]), axis=1
        )
        pivot_df['Share_2024'] = pivot_df[2024] / pivot_df[2024].sum() * 100
        growth_std = pivot_df['Growth_2022_2024'].std()
        
        return {
            'df_regional': pivot_df.reset_index(),
            'polarization_score': growth_std / 10,
            'growth_std': growth_std
        }
    except:
        return {'df_regional': pd.DataFrame(), 'polarization_score': 0, 'growth_std': 0}

def analyze_top_corporations_shift(df):
    """Corporation market shift analysis"""
    try:
        corp_data = []
        for year in [2022, 2023, 2024]:
            value_col = f'MAT Q3 {year} USD MNF'
            corp_values = df.groupby('Corporation')[value_col].sum().sort_values(ascending=False).head(10)
            total_value = df[value_col].sum()
            
            for rank, (corp, value) in enumerate(corp_values.items(), 1):
                corp_data.append({
                    'Year': year,
                    'Corporation': corp,
                    'Rank': rank,
                    'Value': value,
                    'Market_Share': (value / total_value * 100) if total_value > 0 else 0
                })
        
        df_corp = pd.DataFrame(corp_data)
        top_3_2022 = df_corp[(df_corp['Year'] == 2022) & (df_corp['Rank'] <= 3)]['Market_Share'].sum()
        top_3_2024 = df_corp[(df_corp['Year'] == 2024) & (df_corp['Rank'] <= 3)]['Market_Share'].sum()
        
        return {
            'df_corp': df_corp,
            'top_3_share_2022': top_3_2022,
            'top_3_share_2024': top_3_2024,
            'concentration_change': top_3_2024 - top_3_2022
        }
    except:
        return {'df_corp': pd.DataFrame(), 'top_3_share_2022': 0, 'top_3_share_2024': 0, 'concentration_change': 0}

def identify_structural_growth_molecules(df):
    """Identify structural growth molecules"""
    try:
        molecule_performance = df.groupby('Molecule').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2022 Standard Units': 'sum',
            'MAT Q3 2023 Standard Units': 'sum',
            'MAT Q3 2024 Standard Units': 'sum'
        }).reset_index()
        
        molecule_performance['Growth_2223'] = molecule_performance.apply(
            lambda x: calculate_growth_rate(x['MAT Q3 2023 USD MNF'], x['MAT Q3 2022 USD MNF']), axis=1
        )
        molecule_performance['Growth_2324'] = molecule_performance.apply(
            lambda x: calculate_growth_rate(x['MAT Q3 2024 USD MNF'], x['MAT Q3 2023 USD MNF']), axis=1
        )
        molecule_performance['CAGR'] = molecule_performance.apply(
            lambda x: calculate_cagr(x['MAT Q3 2024 USD MNF'], x['MAT Q3 2022 USD MNF'], 2), axis=1
        )
        molecule_performance['Volume_CAGR'] = molecule_performance.apply(
            lambda x: calculate_cagr(x['MAT Q3 2024 Standard Units'], x['MAT Q3 2022 Standard Units'], 2), axis=1
        )
        
        structural_growth = molecule_performance[
            (molecule_performance['Growth_2223'] > 5) &
            (molecule_performance['Growth_2324'] > 5) &
            (molecule_performance['Volume_CAGR'] > 3) &
            (molecule_performance['MAT Q3 2024 USD MNF'] > molecule_performance['MAT Q3 2024 USD MNF'].quantile(0.25))
        ].sort_values('CAGR', ascending=False)
        
        return structural_growth
    except:
        return pd.DataFrame()

def detect_relaunch_signals(df):
    """Detect relaunch signals"""
    try:
        product_df = df.groupby(['Molecule', 'Chemical Salt', 'International Strength', 'International Pack']).agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'Manufacturer': 'first'
        }).reset_index()
        
        product_df['Active_2022'] = product_df['MAT Q3 2022 USD MNF'] > 0
        product_df['Active_2024'] = product_df['MAT Q3 2024 USD MNF'] > 0
        
        relaunch_candidates = product_df[
            (~product_df['Active_2022']) &
            (product_df['Active_2024']) &
            (product_df['MAT Q3 2024 USD MNF'] > product_df['MAT Q3 2024 USD MNF'].quantile(0.50))
        ]
        
        molecule_counts = df.groupby('Molecule').agg({
            'International Strength': 'nunique',
            'Chemical Salt': 'nunique',
            'International Pack': 'nunique'
        }).reset_index()
        molecule_counts.columns = ['Molecule', 'Strength_Variants', 'Salt_Variants', 'Pack_Variants']
        
        high_variation = molecule_counts[
            (molecule_counts['Strength_Variants'] > 5) |
            (molecule_counts['Salt_Variants'] > 2) |
            (molecule_counts['Pack_Variants'] > 10)
        ]
        
        return {'relaunch_candidates': relaunch_candidates, 'high_variation_molecules': high_variation}
    except:
        return {'relaunch_candidates': pd.DataFrame(), 'high_variation_molecules': pd.DataFrame()}

def detect_saturation_commoditization(df):
    """Detect saturation and commoditization"""
    try:
        molecule_metrics = df.groupby('Molecule').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2024 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2022 Standard Units': 'sum',
            'MAT Q3 2024 Standard Units': 'sum',
            'Manufacturer': 'nunique'
        }).reset_index()
        
        molecule_metrics['Value_Growth'] = molecule_metrics.apply(
            lambda x: calculate_growth_rate(x['MAT Q3 2024 USD MNF'], x['MAT Q3 2022 USD MNF']), axis=1
        )
        molecule_metrics['Price_Change'] = molecule_metrics.apply(
            lambda x: calculate_growth_rate(x['MAT Q3 2024 SU Avg Price USD MNF'], x['MAT Q3 2022 SU Avg Price USD MNF']), axis=1
        )
        molecule_metrics['Volume_Growth'] = molecule_metrics.apply(
            lambda x: calculate_growth_rate(x['MAT Q3 2024 Standard Units'], x['MAT Q3 2022 Standard Units']), axis=1
        )
        
        saturation_signals = molecule_metrics[
            (molecule_metrics['Volume_Growth'] < 2) &
            (molecule_metrics['Value_Growth'] < 5)
        ].sort_values('MAT Q3 2024 USD MNF', ascending=False)
        
        commoditization_signals = molecule_metrics[
            (molecule_metrics['Price_Change'] < -5) &
            (molecule_metrics['Manufacturer'] > 5)
        ].sort_values('Price_Change')
        
        return {
            'saturation_molecules': saturation_signals,
            'commoditization_molecules': commoditization_signals
        }
    except:
        return {'saturation_molecules': pd.DataFrame(), 'commoditization_molecules': pd.DataFrame()}

def calculate_manufacturer_pricing_power(df):
    """Calculate manufacturer pricing power"""
    try:
        manufacturer_metrics = df.groupby('Manufacturer').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2024 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2022 Standard Units': 'sum',
            'MAT Q3 2024 Standard Units': 'sum'
        }).reset_index()
        
        manufacturer_metrics['Price_Growth'] = manufacturer_metrics.apply(
            lambda x: calculate_growth_rate(x['MAT Q3 2024 SU Avg Price USD MNF'], x['MAT Q3 2022 SU Avg Price USD MNF']), axis=1
        )
        manufacturer_metrics['Volume_Growth'] = manufacturer_metrics.apply(
            lambda x: calculate_growth_rate(x['MAT Q3 2024 Standard Units'], x['MAT Q3 2022 Standard Units']), axis=1
        )
        manufacturer_metrics['Value_Growth'] = manufacturer_metrics.apply(
            lambda x: calculate_growth_rate(x['MAT Q3 2024 USD MNF'], x['MAT Q3 2022 USD MNF']), axis=1
        )
        
        manufacturer_metrics['Pricing_Power_Score'] = (
            manufacturer_metrics['Price_Growth'] * 0.5 +
            manufacturer_metrics['Value_Growth'] * 0.3 +
            manufacturer_metrics['Volume_Growth'].clip(0, 50) * 0.2
        ).clip(0, 100)
        
        return manufacturer_metrics.sort_values('Pricing_Power_Score', ascending=False)
    except:
        return pd.DataFrame()

def calculate_manufacturer_volume_scale(df):
    """Calculate manufacturer volume scale"""
    try:
        manufacturer_metrics = df.groupby('Manufacturer').agg({
            'MAT Q3 2024 Standard Units': 'sum',
            'MAT Q3 2022 Standard Units': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'Molecule': 'nunique'
        }).reset_index()
        
        manufacturer_metrics['Volume_Growth'] = manufacturer_metrics.apply(
            lambda x: calculate_growth_rate(x['MAT Q3 2024 Standard Units'], x['MAT Q3 2022 Standard Units']), axis=1
        )
        
        total_volume_2024 = manufacturer_metrics['MAT Q3 2024 Standard Units'].sum()
        manufacturer_metrics['Volume_Share'] = (
            manufacturer_metrics['MAT Q3 2024 Standard Units'] / total_volume_2024 * 100
        )
        
        manufacturer_metrics['Volume_Scale_Score'] = (
            manufacturer_metrics['Volume_Share'] * 0.4 +
            manufacturer_metrics['Volume_Growth'].clip(0, 50) * 0.3 +
            manufacturer_metrics['Molecule'].clip(0, 20) * 0.3
        ).clip(0, 100)
        
        return manufacturer_metrics.sort_values('Volume_Scale_Score', ascending=False)
    except:
        return pd.DataFrame()

def detect_margin_erosion(df):
    """Detect margin erosion"""
    try:
        manufacturer_metrics = df.groupby('Manufacturer').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2022 Standard Units': 'sum',
            'MAT Q3 2024 Standard Units': 'sum'
        }).reset_index()
        
        manufacturer_metrics['Unit_Margin_2022'] = (
            manufacturer_metrics['MAT Q3 2022 USD MNF'] / manufacturer_metrics['MAT Q3 2022 Standard Units']
        )
        manufacturer_metrics['Unit_Margin_2024'] = (
            manufacturer_metrics['MAT Q3 2024 USD MNF'] / manufacturer_metrics['MAT Q3 2024 Standard Units']
        )
        manufacturer_metrics['Margin_Change'] = manufacturer_metrics.apply(
            lambda x: calculate_growth_rate(x['Unit_Margin_2024'], x['Unit_Margin_2022']), axis=1
        )
        manufacturer_metrics['Erosion_Flag'] = manufacturer_metrics['Margin_Change'] < -10
        
        return manufacturer_metrics[manufacturer_metrics['Erosion_Flag']].sort_values('Margin_Change')
    except:
        return pd.DataFrame()

def detect_molecule_dependency_risk(df):
    """Detect molecule dependency risk"""
    try:
        manufacturer_molecule = df.groupby(['Manufacturer', 'Molecule']).agg({
            'MAT Q3 2024 USD MNF': 'sum'
        }).reset_index()
        
        manufacturer_total = df.groupby('Manufacturer').agg({
            'MAT Q3 2024 USD MNF': 'sum'
        }).reset_index()
        manufacturer_total.columns = ['Manufacturer', 'Total_Value']
        
        merged = manufacturer_molecule.merge(manufacturer_total, on='Manufacturer')
        merged['Molecule_Share'] = (merged['MAT Q3 2024 USD MNF'] / merged['Total_Value'] * 100)
        
        top_molecule_dependency = merged.loc[merged.groupby('Manufacturer')['Molecule_Share'].idxmax()]
        top_molecule_dependency['Dependency_Risk'] = top_molecule_dependency['Molecule_Share'] > 30
        
        return top_molecule_dependency[top_molecule_dependency['Dependency_Risk']].sort_values('Molecule_Share', ascending=False)
    except:
        return pd.DataFrame()

def analyze_su_vs_unit_price_divergence(df):
    """Analyze SU vs Unit price divergence"""
    try:
        price_analysis = df.groupby('Molecule').agg({
            'MAT Q3 2024 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2024 Unit Avg Price USD MNF': 'mean',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2024 Standard Units': 'sum',
            'MAT Q3 2024 Units': 'sum'
        }).reset_index()
        
        price_analysis['SU_Unit_Ratio'] = (
            price_analysis['MAT Q3 2024 SU Avg Price USD MNF'] / 
            price_analysis['MAT Q3 2024 Unit Avg Price USD MNF']
        )
        price_analysis['Price_Divergence'] = abs(price_analysis['SU_Unit_Ratio'] - 1) * 100
        
        return price_analysis[price_analysis['Price_Divergence'] > 20].sort_values('Price_Divergence', ascending=False)
    except:
        return pd.DataFrame()

def analyze_pack_size_optimization(df):
    """Analyze pack size optimization"""
    try:
        pack_analysis = df.groupby(['Molecule', 'International Pack', 'International Size']).agg({
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2024 Standard Units': 'sum',
            'MAT Q3 2024 Units': 'sum',
            'MAT Q3 2024 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2024 Unit Avg Price USD MNF': 'mean'
        }).reset_index()
        
        pack_analysis['Units_per_SU'] = (
            pack_analysis['MAT Q3 2024 Units'] / pack_analysis['MAT Q3 2024 Standard Units']
        )
        pack_analysis['Value_per_Unit'] = (
            pack_analysis['MAT Q3 2024 USD MNF'] / pack_analysis['MAT Q3 2024 Units']
        )
        
        molecule_avg_price = pack_analysis.groupby('Molecule')['Value_per_Unit'].mean().reset_index()
        molecule_avg_price.columns = ['Molecule', 'Avg_Value_per_Unit']
        
        pack_analysis = pack_analysis.merge(molecule_avg_price, on='Molecule')
        pack_analysis['Price_Index'] = (pack_analysis['Value_per_Unit'] / pack_analysis['Avg_Value_per_Unit']) * 100
        
        premium_packs = pack_analysis[pack_analysis['Price_Index'] > 120].sort_values('Price_Index', ascending=False)
        discount_packs = pack_analysis[pack_analysis['Price_Index'] < 80].sort_values('Price_Index')
        
        return {'premium_packs': premium_packs, 'discount_packs': discount_packs, 'all_packs': pack_analysis}
    except:
        return {'premium_packs': pd.DataFrame(), 'discount_packs': pd.DataFrame(), 'all_packs': pd.DataFrame()}

def detect_hidden_discounting(df):
    """Detect hidden discounting"""
    try:
        manufacturer_pricing = df.groupby('Manufacturer').agg({
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2024 Standard Units': 'sum',
            'MAT Q3 2024 Units': 'sum',
            'MAT Q3 2024 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2024 Unit Avg Price USD MNF': 'mean',
            'MAT Q3 2023 SU Avg Price USD MNF': 'mean'
        }).reset_index()
        
        manufacturer_pricing['Realized_SU_Price'] = (
            manufacturer_pricing['MAT Q3 2024 USD MNF'] / manufacturer_pricing['MAT Q3 2024 Standard Units']
        )
        manufacturer_pricing['Reported_SU_Price'] = manufacturer_pricing['MAT Q3 2024 SU Avg Price USD MNF']
        manufacturer_pricing['Price_Realization_Rate'] = (
            manufacturer_pricing['Realized_SU_Price'] / manufacturer_pricing['Reported_SU_Price'] * 100
        )
        manufacturer_pricing['Hidden_Discount_Signal'] = manufacturer_pricing['Price_Realization_Rate'] < 95
        
        return manufacturer_pricing[manufacturer_pricing['Hidden_Discount_Signal']].sort_values('Price_Realization_Rate')
    except:
        return pd.DataFrame()

def generate_executive_summary(df):
    """Generate executive summary"""
    try:
        decomp_2324 = calculate_price_volume_decomposition(df, 2024, 2023)
        specialty_premium_2024 = analyze_specialty_premium(df, 2024)
        specialty_premium_2022 = analyze_specialty_premium(df, 2022)
        concentration_2024 = calculate_portfolio_concentration(df, 2024)
        exits = detect_product_exits(df)
        fragility = analyze_growth_fragility(df)
        
        return {
            'price_volume_decomp': decomp_2324,
            'specialty_premium': specialty_premium_2024,
            'premium_change': specialty_premium_2024['premium_pct'] - specialty_premium_2022['premium_pct'],
            'concentration': concentration_2024,
            'product_exits_count': len(exits),
            'fragility': fragility,
            'total_value_2024': df['MAT Q3 2024 USD MNF'].sum(),
            'total_value_2022': df['MAT Q3 2022 USD MNF'].sum(),
            'cagr': calculate_cagr(df['MAT Q3 2024 USD MNF'].sum(), df['MAT Q3 2022 USD MNF'].sum(), 2)
        }
    except:
        return {}

def create_metric_card(title, value, delta=None, delta_text="", prefix="", suffix=""):
    """Create metric card HTML"""
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
    """Create insight box HTML"""
    return f"""
    <div class="{box_type}-box">
        <div class="insight-title">{title}</div>
        <div class="insight-text">{content}</div>
    </div>
    """

def filter_dataframe(df, filters):
    """Filter dataframe"""
    filtered_df = df.copy()
    
    if 'country' in filters and filters['country']:
        if 'Tümü' not in filters['country']:
            filtered_df = filtered_df[filtered_df['Country'].isin(filters['country'])]
    
    if 'corporation' in filters and filters['corporation']:
        if 'Tümü' not in filters['corporation']:
            filtered_df = filtered_df[filtered_df['Corporation'].isin(filters['corporation'])]
    
    if 'manufacturer' in filters and filters['manufacturer']:
        if 'Tümü' not in filters['manufacturer']:
            filtered_df = filtered_df[filtered_df['Manufacturer'].isin(filters['manufacturer'])]
    
    if 'molecule' in filters and filters['molecule']:
        if 'Tümü' not in filters['molecule']:
            filtered_df = filtered_df[filtered_df['Molecule'].isin(filters['molecule'])]
    
    if 'specialty' in filters and filters['specialty']:
        if 'Tümü' not in filters['specialty']:
            filtered_df = filtered_df[filtered_df['Specialty Product'].isin(filters['specialty'])]
    
    return filtered_df

def main():
    """Main application function"""
    load_custom_css()
    
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #1a4d7a 0%, #0f2847 100%); border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; font-size: 48px; margin: 0;'>💊 Pharma Analytics Intelligence Platform</h1>
        <p style='color: #e8ecf1; font-size: 18px; margin-top: 10px;'>Principal Data Engineering & Advanced Pharmaceutical Market Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "📁 Veri Dosyasını Yükleyin (CSV veya Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="500K+ satırlık büyük dosyalar desteklenmektedir"
    )
    
    if uploaded_file is not None:
        with st.spinner('🔄 Veri yükleniyor ve işleniyor...'):
            df = load_data_cached(uploaded_file)
            
            if df is not None:
                df = process_dataframe(df)
                
                st.success(f'✅ Veri başarıyla yüklendi: {len(df):,} satır')
                
                st.sidebar.markdown("""
                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #1a4d7a 0%, #0f2847 100%); border-radius: 10px; margin-bottom: 20px;'>
                    <h2 style='color: white; margin: 0;'>🎯 Filtreler</h2>
                </div>
                """, unsafe_allow_html=True)
                
                filters = {}
                
                countries = ['Tümü'] + sorted(df['Country'].unique().tolist())
                filters['country'] = st.sidebar.multiselect(
                    '🌍 Ülke',
                    options=countries,
                    default=['Tümü']
                )
                
                corporations = ['Tümü'] + sorted(df['Corporation'].unique().tolist())
                filters['corporation'] = st.sidebar.multiselect(
                    '🏢 Şirket',
                    options=corporations,
                    default=['Tümü']
                )
                
                manufacturers = ['Tümü'] + sorted(df['Manufacturer'].unique().tolist())
                filters['manufacturer'] = st.sidebar.multiselect(
                    '🏭 Üretici',
                    options=manufacturers,
                    default=['Tümü']
                )
                
                molecules = ['Tümü'] + sorted(df['Molecule'].unique().tolist())
                filters['molecule'] = st.sidebar.multiselect(
                    '⚗️ Molekül',
                    options=molecules,
                    default=['Tümü']
                )
                
                specialty_options = ['Tümü'] + sorted(df['Specialty Product'].unique().tolist())
                filters['specialty'] = st.sidebar.multiselect(
                    '💎 Specialty Ürün',
                    options=specialty_options,
                    default=['Tümü']
                )
                
                filtered_df = filter_dataframe(df, filters)
                
                st.sidebar.markdown(f"""
                <div style='background: #28a745; color: white; padding: 15px; border-radius: 10px; margin-top: 20px; text-align: center;'>
                    <strong>Filtrelenmiş Veri</strong><br>
                    <span style='font-size: 24px;'>{len(filtered_df):,}</span> satır
                </div>
                """, unsafe_allow_html=True)
                
                tabs = st.tabs([
                    "📊 Executive Summary",
                    "🔍 Pazar Kaymaları",
                    "⚗️ Molekül Zekâsı",
                    "🏭 Üretici Skorlama",
                    "💰 Fiyat & Mix Analizi",
                    "📈 Detaylı Grafikler",
                    "📋 Veri Tablosu"
                ])
                
                with tabs[0]:
                    st.markdown('<div class="section-header">📊 Executive Summary</div>', unsafe_allow_html=True)
                    
                    summary = generate_executive_summary(filtered_df)
                    
                    if summary:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(
                                create_metric_card(
                                    "Toplam Pazar Değeri (2024)",
                                    f"${summary['total_value_2024']/1e6:.1f}M",
                                    delta=summary['cagr'],
                                    delta_text="CAGR"
                                ),
                                unsafe_allow_html=True
                            )
                        
                        with col2:
                            st.markdown(
                                create_metric_card(
                                    "Fiyat Etkisi",
                                    f"{summary['price_volume_decomp']['price_effect_pct']:.1f}%",
                                    delta=summary['price_volume_decomp']['price_effect_pct'],
                                    delta_text="katkı"
                                ),
                                unsafe_allow_html=True
                            )
                        
                        with col3:
                            st.markdown(
                                create_metric_card(
                                    "Hacim Etkisi",
                                    f"{summary['price_volume_decomp']['volume_effect_pct']:.1f}%",
                                    delta=summary['price_volume_decomp']['volume_effect_pct'],
                                    delta_text="katkı"
                                ),
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                create_metric_card(
                                    "Specialty Primi",
                                    f"{summary['specialty_premium']['premium_pct']:.1f}%",
                                    delta=summary['premium_change'],
                                    delta_text="değişim"
                                ),
                                unsafe_allow_html=True
                            )
                        
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<div class="section-header">🎯 Portföy Yoğunlaşma Riski</div>', unsafe_allow_html=True)
                            
                            conc = summary['concentration']
                            
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=conc['top_3'],
                                title={'text': "Top 3 Molekül Payı (%)"},
                                delta={'reference': 50},
                                gauge={
                                    'axis': {'range': [None, 100]},
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
                        
                        with col2:
                            st.markdown('<div class="section-header">⚠️ Büyüme Kırılganlık Skoru</div>', unsafe_allow_html=True)
                            
                            frag = summary['fragility']
                            
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=frag['fragility_score'],
                                title={'text': "Kırılganlık Skoru (0-100)"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#dc3545"},
                                    'steps': [
                                        {'range': [0, 30], 'color': "#28a745"},
                                        {'range': [30, 60], 'color': "#ffc107"},
                                        {'range': [60, 100], 'color': "#dc3545"}
                                    ]
                                }
                            ))
                            
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                
                with tabs[6]:
                    st.markdown('<div class="section-header">📋 Veri Tablosu</div>', unsafe_allow_html=True)
                    
                    display_columns = st.multiselect(
                        'Görüntülenecek kolonları seçin:',
                        options=filtered_df.columns.tolist(),
                        default=['Molecule', 'Manufacturer', 'Country', 'Specialty Product', 
                                 'MAT Q3 2024 USD MNF', 'MAT Q3 2024 Standard Units']
                    )
                    
                    if display_columns:
                        st.dataframe(
                            filtered_df[display_columns].head(100),
                            use_container_width=True,
                            height=600
                        )
                        
                        csv = filtered_df[display_columns].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 CSV İndir",
                            data=csv,
                            file_name='pharma_analytics_filtered.csv',
                            mime='text/csv',
                        )
                
                st.markdown("""
                <div class="footer">
                    <strong>Pharma Analytics Intelligence Platform</strong><br>
                    Professional Data Engineering & Advanced Market Analytics<br>
                    © 2024 - Version 3.0.0
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="insight-box">
            <div class="insight-title">👋 Hoş Geldiniz!</div>
            <div class="insight-text">
                Lütfen analiz için bir CSV veya Excel dosyası yükleyin.<br><br>
                <strong>Özellikler:</strong><br>
                ✅ 500K+ satırlık büyük veri desteği<br>
                ✅ Virgül ondalık ayraç desteği<br>
                ✅ Gerçek zamanlı analitik<br>
                ✅ 20+ ileri düzey içgörü<br>
                ✅ Production-grade performans
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
