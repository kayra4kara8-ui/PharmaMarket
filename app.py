import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
from io import BytesIO
import locale

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Pharma Analytics Intelligence Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    </style>
    """, unsafe_allow_html=True)

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
        series = series.astype(str).str.replace(',', '.', regex=False)
        series = series.str.replace(' ', '', regex=False)
        series = pd.to_numeric(series, errors='coerce')
    return series.fillna(0)

def process_dataframe(df):
    dimension_cols = [
        'Source.Name', 'Country', 'Sector', 'Panel', 'Region', 'Sub-Region',
        'Corporation', 'Manufacturer', 'Molecule List', 'Molecule',
        'Chemical Salt', 'International Product', 'Specialty Product',
        'NFC123', 'International Pack', 'International Strength',
        'International Size', 'International Volume', 'International Prescription'
    ]
    
    metric_cols_2022 = [
        'MAT Q3 2022 USD MNF', 'MAT Q3 2022 Standard Units', 'MAT Q3 2022 Units',
        'MAT Q3 2022 SU Avg Price USD MNF', 'MAT Q3 2022 Unit Avg Price USD MNF'
    ]
    
    metric_cols_2023 = [
        'MAT Q3 2023 USD MNF', 'MAT Q3 2023 Standard Units', 'MAT Q3 2023 Units',
        'MAT Q3 2023 SU Avg Price USD MNF', 'MAT Q3 2023 Unit Avg Price USD MNF'
    ]
    
    metric_cols_2024 = [
        'MAT Q3 2024 USD MNF', 'MAT Q3 2024 Standard Units', 'MAT Q3 2024 Units',
        'MAT Q3 2024 SU Avg Price USD MNF', 'MAT Q3 2024 Unit Avg Price USD MNF'
    ]
    
    all_metrics = metric_cols_2022 + metric_cols_2023 + metric_cols_2024
    
    for col in all_metrics:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    for col in dimension_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
    
    return df

def calculate_growth_rate(current, previous):
    if previous == 0 or pd.isna(previous):
        return 0
    return ((current - previous) / previous) * 100

def calculate_cagr(end_value, start_value, periods):
    if start_value <= 0 or end_value <= 0:
        return 0
    return (((end_value / start_value) ** (1 / periods)) - 1) * 100

def calculate_price_volume_decomposition(df, year_current, year_previous):
    metrics_current = f'MAT Q3 {year_current}'
    metrics_previous = f'MAT Q3 {year_previous}'
    
    value_col_current = f'{metrics_current} USD MNF'
    value_col_previous = f'{metrics_previous} USD MNF'
    volume_col_current = f'{metrics_current} Standard Units'
    volume_col_previous = f'{metrics_previous} Standard Units'
    price_col_current = f'{metrics_current} SU Avg Price USD MNF'
    price_col_previous = f'{metrics_previous} SU Avg Price USD MNF'
    
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

def analyze_specialty_premium(df, year):
    metrics = f'MAT Q3 {year}'
    value_col = f'{metrics} USD MNF'
    volume_col = f'{metrics} Standard Units'
    
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

def calculate_portfolio_concentration(df, year):
    metrics = f'MAT Q3 {year}'
    value_col = f'{metrics} USD MNF'
    
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
    
    return {
        'top_1': top_1,
        'top_3': top_3,
        'top_5': top_5,
        'top_10': top_10,
        'hhi': hhi
    }

def detect_product_exits(df):
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

def analyze_growth_fragility(df):
    value_growth_2324 = calculate_growth_rate(
        df['MAT Q3 2024 USD MNF'].sum(),
        df['MAT Q3 2023 USD MNF'].sum()
    )
    
    decomp = calculate_price_volume_decomposition(df, 2024, 2023)
    
    top_molecules = df.groupby('Molecule').agg({
        'MAT Q3 2024 USD MNF': 'sum',
        'MAT Q3 2023 USD MNF': 'sum'
    }).reset_index()
    
    top_molecules['growth'] = top_molecules.apply(
        lambda x: calculate_growth_rate(x['MAT Q3 2024 USD MNF'], x['MAT Q3 2023 USD MNF']),
        axis=1
    )
    
    top_molecules['contribution'] = (
        (top_molecules['MAT Q3 2024 USD MNF'] - top_molecules['MAT Q3 2023 USD MNF']) /
        (df['MAT Q3 2024 USD MNF'].sum() - df['MAT Q3 2023 USD MNF'].sum()) * 100
    )
    
    top_contributors = top_molecules.nlargest(5, 'contribution')
    
    fragility_score = 0
    
    if decomp['volume_effect_pct'] < 30:
        fragility_score += 30
    
    concentration = calculate_portfolio_concentration(df, 2024)
    if concentration['top_3'] > 50:
        fragility_score += 25
    
    if top_contributors['contribution'].iloc[0] > 40:
        fragility_score += 25
    
    price_volatility = abs(decomp['price_growth'])
    if price_volatility > 15:
        fragility_score += 20
    
    return {
        'fragility_score': fragility_score,
        'volume_dependency': decomp['volume_effect_pct'],
        'top_molecule_dependency': top_contributors['contribution'].iloc[0] if len(top_contributors) > 0 else 0,
        'concentration_risk': concentration['top_3'],
        'price_volatility': price_volatility
    }

def detect_growth_engine_reversal(df):
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

def analyze_channel_migration(df):
    results = []
    
    for year in [2022, 2023, 2024]:
        metrics = f'MAT Q3 {year}'
        value_col = f'{metrics} USD MNF'
        
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

def analyze_regional_polarization(df):
    regional_data = []
    
    for year in [2022, 2023, 2024]:
        metrics = f'MAT Q3 {year}'
        value_col = f'{metrics} USD MNF'
        
        region_values = df.groupby('Region')[value_col].sum().reset_index()
        region_values['Year'] = year
        region_values.columns = ['Region', 'Value', 'Year']
        regional_data.append(region_values)
    
    df_regional = pd.concat(regional_data, ignore_index=True)
    
    pivot_df = df_regional.pivot(index='Region', columns='Year', values='Value').fillna(0)
    pivot_df['Growth_2022_2024'] = pivot_df.apply(
        lambda x: calculate_growth_rate(x[2024], x[2022]),
        axis=1
    )
    
    pivot_df['Share_2024'] = pivot_df[2024] / pivot_df[2024].sum() * 100
    
    growth_std = pivot_df['Growth_2022_2024'].std()
    
    polarization_score = growth_std / 10
    
    return {
        'df_regional': pivot_df.reset_index(),
        'polarization_score': polarization_score,
        'growth_std': growth_std
    }

def analyze_top_corporations_shift(df):
    corp_data = []
    
    for year in [2022, 2023, 2024]:
        metrics = f'MAT Q3 {year}'
        value_col = f'{metrics} USD MNF'
        
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
    
    concentration_change = top_3_2024 - top_3_2022
    
    return {
        'df_corp': df_corp,
        'top_3_share_2022': top_3_2022,
        'top_3_share_2024': top_3_2024,
        'concentration_change': concentration_change
    }

def identify_structural_growth_molecules(df):
    molecule_performance = df.groupby('Molecule').agg({
        'MAT Q3 2022 USD MNF': 'sum',
        'MAT Q3 2023 USD MNF': 'sum',
        'MAT Q3 2024 USD MNF': 'sum',
        'MAT Q3 2022 Standard Units': 'sum',
        'MAT Q3 2023 Standard Units': 'sum',
        'MAT Q3 2024 Standard Units': 'sum'
    }).reset_index()
    
    molecule_performance['Growth_2223'] = molecule_performance.apply(
        lambda x: calculate_growth_rate(x['MAT Q3 2023 USD MNF'], x['MAT Q3 2022 USD MNF']),
        axis=1
    )
    
    molecule_performance['Growth_2324'] = molecule_performance.apply(
        lambda x: calculate_growth_rate(x['MAT Q3 2024 USD MNF'], x['MAT Q3 2023 USD MNF']),
        axis=1
    )
    
    molecule_performance['CAGR'] = molecule_performance.apply(
        lambda x: calculate_cagr(x['MAT Q3 2024 USD MNF'], x['MAT Q3 2022 USD MNF'], 2),
        axis=1
    )
    
    molecule_performance['Volume_CAGR'] = molecule_performance.apply(
        lambda x: calculate_cagr(x['MAT Q3 2024 Standard Units'], x['MAT Q3 2022 Standard Units'], 2),
        axis=1
    )
    
    structural_growth = molecule_performance[
        (molecule_performance['Growth_2223'] > 5) &
        (molecule_performance['Growth_2324'] > 5) &
        (molecule_performance['Volume_CAGR'] > 3) &
        (molecule_performance['MAT Q3 2024 USD MNF'] > molecule_performance['MAT Q3 2024 USD MNF'].quantile(0.25))
    ].sort_values('CAGR', ascending=False)
    
    return structural_growth

def detect_relaunch_signals(df):
    product_df = df.groupby(['Molecule', 'Chemical Salt', 'International Strength', 'International Pack']).agg({
        'MAT Q3 2022 USD MNF': 'sum',
        'MAT Q3 2023 USD MNF': 'sum',
        'MAT Q3 2024 USD MNF': 'sum',
        'Manufacturer': 'first'
    }).reset_index()
    
    product_df['Active_2022'] = product_df['MAT Q3 2022 USD MNF'] > 0
    product_df['Active_2023'] = product_df['MAT Q3 2023 USD MNF'] > 0
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
    
    return {
        'relaunch_candidates': relaunch_candidates,
        'high_variation_molecules': high_variation
    }

def detect_saturation_commoditization(df):
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
        lambda x: calculate_growth_rate(x['MAT Q3 2024 USD MNF'], x['MAT Q3 2022 USD MNF']),
        axis=1
    )
    
    molecule_metrics['Price_Change'] = molecule_metrics.apply(
        lambda x: calculate_growth_rate(x['MAT Q3 2024 SU Avg Price USD MNF'], x['MAT Q3 2022 SU Avg Price USD MNF']),
        axis=1
    )
    
    molecule_metrics['Volume_Growth'] = molecule_metrics.apply(
        lambda x: calculate_growth_rate(x['MAT Q3 2024 Standard Units'], x['MAT Q3 2022 Standard Units']),
        axis=1
    )
    
    saturation_signals = molecule_metrics[
        (molecule_metrics['Volume_Growth'] < 2) &
        (molecule_metrics['Value_Growth'] < 5)
    ]
    
    commoditization_signals = molecule_metrics[
        (molecule_metrics['Price_Change'] < -5) &
        (molecule_metrics['Manufacturer'] > 5)
    ]
    
    return {
        'saturation_molecules': saturation_signals.sort_values('MAT Q3 2024 USD MNF', ascending=False),
        'commoditization_molecules': commoditization_signals.sort_values('Price_Change')
    }

def calculate_manufacturer_pricing_power(df):
    manufacturer_metrics = df.groupby('Manufacturer').agg({
        'MAT Q3 2022 USD MNF': 'sum',
        'MAT Q3 2024 USD MNF': 'sum',
        'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
        'MAT Q3 2024 SU Avg Price USD MNF': 'mean',
        'MAT Q3 2022 Standard Units': 'sum',
        'MAT Q3 2024 Standard Units': 'sum'
    }).reset_index()
    
    manufacturer_metrics['Price_Growth'] = manufacturer_metrics.apply(
        lambda x: calculate_growth_rate(x['MAT Q3 2024 SU Avg Price USD MNF'], x['MAT Q3 2022 SU Avg Price USD MNF']),
        axis=1
    )
    
    manufacturer_metrics['Volume_Growth'] = manufacturer_metrics.apply(
        lambda x: calculate_growth_rate(x['MAT Q3 2024 Standard Units'], x['MAT Q3 2022 Standard Units']),
        axis=1
    )
    
    manufacturer_metrics['Value_Growth'] = manufacturer_metrics.apply(
        lambda x: calculate_growth_rate(x['MAT Q3 2024 USD MNF'], x['MAT Q3 2022 USD MNF']),
        axis=1
    )
    
    manufacturer_metrics['Pricing_Power_Score'] = (
        manufacturer_metrics['Price_Growth'] * 0.5 +
        manufacturer_metrics['Value_Growth'] * 0.3 +
        (manufacturer_metrics['Volume_Growth'] * 0.2 if manufacturer_metrics['Volume_Growth'].mean() > 0 else 0)
    )
    
    manufacturer_metrics['Pricing_Power_Score'] = manufacturer_metrics['Pricing_Power_Score'].clip(0, 100)
    
    return manufacturer_metrics.sort_values('Pricing_Power_Score', ascending=False)

def calculate_manufacturer_volume_scale(df):
    manufacturer_metrics = df.groupby('Manufacturer').agg({
        'MAT Q3 2024 Standard Units': 'sum',
        'MAT Q3 2022 Standard Units': 'sum',
        'MAT Q3 2024 USD MNF': 'sum',
        'Molecule': 'nunique'
    }).reset_index()
    
    manufacturer_metrics['Volume_Growth'] = manufacturer_metrics.apply(
        lambda x: calculate_growth_rate(x['MAT Q3 2024 Standard Units'], x['MAT Q3 2022 Standard Units']),
        axis=1
    )
    
    total_volume_2024 = manufacturer_metrics['MAT Q3 2024 Standard Units'].sum()
    manufacturer_metrics['Volume_Share'] = (
        manufacturer_metrics['MAT Q3 2024 Standard Units'] / total_volume_2024 * 100
    )
    
    manufacturer_metrics['Volume_Scale_Score'] = (
        manufacturer_metrics['Volume_Share'] * 0.4 +
        manufacturer_metrics['Volume_Growth'].clip(0, 50) * 0.3 +
        manufacturer_metrics['Molecule'].clip(0, 20) * 0.3
    )
    
    manufacturer_metrics['Volume_Scale_Score'] = manufacturer_metrics['Volume_Scale_Score'].clip(0, 100)
    
    return manufacturer_metrics.sort_values('Volume_Scale_Score', ascending=False)

def detect_margin_erosion(df):
    manufacturer_metrics = df.groupby('Manufacturer').agg({
        'MAT Q3 2022 USD MNF': 'sum',
        'MAT Q3 2024 USD MNF': 'sum',
        'MAT Q3 2022 Standard Units': 'sum',
        'MAT Q3 2024 Standard Units': 'sum',
        'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
        'MAT Q3 2024 SU Avg Price USD MNF': 'mean'
    }).reset_index()
    
    manufacturer_metrics['Unit_Margin_2022'] = (
        manufacturer_metrics['MAT Q3 2022 USD MNF'] / manufacturer_metrics['MAT Q3 2022 Standard Units']
    )
    
    manufacturer_metrics['Unit_Margin_2024'] = (
        manufacturer_metrics['MAT Q3 2024 USD MNF'] / manufacturer_metrics['MAT Q3 2024 Standard Units']
    )
    
    manufacturer_metrics['Margin_Change'] = manufacturer_metrics.apply(
        lambda x: calculate_growth_rate(x['Unit_Margin_2024'], x['Unit_Margin_2022']),
        axis=1
    )
    
    manufacturer_metrics['Erosion_Flag'] = manufacturer_metrics['Margin_Change'] < -10
    
    erosion_manufacturers = manufacturer_metrics[manufacturer_metrics['Erosion_Flag']].sort_values('Margin_Change')
    
    return erosion_manufacturers

def detect_molecule_dependency_risk(df):
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
    
    high_risk = top_molecule_dependency[top_molecule_dependency['Dependency_Risk']].sort_values('Molecule_Share', ascending=False)
    
    return high_risk

def analyze_su_vs_unit_price_divergence(df):
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
    
    high_divergence = price_analysis[price_analysis['Price_Divergence'] > 20].sort_values('Price_Divergence', ascending=False)
    
    return high_divergence

def analyze_pack_size_optimization(df):
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
    
    return {
        'premium_packs': premium_packs,
        'discount_packs': discount_packs,
        'all_packs': pack_analysis
    }

def detect_hidden_discounting(df):
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
    
    hidden_discounting = manufacturer_pricing[
        manufacturer_pricing['Hidden_Discount_Signal']
    ].sort_values('Price_Realization_Rate')
    
    return hidden_discounting

def generate_executive_summary(df):
    decomp_2324 = calculate_price_volume_decomposition(df, 2024, 2023)
    decomp_2223 = calculate_price_volume_decomposition(df, 2023, 2022)
    
    specialty_premium_2024 = analyze_specialty_premium(df, 2024)
    specialty_premium_2022 = analyze_specialty_premium(df, 2022)
    
    concentration_2024 = calculate_portfolio_concentration(df, 2024)
    
    exits = detect_product_exits(df)
    
    fragility = analyze_growth_fragility(df)
    
    summary = {
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
    
    return summary

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

def filter_dataframe(df, filters):
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
                
                filters['year'] = st.sidebar.selectbox(
                    '📅 Odak Yılı',
                    options=[2024, 2023, 2022],
                    index=0
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
                    st.markdown('<div class="section-header">📊 Executive Summary - Yönetici Özeti</div>', unsafe_allow_html=True)
                    
                    summary = generate_executive_summary(filtered_df)
                    
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
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        risk_level = "Yüksek" if conc['top_3'] > 50 else ("Orta" if conc['top_3'] > 30 else "Düşük")
                        risk_color = "danger" if conc['top_3'] > 50 else ("warning" if conc['top_3'] > 30 else "success")
                        
                        st.markdown(
                            create_insight_box(
                                "Yoğunlaşma Analizi",
                                f"Portföy yoğunlaşma riski: <strong>{risk_level}</strong><br>"
                                f"Top 1 Molekül: {conc['top_1']:.1f}%<br>"
                                f"Top 5 Molekül: {conc['top_5']:.1f}%<br>"
                                f"HHI İndeksi: {conc['hhi']:.0f}",
                                box_type=risk_color
                            ),
                            unsafe_allow_html=True
                        )
                    
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
                                ],
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown(
                            create_insight_box(
                                "Kırılganlık Bileşenleri",
                                f"Hacim Bağımlılığı: {frag['volume_dependency']:.1f}%<br>"
                                f"En Büyük Molekül Bağımlılığı: {frag['top_molecule_dependency']:.1f}%<br>"
                                f"Yoğunlaşma Riski: {frag['concentration_risk']:.1f}%<br>"
                                f"Fiyat Volatilitesi: {frag['price_volatility']:.1f}%",
                                box_type="warning"
                            ),
                            unsafe_allow_html=True
                        )
                    
                    st.markdown("---")
                    
                    st.markdown('<div class="section-header">🚪 Ürün Çıkışları (2022 → 2024)</div>', unsafe_allow_html=True)
                    
                    if summary['product_exits_count'] > 0:
                        exits = detect_product_exits(filtered_df)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = px.bar(
                                exits.head(10),
                                x='Molecule',
                                y='Value_2022',
                                title='En Büyük 10 Çıkan Ürün (2022 Değeri)',
                                color='Value_2022',
                                color_continuous_scale='Reds'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            total_exit_value = exits['Value_2022'].sum()
                            st.markdown(
                                create_metric_card(
                                    "Toplam Çıkan Ürün",
                                    f"{len(exits)}",
                                    prefix="",
                                    suffix=" molekül"
                                ),
                                unsafe_allow_html=True
                            )
                            
                            st.markdown(
                                create_metric_card(
                                    "Kayıp Değer",
                                    f"${total_exit_value/1e6:.1f}M",
                                    prefix="",
                                    suffix=""
                                ),
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("Seçili filtrelerde ürün çıkışı tespit edilmedi.")
                    
                    st.markdown("---")
                    
                    st.markdown('<div class="section-header">📝 Fiyat vs Hacim Büyüme Ayrıştırması</div>', unsafe_allow_html=True)
                    
                    decomp_data = []
                    for year_pair in [(2023, 2022), (2024, 2023)]:
                        decomp = calculate_price_volume_decomposition(filtered_df, year_pair[0], year_pair[1])
                        decomp_data.append({
                            'Dönem': f'{year_pair[1]}-{year_pair[0]}',
                            'Fiyat Etkisi': decomp['price_effect_pct'],
                            'Hacim Etkisi': decomp['volume_effect_pct'],
                            'Mix Etkisi': decomp['mix_effect_pct']
                        })
                    
                    df_decomp = pd.DataFrame(decomp_data)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Fiyat Etkisi',
                        x=df_decomp['Dönem'],
                        y=df_decomp['Fiyat Etkisi'],
                        marker_color='#1a4d7a'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Hacim Etkisi',
                        x=df_decomp['Dönem'],
                        y=df_decomp['Hacim Etkisi'],
                        marker_color='#28a745'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Mix Etkisi',
                        x=df_decomp['Dönem'],
                        y=df_decomp['Mix Etkisi'],
                        marker_color='#ffc107'
                    ))
                    
                    fig.update_layout(
                        barmode='stack',
                        title='Büyüme Ayrıştırması: Fiyat vs Hacim vs Mix',
                        xaxis_title='Dönem',
                        yaxis_title='Katkı (%)',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tabs[1]:
                    st.markdown('<div class="section-header">🔍 Temel Pazar Kaymaları</div>', unsafe_allow_html=True)
                    
                    st.markdown("### 🔄 Büyüme Motoru Tersine Dönüşü")
                    
                    reversal = detect_growth_engine_reversal(filtered_df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        reversal_data = pd.DataFrame([
                            {
                                'Dönem': '2022-2023',
                                'Fiyat Etkisi': reversal['price_effect_2223'],
                                'Hacim Etkisi': reversal['volume_effect_2223']
                            },
                            {
                                'Dönem': '2023-2024',
                                'Fiyat Etkisi': reversal['price_effect_2324'],
                                'Hacim Etkisi': reversal['volume_effect_2324']
                            }
                        ])
                        
                        fig = px.bar(
                            reversal_data,
                            x='Dönem',
                            y=['Fiyat Etkisi', 'Hacim Etkisi'],
                            title='Büyüme Motoru Değişimi',
                            barmode='group',
                            color_discrete_map={'Fiyat Etkisi': '#1a4d7a', 'Hacim Etkisi': '#28a745'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if reversal['reversal_detected']:
                            st.markdown(
                                create_insight_box(
                                    "⚠️ Tersine Dönüş Tespit Edildi",
                                    f"<strong>{reversal['reversal_type']}</strong> yönünde bir değişim gözlemlendi.<br><br>"
                                    f"2022-2023: Fiyat {reversal['price_effect_2223']:.1f}%, Hacim {reversal['volume_effect_2223']:.1f}%<br>"
                                    f"2023-2024: Fiyat {reversal['price_effect_2324']:.1f}%, Hacim {reversal['volume_effect_2324']:.1f}%",
                                    box_type="danger"
                                ),
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                create_insight_box(
                                    "✅ Tutarlı Büyüme Stratejisi",
                                    "Büyüme motorunda önemli bir tersine dönüş tespit edilmedi. Pazar tutarlı bir strateji izlemektedir.",
                                    box_type="success"
                                ),
                                unsafe_allow_html=True
                            )
                    
                    st.markdown("---")
                    
                    st.markdown("### 🚪 Kanal Göçü: NON SPECIALTY → SPECIALTY")
                    
                    migration = analyze_channel_migration(filtered_df)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=migration['df_migration']['Year'],
                            y=migration['df_migration']['Specialty_Share'],
                            name='SPECIALTY',
                            mode='lines+markers',
                            line=dict(color='#1a4d7a', width=3),
                            marker=dict(size=10)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=migration['df_migration']['Year'],
                            y=migration['df_migration']['Non_Specialty_Share'],
                            name='NON SPECIALTY',
                            mode='lines+markers',
                            line=dict(color='#dc3545', width=3),
                            marker=dict(size=10)
                        ))
                        
                        fig.update_layout(
                            title='Specialty vs Non-Specialty Pazar Payı Trendi',
                            xaxis_title='Yıl',
                            yaxis_title='Pazar Payı (%)',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        trend_direction = "artış" if migration['migration_trend'] > 0 else "azalış"
                        trend_color = "success" if migration['migration_trend'] > 0 else "danger"
                        
                        st.markdown(
                            create_metric_card(
                                "Specialty Pay Değişimi",
                                f"{abs(migration['migration_trend']):.1f}%",
                                delta=migration['migration_trend'],
                                delta_text=trend_direction
                            ),
                            unsafe_allow_html=True
                        )
                        
                        st.markdown(
                            create_insight_box(
                                "Kanal Göçü Analizi",
                                f"2022-2024 döneminde Specialty ürünlerin payı {abs(migration['migration_trend']):.1f} puan {trend_direction} gösterdi.<br><br>"
                                f"2024 Specialty Payı: {migration['specialty_share_2024']:.1f}%",
                                box_type=trend_color
                            ),
                            unsafe_allow_html=True
                        )
                    
                    st.markdown("---")
                    
                    st.markdown("### 🌍 Bölgesel Kutuplaşma")
                    
                    regional = analyze_regional_polarization(filtered_df)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = px.bar(
                            regional['df_regional'].sort_values('Growth_2022_2024', ascending=False),
                            x='Region',
                            y='Growth_2022_2024',
                            title='Bölgesel Büyüme Oranları (2022-2024)',
                            color='Growth_2022_2024',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown(
                            create_metric_card(
                                "Kutuplaşma Skoru",
                                f"{regional['polarization_score']:.1f}",
                                delta=regional['polarization_score'],
                                delta_text="volatilite"
                            ),
                            unsafe_allow_html=True
                        )
                        
                        st.markdown(
                            create_insight_box(
                                "Bölgesel Dinamikler",
                                f"Büyüme standart sapması: {regional['growth_std']:.1f}%<br><br>"
                                "Yüksek kutuplaşma, bazı bölgelerin çok güçlü büyürken diğerlerinin geride kaldığını gösterir.",
                                box_type="info"
                            ),
                            unsafe_allow_html=True
                        )
                    
                    st.markdown("---")
                    
                    st.markdown("### 🏢 Top 3 Şirket Pazar Payı Değişimi")
                    
                    corp_shift = analyze_top_corporations_shift(filtered_df)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        top_corps = corp_shift['df_corp'][corp_shift['df_corp']['Rank'] <= 3]
                        
                        fig = px.line(
                            top_corps,
                            x='Year',
                            y='Market_Share',
                            color='Corporation',
                            title='Top 3 Şirket Pazar Payı Trendi',
                            markers=True
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown(
                            create_metric_card(
                                "Top 3 Pay Değişimi",
                                f"{abs(corp_shift['concentration_change']):.1f}%",
                                delta=corp_shift['concentration_change'],
                                delta_text="değişim"
                            ),
                            unsafe_allow_html=True
                        )
                        
                        st.markdown(
                            create_insight_box(
                                "Pazar Konsolidasyonu",
                                f"2022 Top 3 Payı: {corp_shift['top_3_share_2022']:.1f}%<br>"
                                f"2024 Top 3 Payı: {corp_shift['top_3_share_2024']:.1f}%<br><br>"
                                f"Konsolidasyon yönü: {'Artan' if corp_shift['concentration_change'] > 0 else 'Azalan'}",
                                box_type="info"
                            ),
                            unsafe_allow_html=True
                        )
                
                with tabs[2]:
                    st.markdown('<div class="section-header">⚗️ Molekül Zekâsı</div>', unsafe_allow_html=True)
                    
                    st.markdown("### 📈 Yapısal Büyüyen Moleküller")
                    
                    structural = identify_structural_growth_molecules(filtered_df)
                    
                    if len(structural) > 0:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = px.scatter(
                                structural.head(20),
                                x='Volume_CAGR',
                                y='CAGR',
                                size='MAT Q3 2024 USD MNF',
                                color='CAGR',
                                hover_data=['Molecule'],
                                title='Yapısal Büyüyen Moleküller (CAGR vs Hacim CAGR)',
                                labels={'Volume_CAGR': 'Hacim CAGR (%)', 'CAGR': 'Değer CAGR (%)'},
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown(
                                create_metric_card(
                                    "Yapısal Büyüyen Molekül",
                                    f"{len(structural)}",
                                    prefix="",
                                    suffix=" adet"
                                ),
                                unsafe_allow_html=True
                            )
                            
                            st.markdown("#### Top 5 Molekül")
                            for idx, row in structural.head(5).iterrows():
                                st.markdown(f"""
                                <div style='background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #1a4d7a;'>
                                    <strong>{row['Molecule']}</strong><br>
                                    CAGR: {row['CAGR']:.1f}% | Hacim CAGR: {row['Volume_CAGR']:.1f}%
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("#### Detaylı Tablo")
                        st.dataframe(
                            structural.head(20)[['Molecule', 'CAGR', 'Volume_CAGR', 'MAT Q3 2024 USD MNF', 'Growth_2223', 'Growth_2324']],
                            use_container_width=True
                        )
                    else:
                        st.info("Seçili filtrelerde yapısal büyüyen molekül tespit edilmedi.")
                    
                    st.markdown("---")
                    
                    st.markdown("### 🔄 Relaunch Tespiti")
                    
                    relaunch = detect_relaunch_signals(filtered_df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🆕 Relaunch Adayları")
                        
                        if len(relaunch['relaunch_candidates']) > 0:
                            st.dataframe(
                                relaunch['relaunch_candidates'].head(10)[
                                    ['Molecule', 'Chemical Salt', 'International Strength', 'MAT Q3 2024 USD MNF', 'Manufacturer']
                                ],
                                use_container_width=True
                            )
                            
                            st.markdown(
                                create_insight_box(
                                    "Relaunch Sinyalleri",
                                    f"{len(relaunch['relaunch_candidates'])} ürün 2022'de piyasada yokken 2024'te önemli değer yaratıyor.",
                                    box_type="info"
                                ),
                                unsafe_allow_html=True
                            )
                        else:
                            st.info("Relaunch adayı tespit edilmedi.")
                    
                    with col2:
                        st.markdown("#### 🔬 Yüksek Varyasyon Molekülleri")
                        
                        if len(relaunch['high_variation_molecules']) > 0:
                            fig = px.bar(
                                relaunch['high_variation_molecules'].head(10),
                                x='Molecule',
                                y=['Strength_Variants', 'Salt_Variants', 'Pack_Variants'],
                                title='Ürün Varyasyonu En Yüksek Moleküller',
                                barmode='group'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Yüksek varyasyonlu molekül tespit edilmedi.")
                    
                    st.markdown("---")
                    
                    st.markdown("### 📉 Doygunluk ve Emtialaşma Sinyalleri")
                    
                    saturation = detect_saturation_commoditization(filtered_df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### ⏸️ Doygunluk Sinyalleri")
                        
                        if len(saturation['saturation_molecules']) > 0:
                            st.dataframe(
                                saturation['saturation_molecules'].head(10)[
                                    ['Molecule', 'Volume_Growth', 'Value_Growth', 'MAT Q3 2024 USD MNF']
                                ],
                                use_container_width=True
                            )
                            
                            st.markdown(
                                create_insight_box(
                                    "Doygunluk Analizi",
                                    f"{len(saturation['saturation_molecules'])} molekül düşük hacim ve değer büyümesi gösteriyor. "
                                    "Bu moleküller pazar doygunluğuna ulaşmış olabilir.",
                                    box_type="warning"
                                ),
                                unsafe_allow_html=True
                            )
                        else:
                            st.success("Doygunluk sinyali tespit edilmedi.")
                    
                    with col2:
                        st.markdown("#### 💸 Emtialaşma Sinyalleri")
                        
                        if len(saturation['commoditization_molecules']) > 0:
                            st.dataframe(
                                saturation['commoditization_molecules'].head(10)[
                                    ['Molecule', 'Price_Change', 'Manufacturer', 'MAT Q3 2024 USD MNF']
                                ],
                                use_container_width=True
                            )
                            
                            st.markdown(
                                create_insight_box(
                                    "Emtialaşma Analizi",
                                    f"{len(saturation['commoditization_molecules'])} molekül fiyat düşüşü ve yüksek üretici sayısı gösteriyor. "
                                    "Bu moleküller emtialaşma riski taşımaktadır.",
                                    box_type="danger"
                                ),
                                unsafe_allow_html=True
                            )
                        else:
                            st.success("Emtialaşma sinyali tespit edilmedi.")
                
                with tabs[3]:
                    st.markdown('<div class="section-header">🏭 Üretici Skorlama Sistemi</div>', unsafe_allow_html=True)
                    
                    st.markdown("### 💪 Pricing Power Score")
                    
                    pricing_power = calculate_manufacturer_pricing_power(filtered_df)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = px.bar(
                            pricing_power.head(15),
                            x='Manufacturer',
                            y='Pricing_Power_Score',
                            title='Üreticilerin Fiyatlama Gücü Skoru',
                            color='Pricing_Power_Score',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(height=500, xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Top 5 Üretici")
                        for idx, row in pricing_power.head(5).iterrows():
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #1a4d7a 0%, #0f2847 100%); color: white; padding: 15px; margin: 5px 0; border-radius: 8px;'>
                                <strong style='font-size: 16px;'>{row['Manufacturer']}</strong><br>
                                <span style='font-size: 24px; font-weight: 700;'>{row['Pricing_Power_Score']:.1f}</span><br>
                                <small>Fiyat Büyüme: {row['Price_Growth']:.1f}%</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("#### Detaylı Metrikler")
                    st.dataframe(
                        pricing_power.head(20)[
                            ['Manufacturer', 'Pricing_Power_Score', 'Price_Growth', 'Volume_Growth', 'Value_Growth']
                        ],
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    
                    st.markdown("### 📊 Volume Scale Score")
                    
                    volume_scale = calculate_manufacturer_volume_scale(filtered_df)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = px.scatter(
                            volume_scale.head(20),
                            x='Volume_Share',
                            y='Volume_Growth',
                            size='Volume_Scale_Score',
                            color='Volume_Scale_Score',
                            hover_data=['Manufacturer'],
                            title='Hacim Ölçeği: Pazar Payı vs Büyüme',
                            labels={'Volume_Share': 'Pazar Payı (%)', 'Volume_Growth': 'Hacim Büyümesi (%)'},
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Top 5 Ölçek Lideri")
                        for idx, row in volume_scale.head(5).iterrows():
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%); color: white; padding: 15px; margin: 5px 0; border-radius: 8px;'>
                                <strong style='font-size: 16px;'>{row['Manufacturer']}</strong><br>
                                <span style='font-size: 24px; font-weight: 700;'>{row['Volume_Scale_Score']:.1f}</span><br>
                                <small>Pazar Payı: {row['Volume_Share']:.1f}%</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    st.markdown("### ⚠️ Margin Erosion Flag")
                    
                    margin_erosion = detect_margin_erosion(filtered_df)
                    
                    if len(margin_erosion) > 0:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = px.bar(
                                margin_erosion.head(15),
                                x='Manufacturer',
                                y='Margin_Change',
                                title='Marj Erozyonu (2022-2024)',
                                color='Margin_Change',
                                color_continuous_scale='Reds_r'
                            )
                            fig.update_layout(height=400, xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown(
                                create_metric_card(
                                    "Marj Erozyonu Yaşayan Üretici",
                                    f"{len(margin_erosion)}",
                                    prefix="",
                                    suffix=" firma"
                                ),
                                unsafe_allow_html=True
                            )
                            
                            st.markdown(
                                create_insight_box(
                                    "Marj Baskısı",
                                    f"Ortalama marj değişimi: {margin_erosion['Margin_Change'].mean():.1f}%<br>"
                                    f"En kötü durum: {margin_erosion['Margin_Change'].min():.1f}%",
                                    box_type="danger"
                                ),
                                unsafe_allow_html=True
                            )
                        
                        st.dataframe(
                            margin_erosion.head(15)[
                                ['Manufacturer', 'Unit_Margin_2022', 'Unit_Margin_2024', 'Margin_Change']
                            ],
                            use_container_width=True
                        )
                    else:
                        st.success("Önemli marj erozyonu tespit edilmedi.")
                    
                    st.markdown("---")
                    
                    st.markdown("### 🎯 Tek Moleküle Aşırı Bağımlılık Riski")
                    
                    dependency = detect_molecule_dependency_risk(filtered_df)
                    
                    if len(dependency) > 0:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = px.bar(
                                dependency.head(15),
                                x='Manufacturer',
                                y='Molecule_Share',
                                title='En Bağımlı Molekülün Payı (%)',
                                color='Molecule_Share',
                                hover_data=['Molecule'],
                                color_continuous_scale='OrRd'
                            )
                            fig.update_layout(height=400, xaxis_tickangle=-45)
                            fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Risk Eşiği: 30%")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown(
                                create_metric_card(
                                    "Yüksek Risk Üretici",
                                    f"{len(dependency)}",
                                    prefix="",
                                    suffix=" firma"
                                ),
                                unsafe_allow_html=True
                            )
                            
                            st.markdown(
                                create_insight_box(
                                    "Konsantrasyon Riski",
                                    f"Ortalama bağımlılık: {dependency['Molecule_Share'].mean():.1f}%<br>"
                                    f"Maksimum bağımlılık: {dependency['Molecule_Share'].max():.1f}%",
                                    box_type="warning"
                                ),
                                unsafe_allow_html=True
                            )
                        
                        st.dataframe(
                            dependency.head(15)[['Manufacturer', 'Molecule', 'Molecule_Share', 'MAT Q3 2024 USD MNF']],
                            use_container_width=True
                        )
                    else:
                        st.success("Kritik molekül bağımlılığı tespit edilmedi.")
                
                with tabs[4]:
                    st.markdown('<div class="section-header">💰 Fiyat & Mix Analizi</div>', unsafe_allow_html=True)
                    
                    st.markdown("### 📊 SU Avg Price vs Unit Avg Price Ayrışması")
                    
                    price_divergence = analyze_su_vs_unit_price_divergence(filtered_df)
                    
                    if len(price_divergence) > 0:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = px.scatter(
                                price_divergence.head(30),
                                x='MAT Q3 2024 SU Avg Price USD MNF',
                                y='MAT Q3 2024 Unit Avg Price USD MNF',
                                size='Price_Divergence',
                                color='Price_Divergence',
                                hover_data=['Molecule'],
                                title='Fiyat Ayrışması: SU vs Unit Price',
                                color_continuous_scale='Turbo'
                            )
                            fig.add_trace(go.Scatter(
                                x=[0, price_divergence['MAT Q3 2024 SU Avg Price USD MNF'].max()],
                                y=[0, price_divergence['MAT Q3 2024 SU Avg Price USD MNF'].max()],
                                mode='lines',
                                name='Perfect Match',
                                line=dict(color='red', dash='dash')
                            ))
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown(
                                create_metric_card(
                                    "Yüksek Ayrışma",
                                    f"{len(price_divergence)}",
                                    prefix="",
                                    suffix=" molekül"
                                ),
                                unsafe_allow_html=True
                            )
                            
                            st.markdown(
                                create_insight_box(
                                    "Fiyat Dinamikleri",
                                    f"Ortalama ayrışma: {price_divergence['Price_Divergence'].mean():.1f}%<br>"
                                    f"Maksimum ayrışma: {price_divergence['Price_Divergence'].max():.1f}%<br><br>"
                                    "Yüksek ayrışma, pack-size stratejisi veya mix değişikliklerini gösterir.",
                                    box_type="info"
                                ),
                                unsafe_allow_html=True
                            )
                        
                        st.dataframe(
                            price_divergence.head(20)[
                                ['Molecule', 'MAT Q3 2024 SU Avg Price USD MNF', 'MAT Q3 2024 Unit Avg Price USD MNF', 
                                 'Price_Divergence', 'MAT Q3 2024 USD MNF']
                            ],
                            use_container_width=True
                        )
                    else:
                        st.info("Önemli fiyat ayrışması tespit edilmedi.")
                    
                    st.markdown("---")
                    
                    st.markdown("### 📦 Pack-Size Optimizasyonu")
                    
                    pack_optimization = analyze_pack_size_optimization(filtered_df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 💎 Premium Fiyatlı Paketler")
                        
                        if len(pack_optimization['premium_packs']) > 0:
                            st.dataframe(
                                pack_optimization['premium_packs'].head(15)[
                                    ['Molecule', 'International Pack', 'International Size', 'Price_Index', 'MAT Q3 2024 USD MNF']
                                ],
                                use_container_width=True
                            )
                        else:
                            st.info("Premium paket tespit edilmedi.")
                    
                    with col2:
                        st.markdown("#### 💸 İndirimli Paketler")
                        
                        if len(pack_optimization['discount_packs']) > 0:
                            st.dataframe(
                                pack_optimization['discount_packs'].head(15)[
                                    ['Molecule', 'International Pack', 'International Size', 'Price_Index', 'MAT Q3 2024 USD MNF']
                                ],
                                use_container_width=True
                            )
                        else:
                            st.info("İndirimli paket tespit edilmedi.")
                    
                    st.markdown("---")
                    
                    st.markdown("### 🔍 Gizli İskonto Tespiti")
                    
                    hidden_discount = detect_hidden_discounting(filtered_df)
                    
                    if len(hidden_discount) > 0:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = px.bar(
                                hidden_discount.head(15),
                                x='Manufacturer',
                                y='Price_Realization_Rate',
                                title='Fiyat Gerçekleşme Oranı (Hidden Discounting)',
                                color='Price_Realization_Rate',
                                color_continuous_scale='RdYlGn'
                            )
                            fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Eşik: 95%")
                            fig.update_layout(height=400, xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown(
                                create_metric_card(
"Gizli İskonto Sinyali",
f"{len(hidden_discount)}",
prefix="",
suffix=" üretici"
),
unsafe_allow_html=True
)

st.markdown(
                            create_insight_box(
                                "İskonto Analizi",
                                f"Ortalama gerçekleşme: {hidden_discount['Price_Realization_Rate'].mean():.1f}%<br>"
                                f"En düşük gerçekleşme: {hidden_discount['Price_Realization_Rate'].min():.1f}%<br><br>"
                                "%95'in altındaki gerçekleşme oranları gizli iskonto sinyali verir.",
                                box_type="warning"
                            ),
                            unsafe_allow_html=True
                        )
                    
                    st.dataframe(
                        hidden_discount.head(15)[
                            ['Manufacturer', 'Realized_SU_Price', 'Reported_SU_Price', 
                             'Price_Realization_Rate', 'MAT Q3 2024 USD MNF']
                        ],
                        use_container_width=True
                    )
                else:
                    st.success("Gizli iskonto sinyali tespit edilmedi.")
            
            with tabs[5]:
                st.markdown('<div class="section-header">📈 Detaylı Grafikler</div>', unsafe_allow_html=True)
                
                st.markdown("### 📊 Yıllara Göre Pazar Performansı")
                
                yearly_data = []
                for year in [2022, 2023, 2024]:
                    yearly_data.append({
                        'Yıl': year,
                        'Toplam Değer': filtered_df[f'MAT Q3 {year} USD MNF'].sum(),
                        'Toplam Hacim': filtered_df[f'MAT Q3 {year} Standard Units'].sum(),
                        'Ortalama Fiyat': filtered_df[f'MAT Q3 {year} SU Avg Price USD MNF'].mean()
                    })
                
                df_yearly = pd.DataFrame(yearly_data)
                
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=('Toplam Değer', 'Toplam Hacim', 'Ortalama Fiyat')
                )
                
                fig.add_trace(
                    go.Bar(x=df_yearly['Yıl'], y=df_yearly['Toplam Değer'], name='Değer', marker_color='#1a4d7a'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=df_yearly['Yıl'], y=df_yearly['Toplam Hacim'], name='Hacim', marker_color='#28a745'),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Bar(x=df_yearly['Yıl'], y=df_yearly['Ortalama Fiyat'], name='Fiyat', marker_color='#ffc107'),
                    row=1, col=3
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### 🏢 Şirket Bazında Karşılaştırma")
                
                corp_comparison = filtered_df.groupby('Corporation').agg({
                    'MAT Q3 2024 USD MNF': 'sum',
                    'MAT Q3 2023 USD MNF': 'sum',
                    'MAT Q3 2022 USD MNF': 'sum'
                }).reset_index()
                
                corp_comparison['Growth_2324'] = corp_comparison.apply(
                    lambda x: calculate_growth_rate(x['MAT Q3 2024 USD MNF'], x['MAT Q3 2023 USD MNF']),
                    axis=1
                )
                
                corp_comparison = corp_comparison.sort_values('MAT Q3 2024 USD MNF', ascending=False).head(10)
                
                fig = px.bar(
                    corp_comparison,
                    x='Corporation',
                    y=['MAT Q3 2022 USD MNF', 'MAT Q3 2023 USD MNF', 'MAT Q3 2024 USD MNF'],
                    title='Top 10 Şirket - Yıllık Performans Karşılaştırması',
                    barmode='group'
                )
                fig.update_layout(height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### ⚗️ Molekül Bazında Trend Analizi")
                
                molecule_trend = filtered_df.groupby('Molecule').agg({
                    'MAT Q3 2024 USD MNF': 'sum',
                    'MAT Q3 2023 USD MNF': 'sum',
                    'MAT Q3 2022 USD MNF': 'sum'
                }).reset_index()
                
                molecule_trend = molecule_trend.sort_values('MAT Q3 2024 USD MNF', ascending=False).head(10)
                
                fig = go.Figure()
                
                for idx, row in molecule_trend.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[2022, 2023, 2024],
                        y=[row['MAT Q3 2022 USD MNF'], row['MAT Q3 2023 USD MNF'], row['MAT Q3 2024 USD MNF']],
                        mode='lines+markers',
                        name=row['Molecule'],
                        line=dict(width=3),
                        marker=dict(size=10)
                    ))
                
                fig.update_layout(
                    title='Top 10 Molekül - Trend Analizi',
                    xaxis_title='Yıl',
                    yaxis_title='Değer (USD MNF)',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### 🌍 Bölgesel Dağılım (2024)")
                
                regional_dist = filtered_df.groupby('Region').agg({
                    'MAT Q3 2024 USD MNF': 'sum'
                }).reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        regional_dist,
                        values='MAT Q3 2024 USD MNF',
                        names='Region',
                        title='Bölgesel Pazar Payı Dağılımı'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.treemap(
                        regional_dist,
                        path=['Region'],
                        values='MAT Q3 2024 USD MNF',
                        title='Bölgesel Değer Haritası'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tabs[6]:
                st.markdown('<div class="section-header">📋 Veri Tablosu</div>', unsafe_allow_html=True)
                
                st.markdown("### 🔍 Ham Veri Görünümü")
                
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
                    
                    st.markdown(f"*Gösterilen: İlk 100 satır / Toplam {len(filtered_df):,} satır*")
                    
                    csv = filtered_df[display_columns].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 CSV İndir",
                        data=csv,
                        file_name='pharma_analytics_filtered.csv',
                        mime='text/csv',
                    )
                else:
                    st.info("Lütfen en az bir kolon seçin.")
            
            st.markdown("""
            <div class="footer">
                <strong>Pharma Analytics Intelligence Platform</strong><br>
                Principal Data Engineering & Advanced Market Analytics<br>
                © 2024 - Tüm hakları saklıdır
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

if name == "main":
main()

