import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import gc
import warnings
import re
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import json
import base64
from scipy import stats
from scipy.stats import zscore
import itertools
import functools

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Pharma Analytics Platform | Q3 MAT Analysis",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #0f1a2f;
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f1a2f 0%, #1a2b4d 100%);
    }
    
    .sidebar .sidebar-content {
        background-color: #1a2b4d;
        color: white;
    }
    
    .kpi-card {
        background-color: #1e3a5f;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 15px;
    }
    
    .kpi-title {
        font-size: 14px;
        color: #93c5fd;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .kpi-value {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
    }
    
    .kpi-change {
        font-size: 12px;
        padding: 3px 8px;
        border-radius: 12px;
        display: inline-block;
        margin-top: 5px;
    }
    
    .positive {
        background-color: #065f46;
        color: #34d399;
    }
    
    .negative {
        background-color: #7f1d1d;
        color: #f87171;
    }
    
    .neutral {
        background-color: #374151;
        color: #9ca3af;
    }
    
    .section-header {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        color: white;
        font-weight: bold;
    }
    
    .insight-card {
        background-color: #1f2937;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #374151;
    }
    
    .insight-title {
        color: #60a5fa;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .data-table {
        background-color: #111827;
        border-radius: 8px;
        padding: 15px;
    }
    
    div[data-testid="stExpander"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: #111827;
        color: white;
    }
    
    .stMultiSelect div[data-baseweb="select"] {
        background-color: #111827;
        color: white;
    }
    
    .stSlider div[data-baseweb="slider"] {
        color: #3b82f6;
    }
    
    .stDataFrame {
        background-color: #111827;
    }
    
    .metric-container {
        background-color: #1e3a5f;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .tab-container {
        background-color: #111827;
        border-radius: 8px;
        padding: 20px;
        margin-top: 20px;
    }
    
    .warning-box {
        background-color: #7c2d12;
        border-left: 4px solid #f97316;
        padding: 15px;
        border-radius: 4px;
        margin: 15px 0;
    }
    
    .success-box {
        background-color: #065f46;
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 4px;
        margin: 15px 0;
    }
    
    .info-box {
        background-color: #1e40af;
        border-left: 4px solid #60a5fa;
        padding: 15px;
        border-radius: 4px;
        margin: 15px 0;
    }
    
    .manufacturer-score {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 14px;
    }
    
    .score-high { background-color: #065f46; color: #34d399; }
    .score-medium { background-color: #854d0e; color: #fbbf24; }
    .score-low { background-color: #7f1d1d; color: #f87171; }
    
    .growth-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .growth-high { background-color: #10b981; }
    .growth-medium { background-color: #f59e0b; }
    .growth-low { background-color: #ef4444; }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

@st.cache_resource(show_spinner=False)
def generate_sample_data():
    np.random.seed(42)
    
    sources = ['IQVIA', 'IMS', 'Intercontinental', 'MarketTrack']
    countries = ['USA', 'Germany', 'France', 'UK', 'Japan', 'Brazil', 'China', 'India', 'Italy', 'Spain']
    sectors = ['Oncology', 'Cardiology', 'Neurology', 'Endocrinology', 'Immunology', 'Respiratory']
    panels = ['Retail', 'Hospital', 'Clinic', 'Combined']
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']
    sub_regions = ['Western Europe', 'Eastern Europe', 'North Asia', 'South Asia', 'Andean', 'Southern Cone']
    corporations = ['Pfizer', 'Novartis', 'Roche', 'Merck', 'GSK', 'Sanofi', 'AstraZeneca', 'Johnson & Johnson', 'AbbVie', 'Bristol-Myers Squibb']
    manufacturers = ['Pfizer Inc', 'Novartis Pharma', 'Roche Holding', 'Merck & Co', 'GlaxoSmithKline', 'Sanofi SA', 
                    'AstraZeneca PLC', 'Johnson & Johnson', 'AbbVie Inc', 'Bristol-Myers Squibb', 'Eli Lilly', 'Amgen',
                    'Gilead Sciences', 'Takeda', 'Bayer', 'Boehringer Ingelheim', 'Novo Nordisk', 'Astellas', 'Daiichi Sankyo']
    
    molecules = ['Atorvastatin', 'Levothyroxine', 'Metformin', 'Lisinopril', 'Amlodipine', 'Metoprolol', 'Albuterol',
                'Omeprazole', 'Losartan', 'Simvastatin', 'Gabapentin', 'Hydrochlorothiazide', 'Sertraline',
                'Montelukast', 'Fluticasone', 'Rosuvastatin', 'Escitalopram', 'Bupropion', 'Duloxetine', 'Pregabalin',
                'Insulin Glargine', 'Adalimumab', 'Etanercept', 'Infliximab', 'Rituximab', 'Bevacizumab', 'Trastuzumab',
                'Pembrolizumab', 'Nivolumab', 'Ibrutinib', 'Venetoclax', 'Acalabrutinib', 'Empagliflozin', 'Dapagliflozin',
                'Semaglutide', 'Liraglutide', 'Apixaban', 'Rivaroxaban', 'Dabigatran', 'Edoxaban']
    
    chemical_salts = ['Calcium', 'Sodium', 'Potassium', 'Hydrochloride', 'Sulfate', 'Acetate', 'Citrate', 'Tartrate']
    specialty_products = ['Yes', 'No']
    nfc123_codes = [f'NFC{str(i).zfill(3)}' for i in range(1, 51)]
    
    data = []
    n_rows = 500000
    
    for i in range(n_rows):
        country = np.random.choice(countries)
        region = 'North America' if country == 'USA' else 'Europe' if country in ['Germany', 'France', 'UK', 'Italy', 'Spain'] else 'Asia Pacific' if country in ['Japan', 'China', 'India'] else 'Latin America' if country == 'Brazil' else 'Middle East'
        
        row = {
            'Source.Name': np.random.choice(sources),
            'Country': country,
            'Sector': np.random.choice(sectors),
            'Panel': np.random.choice(panels),
            'Region': region,
            'Sub-Region': np.random.choice(sub_regions),
            'Corporation': np.random.choice(corporations),
            'Manufacturer': np.random.choice(manufacturers),
            'Molecule List': ', '.join(np.random.choice(molecules, size=np.random.randint(1, 4), replace=False)),
            'Molecule': np.random.choice(molecules),
            'Chemical Salt': np.random.choice(chemical_salts),
            'International Product': f'INT{np.random.randint(1000, 9999)}',
            'Specialty Product': np.random.choice(specialty_products, p=[0.3, 0.7]),
            'NFC123': np.random.choice(nfc123_codes),
            'International Pack': np.random.choice(['Bottle', 'Blister', 'Vial', 'Syringe', 'Pen', 'Ampoule']),
            'International Strength': f'{np.random.choice([5, 10, 20, 40, 50, 100, 200, 500])} {np.random.choice(["mg", "mcg", "IU"])}',
            'International Size': np.random.randint(1, 100),
            'International Volume': np.random.choice([1, 2.5, 5, 10, 20, 30, 50, 100, 250, 500]),
            'International Prescription': np.random.choice(['Rx', 'OTC', 'Both']),
        }
        
        base_usd_2022 = np.random.lognormal(mean=5, sigma=2)
        base_units_2022 = np.random.lognormal(mean=8, sigma=1.5)
        
        growth_factor_2023 = np.random.uniform(0.8, 1.3)
        growth_factor_2024 = np.random.uniform(0.7, 1.4)
        
        specialty_multiplier = 2.5 if row['Specialty Product'] == 'Yes' else 1.0
        oncology_multiplier = 3.0 if row['Sector'] == 'Oncology' else 1.0
        usa_multiplier = 1.5 if row['Country'] == 'USA' else 1.0
        
        price_multiplier = specialty_multiplier * oncology_multiplier * usa_multiplier
        
        row.update({
            'MAT Q3 2022 USD MNF': round(base_usd_2022 * price_multiplier * np.random.uniform(0.8, 1.2), 2),
            'MAT Q3 2022 Standard Units': round(base_units_2022 * np.random.uniform(0.9, 1.1)),
            'MAT Q3 2022 Units': round(base_units_2022 * np.random.uniform(0.8, 1.2) * np.random.randint(1, 10)),
            'MAT Q3 2022 SU Avg Price USD MNF': round(price_multiplier * np.random.uniform(0.5, 2.0), 3),
            'MAT Q3 2022 Unit Avg Price USD MNF': round(price_multiplier * np.random.uniform(0.1, 1.0), 3),
            
            'MAT Q3 2023 USD MNF': round(base_usd_2022 * growth_factor_2023 * price_multiplier * np.random.uniform(0.8, 1.2), 2),
            'MAT Q3 2023 Standard Units': round(base_units_2022 * growth_factor_2023 * np.random.uniform(0.9, 1.1)),
            'MAT Q3 2023 Units': round(base_units_2022 * growth_factor_2023 * np.random.uniform(0.8, 1.2) * np.random.randint(1, 10)),
            'MAT Q3 2023 SU Avg Price USD MNF': round(price_multiplier * np.random.uniform(0.5, 2.0) * (1 + np.random.uniform(-0.1, 0.2)), 3),
            'MAT Q3 2023 Unit Avg Price USD MNF': round(price_multiplier * np.random.uniform(0.1, 1.0) * (1 + np.random.uniform(-0.1, 0.2)), 3),
            
            'MAT Q3 2024 USD MNF': round(base_usd_2022 * growth_factor_2023 * growth_factor_2024 * price_multiplier * np.random.uniform(0.8, 1.2), 2),
            'MAT Q3 2024 Standard Units': round(base_units_2022 * growth_factor_2023 * growth_factor_2024 * np.random.uniform(0.9, 1.1)),
            'MAT Q3 2024 Units': round(base_units_2022 * growth_factor_2023 * growth_factor_2024 * np.random.uniform(0.8, 1.2) * np.random.randint(1, 10)),
            'MAT Q3 2024 SU Avg Price USD MNF': round(price_multiplier * np.random.uniform(0.5, 2.0) * (1 + np.random.uniform(-0.2, 0.3)), 3),
            'MAT Q3 2024 Unit Avg Price USD MNF': round(price_multiplier * np.random.uniform(0.1, 1.0) * (1 + np.random.uniform(-0.2, 0.3)), 3),
        })
        
        data.append(row)
        
        if i % 50000 == 0 and i > 0:
            gc.collect()
    
    return pd.DataFrame(data)

@st.cache_data(show_spinner=False, ttl=3600)
def load_data():
    with st.spinner('🚀 Generating 500,000+ rows of pharmaceutical market data...'):
        df = generate_sample_data()
        
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].apply(lambda x: f"{x:,.3f}".replace('.', ',').replace(',', 'X').replace('.', ',').replace('X', '.') 
                                   if np.random.random() < 0.3 else x)
        
        numeric_cols = [
            'MAT Q3 2022 USD MNF', 'MAT Q3 2022 Standard Units', 'MAT Q3 2022 Units',
            'MAT Q3 2022 SU Avg Price USD MNF', 'MAT Q3 2022 Unit Avg Price USD MNF',
            'MAT Q3 2023 USD MNF', 'MAT Q3 2023 Standard Units', 'MAT Q3 2023 Units',
            'MAT Q3 2023 SU Avg Price USD MNF', 'MAT Q3 2023 Unit Avg Price USD MNF',
            'MAT Q3 2024 USD MNF', 'MAT Q3 2024 Standard Units', 'MAT Q3 2024 Units',
            'MAT Q3 2024 SU Avg Price USD MNF', 'MAT Q3 2024 Unit Avg Price USD MNF'
        ]
        
        for col in numeric_cols:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

def compute_growth_metrics(df):
    metrics = {}
    
    total_usd_2022 = df['MAT Q3 2022 USD MNF'].sum()
    total_usd_2023 = df['MAT Q3 2023 USD MNF'].sum()
    total_usd_2024 = df['MAT Q3 2024 USD MNF'].sum()
    
    metrics['total_growth_2023'] = ((total_usd_2023 - total_usd_2022) / total_usd_2022 * 100) if total_usd_2022 > 0 else 0
    metrics['total_growth_2024'] = ((total_usd_2024 - total_usd_2023) / total_usd_2023 * 100) if total_usd_2023 > 0 else 0
    
    specialty_df = df[df['Specialty Product'] == 'Yes']
    non_specialty_df = df[df['Specialty Product'] == 'No']
    
    metrics['specialty_growth_2023'] = ((specialty_df['MAT Q3 2023 USD MNF'].sum() - specialty_df['MAT Q3 2022 USD MNF'].sum()) / 
                                       specialty_df['MAT Q3 2022 USD MNF'].sum() * 100) if specialty_df['MAT Q3 2022 USD MNF'].sum() > 0 else 0
    metrics['non_specialty_growth_2023'] = ((non_specialty_df['MAT Q3 2023 USD MNF'].sum() - non_specialty_df['MAT Q3 2022 USD MNF'].sum()) / 
                                           non_specialty_df['MAT Q3 2022 USD MNF'].sum() * 100) if non_specialty_df['MAT Q3 2022 USD MNF'].sum() > 0 else 0
    
    metrics['specialty_growth_2024'] = ((specialty_df['MAT Q3 2024 USD MNF'].sum() - specialty_df['MAT Q3 2023 USD MNF'].sum()) / 
                                       specialty_df['MAT Q3 2023 USD MNF'].sum() * 100) if specialty_df['MAT Q3 2023 USD MNF'].sum() > 0 else 0
    metrics['non_specialty_growth_2024'] = ((non_specialty_df['MAT Q3 2024 USD MNF'].sum() - non_specialty_df['MAT Q3 2023 USD MNF'].sum()) / 
                                           non_specialty_df['MAT Q3 2023 USD MNF'].sum() * 100) if non_specialty_df['MAT Q3 2023 USD MNF'].sum() > 0 else 0
    
    metrics['specialty_premium_2023'] = (specialty_df['MAT Q3 2023 SU Avg Price USD MNF'].mean() / 
                                        non_specialty_df['MAT Q3 2023 SU Avg Price USD MNF'].mean()) if non_specialty_df['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0
    metrics['specialty_premium_2024'] = (specialty_df['MAT Q3 2024 SU Avg Price USD MNF'].mean() / 
                                        non_specialty_df['MAT Q3 2024 SU Avg Price USD MNF'].mean()) if non_specialty_df['MAT Q3 2024 SU Avg Price USD MNF'].mean() > 0 else 0
    
    top_3_2022 = df.groupby('Corporation')['MAT Q3 2022 USD MNF'].sum().nlargest(3).sum()
    top_3_2023 = df.groupby('Corporation')['MAT Q3 2023 USD MNF'].sum().nlargest(3).sum()
    top_3_2024 = df.groupby('Corporation')['MAT Q3 2024 USD MNF'].sum().nlargest(3).sum()
    
    metrics['concentration_2022'] = (top_3_2022 / total_usd_2022 * 100) if total_usd_2022 > 0 else 0
    metrics['concentration_2023'] = (top_3_2023 / total_usd_2023 * 100) if total_usd_2023 > 0 else 0
    metrics['concentration_2024'] = (top_3_2024 / total_usd_2024 * 100) if total_usd_2024 > 0 else 0
    
    exit_products = df[(df['MAT Q3 2022 USD MNF'] > 10000) & (df['MAT Q3 2023 USD MNF'] == 0)]
    metrics['exit_products_count'] = len(exit_products)
    metrics['exit_products_value'] = exit_products['MAT Q3 2022 USD MNF'].sum()
    
    fragility_score = 0
    if metrics['total_growth_2024'] > 20:
        fragility_score += 30
    if metrics['concentration_2024'] > 60:
        fragility_score += 25
    if metrics['exit_products_count'] > 50:
        fragility_score += 20
    if abs(metrics['specialty_growth_2024'] - metrics['non_specialty_growth_2024']) > 15:
        fragility_score += 25
    metrics['fragility_score'] = min(100, fragility_score)
    
    return metrics

def generate_executive_summary(metrics, df):
    summary_parts = []
    
    summary_parts.append(f"### 📈 Executive Market Summary")
    summary_parts.append(f"**Total Market Growth:** {metrics['total_growth_2024']:+.1f}% in 2024 ({metrics['total_growth_2023']:+.1f}% in 2023)")
    
    if metrics['total_growth_2024'] > metrics['total_growth_2023']:
        summary_parts.append(f"• **Acceleration:** Market growth accelerated by {metrics['total_growth_2024'] - metrics['total_growth_2023']:+.1f}pp")
    else:
        summary_parts.append(f"• **Deceleration:** Market growth slowed by {metrics['total_growth_2023'] - metrics['total_growth_2024']:+.1f}pp")
    
    price_growth_2024 = (df['MAT Q3 2024 SU Avg Price USD MNF'].mean() - df['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / df['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100
    volume_growth_2024 = (df['MAT Q3 2024 Standard Units'].sum() - df['MAT Q3 2023 Standard Units'].sum()) / df['MAT Q3 2023 Standard Units'].sum() * 100
    
    summary_parts.append(f"**Growth Decomposition:**")
    summary_parts.append(f"• Price Contribution: {price_growth_2024:+.1f}%")
    summary_parts.append(f"• Volume Contribution: {volume_growth_2024:+.1f}%")
    
    if price_growth_2024 > volume_growth_2024:
        summary_parts.append(f"• **Price-led growth** dominates market expansion")
    else:
        summary_parts.append(f"• **Volume-led growth** drives market expansion")
    
    summary_parts.append(f"**Specialty Premium:**")
    summary_parts.append(f"• 2024: {metrics['specialty_premium_2024']:.1f}x price multiple")
    summary_parts.append(f"• 2023: {metrics['specialty_premium_2023']:.1f}x price multiple")
    
    premium_change = metrics['specialty_premium_2024'] - metrics['specialty_premium_2023']
    if premium_change > 0.1:
        summary_parts.append(f"• **Warning:** Specialty premium expanding (+{premium_change:.1f}x)")
    elif premium_change < -0.1:
        summary_parts.append(f"• **Opportunity:** Specialty premium contracting ({premium_change:.1f}x)")
    
    summary_parts.append(f"**Portfolio Concentration:**")
    summary_parts.append(f"• Top 3 Corporations: {metrics['concentration_2024']:.1f}% market share")
    concentration_change = metrics['concentration_2024'] - metrics['concentration_2023']
    if concentration_change > 2:
        summary_parts.append(f"• **Risk:** Market concentration increasing (+{concentration_change:.1f}pp)")
    elif concentration_change < -2:
        summary_parts.append(f"• **Improvement:** Market becoming more competitive ({concentration_change:+.1f}pp)")
    
    summary_parts.append(f"**Product Exit Detection:**")
    summary_parts.append(f"• {metrics['exit_products_count']} products exited the market")
    summary_parts.append(f"• ${metrics['exit_products_value']:,.0f} in lost revenue")
    
    summary_parts.append(f"**Growth Fragility Score:** {metrics['fragility_score']}/100")
    if metrics['fragility_score'] > 70:
        summary_parts.append(f"• **🚨 HIGH RISK:** Market growth is highly fragile")
    elif metrics['fragility_score'] > 40:
        summary_parts.append(f"• **⚠️ MEDIUM RISK:** Monitor market stability")
    else:
        summary_parts.append(f"• **✅ LOW RISK:** Market growth appears sustainable")
    
    return "\n\n".join(summary_parts)

def detect_market_shifts(df):
    shifts = []
    
    regional_growth = df.groupby('Region').apply(lambda x: (
        (x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / 
        x['MAT Q3 2023 USD MNF'].sum() * 100 if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0
    )).sort_values(ascending=False)
    
    if len(regional_growth) >= 2:
        fastest = regional_growth.index[0]
        slowest = regional_growth.index[-1]
        gap = regional_growth.iloc[0] - regional_growth.iloc[-1]
        if gap > 15:
            shifts.append(f"**Regional Polarization:** {fastest} growing at {regional_growth.iloc[0]:+.1f}% vs {slowest} at {regional_growth.iloc[-1]:+.1f}% ({gap:.1f}pp gap)")
    
    sector_migration = df.groupby(['Sector', 'Specialty Product']).apply(lambda x: x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()).unstack()
    if 'Yes' in sector_migration.columns and 'No' in sector_migration.columns:
        sector_migration['Shift'] = sector_migration['Yes'] - sector_migration['No']
        major_shift = sector_migration['Shift'].abs().idxmax()
        if sector_migration.loc[major_shift, 'Shift'] > 10000000:
            direction = "toward Specialty" if sector_migration.loc[major_shift, 'Shift'] > 0 else "away from Specialty"
            shifts.append(f"**Channel Migration:** {major_shift} shows major shift {direction} (${abs(sector_migration.loc[major_shift, 'Shift']):,.0f})")
    
    corporation_growth = df.groupby('Corporation').apply(lambda x: (
        (x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / 
        x['MAT Q3 2023 USD MNF'].sum() * 100 if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0
    )).sort_values(ascending=False)
    
    if len(corporation_growth) >= 5:
        top_gainer = corporation_growth.index[0]
        top_loser = corporation_growth.index[-1]
        if corporation_growth.iloc[0] > 20 and corporation_growth.iloc[-1] < -10:
            shifts.append(f"**Growth Divergence:** {top_gainer} growing at {corporation_growth.iloc[0]:+.1f}% while {top_loser} declining at {corporation_growth.iloc[-1]:+.1f}%")
    
    engine_reversal = df.groupby(['Country', 'Specialty Product']).apply(lambda x: x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()).unstack()
    if 'Yes' in engine_reversal.columns and 'No' in engine_reversal.columns:
        engine_reversal['Specialty Share'] = engine_reversal['Yes'] / (engine_reversal['Yes'] + engine_reversal['No'])
        reversal_countries = engine_reversal[engine_reversal['Specialty Share'] > 0.7].index.tolist()
        if reversal_countries:
            shifts.append(f"**Growth Engine Reversal:** {len(reversal_countries)} countries now driven by Specialty (>70% share)")
    
    top_3_2023 = df.groupby('Corporation')['MAT Q3 2023 USD MNF'].sum().nlargest(3).index.tolist()
    top_3_2024 = df.groupby('Corporation')['MAT Q3 2024 USD MNF'].sum().nlargest(3).index.tolist()
    
    changes = set(top_3_2023) ^ set(top_3_2024)
    if changes:
        shifts.append(f"**Top-3 Shakeup:** {', '.join(changes)} entered/exited top 3 rankings")
    
    return shifts

def analyze_molecules(df):
    insights = []
    
    molecule_growth = df.groupby('Molecule').apply(lambda x: pd.Series({
        'growth_2023': ((x['MAT Q3 2023 USD MNF'].sum() - x['MAT Q3 2022 USD MNF'].sum()) / 
                       x['MAT Q3 2022 USD MNF'].sum() * 100) if x['MAT Q3 2022 USD MNF'].sum() > 0 else 0,
        'growth_2024': ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / 
                       x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0,
        'total_2024': x['MAT Q3 2024 USD MNF'].sum()
    }))
    
    structural_growth = molecule_growth[
        (molecule_growth['growth_2023'] > 10) & 
        (molecule_growth['growth_2024'] > 10) &
        (molecule_growth['total_2024'] > 1000000)
    ].sort_values('growth_2024', ascending=False)
    
    if len(structural_growth) > 0:
        top_molecules = structural_growth.head(3).index.tolist()
        insights.append(f"**Structural Growth Molecules:** {', '.join(top_molecules)} showing consistent >10% growth")
    
    relaunch_candidates = []
    for molecule in df['Molecule'].unique():
        molecule_data = df[df['Molecule'] == molecule]
        
        salt_changes = molecule_data.groupby('Chemical Salt')['MAT Q3 2024 USD MNF'].sum().sort_values(ascending=False)
        if len(salt_changes) >= 2:
            top_salt = salt_changes.index[0]
            second_salt = salt_changes.index[1]
            if salt_changes.iloc[0] > salt_changes.iloc[1] * 3:
                relaunch_candidates.append(f"{molecule} ({top_salt} dominating)")
        
        pack_growth = molecule_data.groupby('International Pack').apply(lambda x: 
            (x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / 
            x['MAT Q3 2023 USD MNF'].sum() * 100 if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0
        )
        
        if len(pack_growth) >= 2:
            fastest_pack = pack_growth.idxmax()
            if pack_growth.max() > 50:
                relaunch_candidates.append(f"{molecule} ({fastest_pack} pack +{pack_growth.max():.0f}%)")
    
    if relaunch_candidates:
        insights.append(f"**Relaunch Detection:** " + "; ".join(relaunch_candidates[:5]))
    
    commoditization = molecule_growth[
        (molecule_growth['growth_2024'] < -5) &
        (molecule_growth['total_2024'] > 5000000)
    ].sort_values('growth_2024')
    
    if len(commoditization) > 0:
        top_commod = commoditization.head(3).index.tolist()
        avg_decline = commoditization.head(3)['growth_2024'].mean()
        insights.append(f"**Commoditization Risk:** {', '.join(top_commod)} declining avg {avg_decline:.1f}% despite scale")
    
    saturation_check = molecule_growth[
        (molecule_growth['growth_2024'] < 2) &
        (molecule_growth['growth_2023'] < 2) &
        (molecule_growth['total_2024'] > 10000000)
    ]
    
    if len(saturation_check) > 0:
        saturated = saturation_check.index.tolist()[:3]
        insights.append(f"**Market Saturation:** {', '.join(saturated)} showing <2% growth despite >$10M revenue")
    
    return insights

def score_manufacturers(df):
    scores = []
    
    manufacturer_stats = df.groupby('Manufacturer').apply(lambda x: pd.Series({
        'total_revenue_2024': x['MAT Q3 2024 USD MNF'].sum(),
        'revenue_growth': ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / 
                          x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0,
        'price_power': (x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                      x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100 if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
        'volume_growth': ((x['MAT Q3 2024 Standard Units'].sum() - x['MAT Q3 2023 Standard Units'].sum()) / 
                         x['MAT Q3 2023 Standard Units'].sum() * 100) if x['MAT Q3 2023 Standard Units'].sum() > 0 else 0,
        'product_count': x['International Product'].nunique(),
        'specialty_share': (x[x['Specialty Product'] == 'Yes']['MAT Q3 2024 USD MNF'].sum() / 
                           x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0,
        'top_product_concentration': (x.groupby('International Product')['MAT Q3 2024 USD MNF'].sum().nlargest(1).sum() / 
                                     x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0,
        'margin_erosion': ((x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() - x['MAT Q3 2023 Unit Avg Price USD MNF'].mean()) / 
                          x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() > 0 else 0
    })).reset_index()
    
    manufacturer_stats['pricing_power_score'] = manufacturer_stats['price_power'].apply(
        lambda x: 100 if x > 10 else 80 if x > 5 else 60 if x > 0 else 40 if x > -5 else 20
    )
    
    manufacturer_stats['volume_scale_score'] = manufacturer_stats['total_revenue_2024'].apply(
        lambda x: 100 if x > 50000000 else 80 if x > 20000000 else 60 if x > 5000000 else 40 if x > 1000000 else 20
    )
    
    manufacturer_stats['margin_erosion_flag'] = manufacturer_stats['margin_erosion'].apply(
        lambda x: 'HIGH' if x < -10 else 'MEDIUM' if x < -5 else 'LOW'
    )
    
    manufacturer_stats['dependence_risk'] = manufacturer_stats['top_product_concentration'].apply(
        lambda x: 'HIGH' if x > 50 else 'MEDIUM' if x > 30 else 'LOW'
    )
    
    for _, row in manufacturer_stats.nlargest(10, 'total_revenue_2024').iterrows():
        score_card = {
            'manufacturer': row['Manufacturer'],
            'revenue': f"${row['total_revenue_2024']:,.0f}",
            'growth': f"{row['revenue_growth']:+.1f}%",
            'pricing_score': row['pricing_power_score'],
            'volume_score': row['volume_scale_score'],
            'margin_flag': row['margin_erosion_flag'],
            'dependence_risk': row['dependence_risk'],
            'specialty_share': f"{row['specialty_share']:.1f}%"
        }
        scores.append(score_card)
    
    return scores

def analyze_pricing_mix(df):
    insights = []
    
    price_divergence = df.groupby('Molecule').apply(lambda x: pd.Series({
        'su_price_growth': ((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                           x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
        'unit_price_growth': ((x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() - x['MAT Q3 2023 Unit Avg Price USD MNF'].mean()) / 
                             x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() > 0 else 0,
        'revenue': x['MAT Q3 2024 USD MNF'].sum()
    }))
    
    large_divergence = price_divergence[
        (abs(price_divergence['su_price_growth'] - price_divergence['unit_price_growth']) > 10) &
        (price_divergence['revenue'] > 1000000)
    ]
    
    if len(large_divergence) > 0:
        top_div = large_divergence.nlargest(3, 'revenue')
        for idx, row in top_div.iterrows():
            diff = row['su_price_growth'] - row['unit_price_growth']
            insight = f"{idx}: SU price {row['su_price_growth']:+.1f}% vs Unit price {row['unit_price_growth']:+.1f}% ({diff:+.1f}pp gap)"
            insights.append(insight)
    
    pack_analysis = df.groupby(['International Pack', 'International Size']).apply(lambda x: pd.Series({
        'avg_price_per_unit': x['MAT Q3 2024 Unit Avg Price USD MNF'].mean(),
        'total_units': x['MAT Q3 2024 Units'].sum(),
        'revenue': x['MAT Q3 2024 USD MNF'].sum(),
        'price_per_size': x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() / x['International Size'].mean() if x['International Size'].mean() > 0 else 0
    })).reset_index()
    
    pack_analysis['efficiency_score'] = pack_analysis['price_per_size'].rank(pct=True) * 100
    
    optimization_candidates = pack_analysis[
        (pack_analysis['efficiency_score'] < 30) &
        (pack_analysis['revenue'] > 500000)
    ].sort_values('efficiency_score')
    
    if len(optimization_candidates) > 0:
        top_candidates = optimization_candidates.head(3)
        for _, row in top_candidates.iterrows():
            insights.append(f"Pack Optimization: {row['International Pack']} {row['International Size']}x - Low efficiency score {row['efficiency_score']:.0f}")
    
    discount_analysis = df.groupby(['Manufacturer', 'Country']).apply(lambda x: pd.Series({
        'price_ratio': x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() / x['MAT Q3 2024 SU Avg Price USD MNF'].mean() if x['MAT Q3 2024 SU Avg Price USD MNF'].mean() > 0 else 0,
        'revenue': x['MAT Q3 2024 USD MNF'].sum(),
        'std_dev': x['MAT Q3 2024 Unit Avg Price USD MNF'].std()
    })).reset_index()
    
    hidden_discounts = discount_analysis[
        (discount_analysis['price_ratio'] < 0.5) &
        (discount_analysis['revenue'] > 1000000) &
        (discount_analysis['std_dev'] > discount_analysis['std_dev'].quantile(0.75))
    ]
    
    if len(hidden_discounts) > 0:
        top_discounts = hidden_discounts.nlargest(3, 'revenue')
        for _, row in top_discounts.iterrows():
            insights.append(f"Hidden Discounting: {row['Manufacturer']} in {row['Country']} - Unit price only {row['price_ratio']:.1%} of SU price")
    
    mix_shift = df.groupby('Specialty Product').apply(lambda x: pd.Series({
        'share_2023': x['MAT Q3 2023 USD MNF'].sum() / df['MAT Q3 2023 USD MNF'].sum() * 100,
        'share_2024': x['MAT Q3 2024 USD MNF'].sum() / df['MAT Q3 2024 USD MNF'].sum() * 100
    })).reset_index()
    
    if len(mix_shift) == 2:
        specialty_shift = mix_shift[mix_shift['Specialty Product'] == 'Yes']
        if not specialty_shift.empty:
            shift = specialty_shift['share_2024'].iloc[0] - specialty_shift['share_2023'].iloc[0]
            if abs(shift) > 2:
                insights.append(f"Mix Shift: Specialty share changed {shift:+.1f}pp (2023: {specialty_shift['share_2023'].iloc[0]:.1f}% → 2024: {specialty_shift['share_2024'].iloc[0]:.1f}%)")
    
    return insights

def create_kpi_card(title, value, change=None, change_label=None):
    if change is not None:
        change_class = "positive" if change > 0 else "negative" if change < 0 else "neutral"
        change_display = f'<span class="kpi-change {change_class}">{change:+.1f}% {change_label if change_label else ""}</span>'
    else:
        change_display = ""
    
    return f'''
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {change_display}
    </div>
    '''

def render_manufacturer_scores(scores):
    html = '<div class="data-table"><h4>🏭 Top Manufacturer Scores</h4><table style="width:100%; border-collapse: collapse;">'
    html += '<tr style="background-color: #1e3a5f;"><th>Manufacturer</th><th>Revenue 2024</th><th>Growth</th><th>Pricing Power</th><th>Volume Scale</th><th>Margin Risk</th><th>Dependence Risk</th><th>Specialty Share</th></tr>'
    
    for score in scores:
        pricing_class = "score-high" if score['pricing_score'] >= 80 else "score-medium" if score['pricing_score'] >= 60 else "score-low"
        volume_class = "score-high" if score['volume_score'] >= 80 else "score-medium" if score['volume_score'] >= 60 else "score-low"
        margin_color = "#ef4444" if score['margin_flag'] == 'HIGH' else "#f59e0b" if score['margin_flag'] == 'MEDIUM' else "#10b981"
        depend_color = "#ef4444" if score['dependence_risk'] == 'HIGH' else "#f59e0b" if score['dependence_risk'] == 'MEDIUM' else "#10b981"
        
        html += f'''
        <tr style="border-bottom: 1px solid #374151;">
            <td style="padding: 10px;"><strong>{score['manufacturer']}</strong></td>
            <td style="padding: 10px;">{score['revenue']}</td>
            <td style="padding: 10px;">{score['growth']}</td>
            <td style="padding: 10px;"><div class="manufacturer-score {pricing_class}">{score['pricing_score']}</div></td>
            <td style="padding: 10px;"><div class="manufacturer-score {volume_class}">{score['volume_score']}</div></td>
            <td style="padding: 10px;"><span style="color:{margin_color}">●</span> {score['margin_flag']}</td>
            <td style="padding: 10px;"><span style="color:{depend_color}">●</span> {score['dependence_risk']}</td>
            <td style="padding: 10px;">{score['specialty_share']}</td>
        </tr>
        '''
    
    html += '</table></div>'
    return html

def create_growth_decomposition_chart(df):
    total_growth_2024 = ((df['MAT Q3 2024 USD MNF'].sum() - df['MAT Q3 2023 USD MNF'].sum()) / 
                        df['MAT Q3 2023 USD MNF'].sum() * 100) if df['MAT Q3 2023 USD MNF'].sum() > 0 else 0
    
    price_effect = ((df['MAT Q3 2024 SU Avg Price USD MNF'].mean() - df['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                   df['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if df['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0
    
    volume_effect = ((df['MAT Q3 2024 Standard Units'].sum() - df['MAT Q3 2023 Standard Units'].sum()) / 
                    df['MAT Q3 2023 Standard Units'].sum() * 100) if df['MAT Q3 2023 Standard Units'].sum() > 0 else 0
    
    mix_effect = total_growth_2024 - price_effect - volume_effect
    
    fig = go.Figure(data=[
        go.Bar(
            name='Price Effect',
            x=['Growth Contribution'],
            y=[price_effect],
            marker_color='#3b82f6',
            text=f'{price_effect:+.1f}%',
            textposition='auto',
        ),
        go.Bar(
            name='Volume Effect',
            x=['Growth Contribution'],
            y=[volume_effect],
            marker_color='#10b981',
            text=f'{volume_effect:+.1f}%',
            textposition='auto',
        ),
        go.Bar(
            name='Mix Effect',
            x=['Growth Contribution'],
            y=[mix_effect],
            marker_color='#8b5cf6',
            text=f'{mix_effect:+.1f}%',
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='2024 Growth Decomposition',
        barmode='stack',
        showlegend=True,
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font_color='white',
        height=400,
    )
    
    return fig

def create_regional_growth_chart(df):
    regional_growth = df.groupby('Region').apply(lambda x: pd.Series({
        'growth_2024': ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / 
                       x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0,
        'share_2024': (x['MAT Q3 2024 USD MNF'].sum() / df['MAT Q3 2024 USD MNF'].sum() * 100) if df['MAT Q3 2024 USD MNF'].sum() > 0 else 0
    })).reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Growth Rate by Region', 'Market Share by Region'),
        specs=[[{'type': 'bar'}, {'type': 'pie'}]]
    )
    
    colors = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444']
    
    fig.add_trace(
        go.Bar(
            x=regional_growth['Region'],
            y=regional_growth['growth_2024'],
            marker_color=colors[:len(regional_growth)],
            text=regional_growth['growth_2024'].apply(lambda x: f'{x:+.1f}%'),
            textposition='auto',
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(
            labels=regional_growth['Region'],
            values=regional_growth['share_2024'],
            marker_colors=colors[:len(regional_growth)],
            textinfo='label+percent',
            hole=0.4,
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Regional Analysis',
        showlegend=False,
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font_color='white',
        height=500,
    )
    
    return fig

def create_molecule_heatmap(df):
    top_molecules = df.groupby('Molecule')['MAT Q3 2024 USD MNF'].sum().nlargest(15).index.tolist()
    top_countries = df.groupby('Country')['MAT Q3 2024 USD MNF'].sum().nlargest(10).index.tolist()
    
    filtered_df = df[(df['Molecule'].isin(top_molecules)) & (df['Country'].isin(top_countries))]
    
    heatmap_data = filtered_df.groupby(['Molecule', 'Country'])['MAT Q3 2024 USD MNF'].sum().unstack()
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis',
        colorbar=dict(title="USD MNF"),
        text=heatmap_data.values,
        texttemplate='%{text:.2s}',
        textfont={"color": "white"}
    ))
    
    fig.update_layout(
        title='Top Molecules by Country (Heatmap)',
        xaxis_title='Country',
        yaxis_title='Molecule',
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font_color='white',
        height=600,
    )
    
    return fig

def create_price_volume_scatter(df):
    manufacturer_stats = df.groupby('Manufacturer').apply(lambda x: pd.Series({
        'price_growth': ((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                        x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
        'volume_growth': ((x['MAT Q3 2024 Standard Units'].sum() - x['MAT Q3 2023 Standard Units'].sum()) / 
                         x['MAT Q3 2023 Standard Units'].sum() * 100) if x['MAT Q3 2023 Standard Units'].sum() > 0 else 0,
        'revenue_2024': x['MAT Q3 2024 USD MNF'].sum(),
        'specialty_share': (x[x['Specialty Product'] == 'Yes']['MAT Q3 2024 USD MNF'].sum() / 
                           x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0
    })).reset_index()
    
    top_20 = manufacturer_stats.nlargest(20, 'revenue_2024')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=top_20['price_growth'],
        y=top_20['volume_growth'],
        mode='markers+text',
        marker=dict(
            size=top_20['revenue_2024'] / top_20['revenue_2024'].max() * 50 + 10,
            color=top_20['specialty_share'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Specialty Share %")
        ),
        text=top_20['Manufacturer'],
        textposition='top center',
        hovertemplate='<b>%{text}</b><br>Price Growth: %{x:.1f}%<br>Volume Growth: %{y:.1f}%<br>Revenue: $%{marker.size:.2s}<br>Specialty Share: %{marker.color:.1f}%<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.add_annotation(x=10, y=10, text="Premium Growers", showarrow=False, font=dict(color="#10b981"))
    fig.add_annotation(x=-10, y=10, text="Volume Drivers", showarrow=False, font=dict(color="#3b82f6"))
    fig.add_annotation(x=-10, y=-10, text="Declining", showarrow=False, font=dict(color="#ef4444"))
    fig.add_annotation(x=10, y=-10, text="Price Increase/Volume Decline", showarrow=False, font=dict(color="#f59e0b"))
    
    fig.update_layout(
        title='Manufacturer Price vs Volume Growth Strategy',
        xaxis_title='Price Growth (%)',
        yaxis_title='Volume Growth (%)',
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font_color='white',
        height=600,
    )
    
    return fig

def create_specialty_analysis_chart(df):
    specialty_analysis = df.groupby(['Country', 'Specialty Product']).apply(lambda x: pd.Series({
        'revenue_2024': x['MAT Q3 2024 USD MNF'].sum(),
        'growth': ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / 
                  x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0
    })).unstack()
    
    top_countries = df.groupby('Country')['MAT Q3 2024 USD MNF'].sum().nlargest(8).index.tolist()
    
    specialty_analysis = specialty_analysis.loc[top_countries]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Specialty vs Non-Specialty Revenue 2024', 'Growth Comparison'),
        vertical_spacing=0.15
    )
    
    fig.add_trace(
        go.Bar(
            name='Specialty',
            x=top_countries,
            y=specialty_analysis[('revenue_2024', 'Yes')],
            marker_color='#8b5cf6',
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            name='Non-Specialty',
            x=top_countries,
            y=specialty_analysis[('revenue_2024', 'No')],
            marker_color='#3b82f6',
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            name='Specialty Growth',
            x=top_countries,
            y=specialty_analysis[('growth', 'Yes')],
            mode='lines+markers',
            line=dict(color='#10b981', width=3),
            marker=dict(size=10),
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            name='Non-Specialty Growth',
            x=top_countries,
            y=specialty_analysis[('growth', 'No')],
            mode='lines+markers',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=10),
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Specialty Product Analysis by Top Countries',
        barmode='stack',
        showlegend=True,
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font_color='white',
        height=700,
    )
    
    return fig

def main():
    st.title("💊 Pharmaceutical Analytics Platform")
    st.markdown("### Q3 MAT Analysis | 500,000+ Records | Real-time Analytics")
    
    df = load_data()
    
    st.sidebar.markdown("## 🔍 Filter Controls")
    
    all_countries = ['All'] + sorted(df['Country'].unique().tolist())
    selected_country = st.sidebar.selectbox("Country", all_countries)
    
    all_corporations = ['All'] + sorted(df['Corporation'].unique().tolist())
    selected_corporation = st.sidebar.selectbox("Corporation", all_corporations)
    
    all_manufacturers = ['All'] + sorted(df['Manufacturer'].unique().tolist())
    selected_manufacturer = st.sidebar.selectbox("Manufacturer", all_manufacturers)
    
    all_molecules = ['All'] + sorted(df['Molecule'].unique().tolist())
    selected_molecule = st.sidebar.selectbox("Molecule", all_molecules)
    
    all_specialty = ['All'] + sorted(df['Specialty Product'].unique().tolist())
    selected_specialty = st.sidebar.selectbox("Specialty Product", all_specialty)
    
    year_options = ['2022', '2023', '2024', 'All']
    selected_year = st.sidebar.selectbox("Year", year_options)
    
    filtered_df = df.copy()
    
    if selected_country != 'All':
        filtered_df = filtered_df[filtered_df['Country'] == selected_country]
    
    if selected_corporation != 'All':
        filtered_df = filtered_df[filtered_df['Corporation'] == selected_corporation]
    
    if selected_manufacturer != 'All':
        filtered_df = filtered_df[filtered_df['Manufacturer'] == selected_manufacturer]
    
    if selected_molecule != 'All':
        filtered_df = filtered_df[filtered_df['Molecule'] == selected_molecule]
    
    if selected_specialty != 'All':
        filtered_df = filtered_df[filtered_df['Specialty Product'] == selected_specialty]
    
    st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,}")
    st.sidebar.markdown(f"**Data Coverage:** {len(df):,} total rows")
    
    if st.sidebar.button("🔄 Reset All Filters"):
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    
    total_revenue_2024 = filtered_df['MAT Q3 2024 USD MNF'].sum()
    total_revenue_2023 = filtered_df['MAT Q3 2023 USD MNF'].sum()
    growth_rate = ((total_revenue_2024 - total_revenue_2023) / total_revenue_2023 * 100) if total_revenue_2023 > 0 else 0
    
    st.sidebar.metric("2024 Revenue", f"${total_revenue_2024:,.0f}", f"{growth_rate:+.1f}%")
    st.sidebar.metric("Products", f"{filtered_df['International Product'].nunique():,}")
    st.sidebar.metric("Manufacturers", f"{filtered_df['Manufacturer'].nunique():,}")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Executive Dashboard", 
        "🔍 Market Intelligence", 
        "🧬 Molecule Analytics",
        "🏭 Manufacturer Scoring",
        "💰 Pricing & Mix",
        "📊 Data Explorer"
    ])
    
    with tab1:
        st.markdown('<div class="section-header">📊 EXECUTIVE SUMMARY & KPI DASHBOARD</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_2024 = filtered_df['MAT Q3 2024 USD MNF'].sum()
            total_2023 = filtered_df['MAT Q3 2023 USD MNF'].sum()
            growth = ((total_2024 - total_2023) / total_2023 * 100) if total_2023 > 0 else 0
            st.markdown(create_kpi_card("Total Revenue 2024", f"${total_2024:,.0f}", growth, "YoY"), unsafe_allow_html=True)
        
        with col2:
            specialty_rev = filtered_df[filtered_df['Specialty Product'] == 'Yes']['MAT Q3 2024 USD MNF'].sum()
            specialty_share = (specialty_rev / total_2024 * 100) if total_2024 > 0 else 0
            prev_share = (filtered_df[filtered_df['Specialty Product'] == 'Yes']['MAT Q3 2023 USD MNF'].sum() / 
                         total_2023 * 100) if total_2023 > 0 else 0
            share_change = specialty_share - prev_share
            st.markdown(create_kpi_card("Specialty Share", f"{specialty_share:.1f}%", share_change, "YoY"), unsafe_allow_html=True)
        
        with col3:
            top_3_share = (filtered_df.groupby('Corporation')['MAT Q3 2024 USD MNF'].sum().nlargest(3).sum() / 
                          total_2024 * 100) if total_2024 > 0 else 0
            prev_top_3 = (filtered_df.groupby('Corporation')['MAT Q3 2023 USD MNF'].sum().nlargest(3).sum() / 
                         total_2023 * 100) if total_2023 > 0 else 0
            concentration_change = top_3_share - prev_top_3
            st.markdown(create_kpi_card("Top 3 Concentration", f"{top_3_share:.1f}%", concentration_change, "YoY"), unsafe_allow_html=True)
        
        with col4:
            avg_price_growth = ((filtered_df['MAT Q3 2024 SU Avg Price USD MNF'].mean() - 
                               filtered_df['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                              filtered_df['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if filtered_df['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0
            st.markdown(create_kpi_card("Avg Price Growth", f"{avg_price_growth:+.1f}%"), unsafe_allow_html=True)
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            volume_growth = ((filtered_df['MAT Q3 2024 Standard Units'].sum() - 
                            filtered_df['MAT Q3 2023 Standard Units'].sum()) / 
                           filtered_df['MAT Q3 2023 Standard Units'].sum() * 100) if filtered_df['MAT Q3 2023 Standard Units'].sum() > 0 else 0
            st.markdown(create_kpi_card("Volume Growth", f"{volume_growth:+.1f}%"), unsafe_allow_html=True)
        
        with col6:
            exit_count = len(filtered_df[(filtered_df['MAT Q3 2023 USD MNF'] > 0) & (filtered_df['MAT Q3 2024 USD MNF'] == 0)])
            st.markdown(create_kpi_card("Product Exits", f"{exit_count}"), unsafe_allow_html=True)
        
        with col7:
            molecule_count = filtered_df['Molecule'].nunique()
            st.markdown(create_kpi_card("Active Molecules", f"{molecule_count}"), unsafe_allow_html=True)
        
        with col8:
            fragility_score = compute_growth_metrics(filtered_df)['fragility_score']
            st.markdown(create_kpi_card("Growth Fragility", f"{fragility_score}/100"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.plotly_chart(create_growth_decomposition_chart(filtered_df), use_container_width=True)
            
            st.plotly_chart(create_regional_growth_chart(filtered_df), use_container_width=True)
        
        with col_right:
            metrics = compute_growth_metrics(filtered_df)
            summary = generate_executive_summary(metrics, filtered_df)
            st.markdown(summary)
            
            st.markdown("---")
            
            st.markdown("### 🚨 Key Market Shifts")
            shifts = detect_market_shifts(filtered_df)
            for shift in shifts[:5]:
                st.markdown(f'<div class="insight-card"><div class="insight-title">🔍 {shift}</div></div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="section-header">🔍 MARKET INTELLIGENCE & TREND ANALYSIS</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_specialty_analysis_chart(filtered_df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_price_volume_scatter(filtered_df), use_container_width=True)
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 🌍 Regional Performance")
            regional_perf = filtered_df.groupby('Region').apply(lambda x: pd.Series({
                'Revenue 2024': f"${x['MAT Q3 2024 USD MNF'].sum():,.0f}",
                'Growth': f"{((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / x['MAT Q3 2023 USD MNF'].sum() * 100):+.1f}%" if x['MAT Q3 2023 USD MNF'].sum() > 0 else "N/A",
                'Share': f"{(x['MAT Q3 2024 USD MNF'].sum() / filtered_df['MAT Q3 2024 USD MNF'].sum() * 100):.1f}%" if filtered_df['MAT Q3 2024 USD MNF'].sum() > 0 else "N/A",
                'Price Trend': f"{((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100):+.1f}%" if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else "N/A"
            })).reset_index()
            
            st.dataframe(regional_perf.style.background_gradient(subset=['Growth'], cmap='RdYlGn'), use_container_width=True)
        
        with col4:
            st.markdown("### 🏢 Corporation Rankings")
            corp_rankings = filtered_df.groupby('Corporation').apply(lambda x: pd.Series({
                'Rank': 0,
                'Revenue 2024': x['MAT Q3 2024 USD MNF'].sum(),
                'Growth': ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0,
                'Specialty %': (x[x['Specialty Product'] == 'Yes']['MAT Q3 2024 USD MNF'].sum() / x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0,
                'Top Molecule': x.groupby('Molecule')['MAT Q3 2024 USD MNF'].sum().idxmax() if len(x) > 0 else ''
            })).reset_index()
            
            corp_rankings = corp_rankings.sort_values('Revenue 2024', ascending=False)
            corp_rankings['Rank'] = range(1, len(corp_rankings) + 1)
            
            st.dataframe(corp_rankings.head(10).style.format({
                'Revenue 2024': '${:,.0f}',
                'Growth': '{:+.1f}%',
                'Specialty %': '{:.1f}%'
            }), use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 📈 Sector Performance Analysis")
        sector_analysis = filtered_df.groupby('Sector').apply(lambda x: pd.Series({
            'Revenue 2024': x['MAT Q3 2024 USD MNF'].sum(),
            'Growth vs 2023': ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0,
            'Avg Price 2024': x['MAT Q3 2024 SU Avg Price USD MNF'].mean(),
            'Price Growth': ((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
            'Specialty Intensity': (x[x['Specialty Product'] == 'Yes']['MAT Q3 2024 USD MNF'].sum() / x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0
        })).reset_index()
        
        sector_analysis = sector_analysis.sort_values('Revenue 2024', ascending=False)
        
        fig_sector = px.bar(
            sector_analysis,
            x='Sector',
            y='Revenue 2024',
            color='Growth vs 2023',
            title='Sector Revenue & Growth',
            color_continuous_scale='RdYlGn',
            text=sector_analysis['Growth vs 2023'].apply(lambda x: f'{x:+.1f}%')
        )
        
        fig_sector.update_layout(
            plot_bgcolor='#1f2937',
            paper_bgcolor='#1f2937',
            font_color='white',
            height=500,
        )
        
        st.plotly_chart(fig_sector, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="section-header">🧬 MOLECULE INTELLIGENCE & PORTFOLIO ANALYSIS</div>', unsafe_allow_html=True)
        
        st.plotly_chart(create_molecule_heatmap(filtered_df), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚀 Growth Champions (Top 10)")
            molecule_growth = filtered_df.groupby('Molecule').apply(lambda x: pd.Series({
                'Revenue 2024': x['MAT Q3 2024 USD MNF'].sum(),
                'Growth 2024 vs 2023': ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0,
                'Growth 2023 vs 2022': ((x['MAT Q3 2023 USD MNF'].sum() - x['MAT Q3 2022 USD MNF'].sum()) / x['MAT Q3 2022 USD MNF'].sum() * 100) if x['MAT Q3 2022 USD MNF'].sum() > 0 else 0,
                'Avg Price 2024': x['MAT Q3 2024 SU Avg Price USD MNF'].mean(),
                'Manufacturers': x['Manufacturer'].nunique()
            })).reset_index()
            
            growth_champions = molecule_growth[
                (molecule_growth['Growth 2024 vs 2023'] > 10) &
                (molecule_growth['Revenue 2024'] > 1000000)
            ].sort_values('Growth 2024 vs 2023', ascending=False).head(10)
            
            if len(growth_champions) > 0:
                st.dataframe(growth_champions.style.format({
                    'Revenue 2024': '${:,.0f}',
                    'Growth 2024 vs 2023': '{:+.1f}%',
                    'Growth 2023 vs 2022': '{:+.1f}%',
                    'Avg Price 2024': '${:.3f}'
                }).background_gradient(subset=['Growth 2024 vs 2023'], cmap='RdYlGn'), use_container_width=True)
            else:
                st.info("No molecules meeting growth champion criteria")
        
        with col2:
            st.markdown("### ⚠️ Declining Molecules (Top 10)")
            declining_molecules = molecule_growth[
                (molecule_growth['Growth 2024 vs 2023'] < -5) &
                (molecule_growth['Revenue 2024'] > 500000)
            ].sort_values('Growth 2024 vs 2023').head(10)
            
            if len(declining_molecules) > 0:
                st.dataframe(declining_molecules.style.format({
                    'Revenue 2024': '${:,.0f}',
                    'Growth 2024 vs 2023': '{:+.1f}%',
                    'Growth 2023 vs 2022': '{:+.1f}%',
                    'Avg Price 2024': '${:.3f}'
                }).background_gradient(subset=['Growth 2024 vs 2023'], cmap='RdYlGn_r'), use_container_width=True)
            else:
                st.info("No significant declining molecules detected")
        
        st.markdown("---")
        
        st.markdown("### 🧪 Molecule Insights")
        insights = analyze_molecules(filtered_df)
        
        for insight in insights:
            st.markdown(f'<div class="insight-card"><div class="insight-title">💡 {insight}</div></div>', unsafe_allow_html=True)
        
        if not insights:
            st.info("No significant molecule insights detected with current filters")
        
        st.markdown("---")
        
        st.markdown("### 📦 Formulation & Salt Analysis")
        salt_analysis = filtered_df.groupby(['Molecule', 'Chemical Salt']).apply(lambda x: pd.Series({
            'Revenue 2024': x['MAT Q3 2024 USD MNF'].sum(),
            'Growth': ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0,
            'Avg Price': x['MAT Q3 2024 SU Avg Price USD MNF'].mean(),
            'Market Share %': (x['MAT Q3 2024 USD MNF'].sum() / filtered_df[filtered_df['Molecule'] == x.name[0]]['MAT Q3 2024 USD MNF'].sum() * 100) if filtered_df[filtered_df['Molecule'] == x.name[0]]['MAT Q3 2024 USD MNF'].sum() > 0 else 0
        })).reset_index()
        
        top_salts = salt_analysis.sort_values('Revenue 2024', ascending=False).head(15)
        
        if len(top_salts) > 0:
            st.dataframe(top_salts.style.format({
                'Revenue 2024': '${:,.0f}',
                'Growth': '{:+.1f}%',
                'Avg Price': '${:.3f}',
                'Market Share %': '{:.1f}%'
            }), use_container_width=True)
    
    with tab4:
        st.markdown('<div class="section-header">🏭 MANUFACTURER SCORING & RISK ASSESSMENT</div>', unsafe_allow_html=True)
        
        scores = score_manufacturers(filtered_df)
        
        st.markdown(render_manufacturer_scores(scores), unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Pricing Power Analysis")
            pricing_power = filtered_df.groupby('Manufacturer').apply(lambda x: pd.Series({
                'Price Growth %': ((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                                  x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
                'Avg Price 2024': x['MAT Q3 2024 SU Avg Price USD MNF'].mean(),
                'Price Premium vs Market': (x['MAT Q3 2024 SU Avg Price USD MNF'].mean() / 
                                           filtered_df['MAT Q3 2024 SU Avg Price USD MNF'].mean() - 1) * 100 if filtered_df['MAT Q3 2024 SU Avg Price USD MNF'].mean() > 0 else 0,
                'Specialty Concentration %': (x[x['Specialty Product'] == 'Yes']['MAT Q3 2024 USD MNF'].sum() / 
                                             x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0
            })).reset_index()
            
            top_pricing = pricing_power.nlargest(10, 'Price Growth %')
            
            fig_pricing = px.bar(
                top_pricing,
                x='Manufacturer',
                y='Price Growth %',
                color='Specialty Concentration %',
                title='Top 10 Manufacturers by Pricing Power',
                color_continuous_scale='Viridis',
                text=top_pricing['Price Growth %'].apply(lambda x: f'{x:+.1f}%')
            )
            
            fig_pricing.update_layout(
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font_color='white',
                height=500,
            )
            
            st.plotly_chart(fig_pricing, use_container_width=True)
        
        with col2:
            st.markdown("### ⚠️ Margin Erosion Risks")
            margin_risk = filtered_df.groupby('Manufacturer').apply(lambda x: pd.Series({
                'Unit Price Change %': ((x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() - x['MAT Q3 2023 Unit Avg Price USD MNF'].mean()) / 
                                       x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() > 0 else 0,
                'SU Price Change %': ((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                                     x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
                'Divergence': (((x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() - x['MAT Q3 2023 Unit Avg Price USD MNF'].mean()) / 
                              x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() * 100) - 
                             ((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                              x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100)) if x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() > 0 and x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
                'Revenue 2024': x['MAT Q3 2024 USD MNF'].sum()
            })).reset_index()
            
            high_risk = margin_risk[
                (margin_risk['Unit Price Change %'] < -5) &
                (margin_risk['Revenue 2024'] > 1000000)
            ].sort_values('Unit Price Change %').head(10)
            
            if len(high_risk) > 0:
                st.dataframe(high_risk.style.format({
                    'Unit Price Change %': '{:+.1f}%',
                    'SU Price Change %': '{:+.1f}%',
                    'Divergence': '{:+.1f}pp',
                    'Revenue 2024': '${:,.0f}'
                }).background_gradient(subset=['Unit Price Change %'], cmap='RdYlGn_r'), use_container_width=True)
            else:
                st.success("No significant margin erosion risks detected")
        
        st.markdown("---")
        
        st.markdown("### 🎯 Dependence Risk Analysis")
        dependence_analysis = filtered_df.groupby('Manufacturer').apply(lambda x: pd.Series({
            'Top Product Share %': (x.groupby('International Product')['MAT Q3 2024 USD MNF'].sum().max() / 
                                   x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0,
            'Top 3 Products Share %': (x.groupby('International Product')['MAT Q3 2024 USD MNF'].sum().nlargest(3).sum() / 
                                      x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0,
            'Molecule Concentration %': (x.groupby('Molecule')['MAT Q3 2024 USD MNF'].sum().max() / 
                                        x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0,
            'Country Concentration %': (x.groupby('Country')['MAT Q3 2024 USD MNF'].sum().max() / 
                                       x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0,
            'Risk Score': 0
        })).reset_index()
        
        dependence_analysis['Risk Score'] = (
            dependence_analysis['Top Product Share %'] * 0.4 +
            dependence_analysis['Top 3 Products Share %'] * 0.3 +
            dependence_analysis['Molecule Concentration %'] * 0.2 +
            dependence_analysis['Country Concentration %'] * 0.1
        )
        
        high_dependence = dependence_analysis[dependence_analysis['Risk Score'] > 50].sort_values('Risk Score', ascending=False)
        
        if len(high_dependence) > 0:
            st.warning(f"**🚨 High Dependence Risk Detected:** {len(high_dependence)} manufacturers with risk score > 50")
            
            fig_risk = px.scatter(
                high_dependence,
                x='Top Product Share %',
                y='Molecule Concentration %',
                size='Risk Score',
                color='Risk Score',
                hover_name='Manufacturer',
                title='Manufacturer Dependence Risk Map',
                color_continuous_scale='RdYlGn_r'
            )
            
            fig_risk.update_layout(
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font_color='white',
                height=500,
            )
            
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.info("No manufacturers with critical dependence risks detected")
    
    with tab5:
        st.markdown('<div class="section-header">💰 PRICING & MIX ANALYSIS</div>', unsafe_allow_html=True)
        
        pricing_insights = analyze_pricing_mix(filtered_df)
        
        if pricing_insights:
            st.markdown("### 🔍 Pricing Intelligence")
            for insight in pricing_insights[:10]:
                st.markdown(f'<div class="insight-card"><div class="insight-title">💰 {insight}</div></div>', unsafe_allow_html=True)
        else:
            st.info("No significant pricing insights detected with current filters")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Price vs Volume Matrix")
            
            matrix_data = filtered_df.groupby(['Molecule', 'Specialty Product']).apply(lambda x: pd.Series({
                'Price Growth %': ((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                                  x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
                'Volume Growth %': ((x['MAT Q3 2024 Standard Units'].sum() - x['MAT Q3 2023 Standard Units'].sum()) / 
                                   x['MAT Q3 2023 Standard Units'].sum() * 100) if x['MAT Q3 2023 Standard Units'].sum() > 0 else 0,
                'Revenue 2024': x['MAT Q3 2024 USD MNF'].sum()
            })).reset_index()
            
            top_matrix = matrix_data.nlargest(20, 'Revenue 2024')
            
            fig_matrix = px.scatter(
                top_matrix,
                x='Price Growth %',
                y='Volume Growth %',
                size='Revenue 2024',
                color='Specialty Product',
                hover_name='Molecule',
                title='Price vs Volume Growth Matrix',
                color_discrete_map={'Yes': '#8b5cf6', 'No': '#3b82f6'}
            )
            
            fig_matrix.update_layout(
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font_color='white',
                height=500,
            )
            
            fig_matrix.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_matrix.add_vline(x=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig_matrix, use_container_width=True)
        
        with col2:
            st.markdown("### 📦 Pack Size Optimization")
            
            pack_analysis = filtered_df.groupby(['International Pack', 'International Size']).apply(lambda x: pd.Series({
                'Avg Price per Unit': x['MAT Q3 2024 Unit Avg Price USD MNF'].mean(),
                'Price per Size Unit': x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() / x['International Size'].mean() if x['International Size'].mean() > 0 else 0,
                'Total Units': x['MAT Q3 2024 Units'].sum(),
                'Revenue': x['MAT Q3 2024 USD MNF'].sum(),
                'Efficiency Ratio': (x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() / x['International Size'].mean()) / 
                                  filtered_df['MAT Q3 2024 Unit Avg Price USD MNF'].mean() * filtered_df['International Size'].mean() if filtered_df['MAT Q3 2024 Unit Avg Price USD MNF'].mean() > 0 and filtered_df['International Size'].mean() > 0 else 0
            })).reset_index()
            
            pack_analysis = pack_analysis[pack_analysis['Revenue'] > 10000].sort_values('Efficiency Ratio', ascending=False)
            
            if len(pack_analysis) > 0:
                st.dataframe(pack_analysis.head(15).style.format({
                    'Avg Price per Unit': '${:.3f}',
                    'Price per Size Unit': '${:.3f}',
                    'Total Units': '{:,.0f}',
                    'Revenue': '${:,.0f}',
                    'Efficiency Ratio': '{:.2f}x'
                }).background_gradient(subset=['Efficiency Ratio'], cmap='RdYlGn'), use_container_width=True)
            else:
                st.info("Insufficient data for pack size analysis")
        
        st.markdown("---")
        
        st.markdown("### 💸 Hidden Discounting Detection")
        
        discount_detection = filtered_df.groupby(['Manufacturer', 'Country']).apply(lambda x: pd.Series({
            'Unit/SU Price Ratio': x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() / x['MAT Q3 2024 SU Avg Price USD MNF'].mean() if x['MAT Q3 2024 SU Avg Price USD MNF'].mean() > 0 else 0,
            'Ratio Change vs 2023': ((x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() / x['MAT Q3 2024 SU Avg Price USD MNF'].mean()) - 
                                    (x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() / x['MAT Q3 2023 SU Avg Price USD MNF'].mean())) if x['MAT Q3 2024 SU Avg Price USD MNF'].mean() > 0 and x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 and x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
            'Revenue 2024': x['MAT Q3 2024 USD MNF'].sum(),
            'Price Variance': x['MAT Q3 2024 Unit Avg Price USD MNF'].std() / x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() if x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() > 0 else 0
        })).reset_index()
        
        hidden_discounts = discount_detection[
            (discount_detection['Unit/SU Price Ratio'] < 0.7) &
            (discount_detection['Revenue 2024'] > 500000) &
            (discount_detection['Price Variance'] > 0.3)
        ].sort_values('Unit/SU Price Ratio')
        
        if len(hidden_discounts) > 0:
            st.warning(f"**⚠️ Potential Hidden Discounting Detected:** {len(hidden_discounts)} manufacturer-country pairs")
            
            fig_discount = px.bar(
                hidden_discounts.head(15),
                x='Manufacturer',
                y='Unit/SU Price Ratio',
                color='Country',
                title='Lowest Unit/SU Price Ratios (Potential Discounting)',
                text=hidden_discounts.head(15)['Unit/SU Price Ratio'].apply(lambda x: f'{x:.2f}')
            )
            
            fig_discount.update_layout(
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font_color='white',
                height=500,
            )
            
            st.plotly_chart(fig_discount, use_container_width=True)
        else:
            st.success("No significant hidden discounting patterns detected")
    
    with tab6:
        st.markdown('<div class="section-header">📊 DATA EXPLORER & RAW DATA</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_rows = st.selectbox("Show Rows", [100, 500, 1000, 5000, 10000], index=0)
        
        with col2:
            sort_by = st.selectbox("Sort By", ['MAT Q3 2024 USD MNF', 'MAT Q3 2023 USD MNF', 'MAT Q3 2022 USD MNF', 
                                              'MAT Q3 2024 SU Avg Price USD MNF', 'MAT Q3 2024 Standard Units'], index=0)
        
        with col3:
            sort_order = st.selectbox("Sort Order", ['Descending', 'Ascending'], index=0)
        
        display_df = filtered_df.sort_values(sort_by, ascending=(sort_order == 'Ascending')).head(show_rows)
        
        st.dataframe(display_df, use_container_width=True)
        
        st.markdown("---")
        
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="pharma_data_filtered.csv" class="stButton">📥 Download Filtered Data (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col_download2:
            sample_data = filtered_df.head(10000).to_csv(index=False)
            b64_sample = base64.b64encode(sample_data.encode()).decode()
            href_sample = f'<a href="data:file/csv;base64,{b64_sample}" download="pharma_data_sample.csv" class="stButton">📥 Download Sample (10k rows)</a>'
            st.markdown(href_sample, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📈 Data Statistics")
        
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        
        stats_df = pd.DataFrame({
            'Column': numeric_cols,
            'Mean': filtered_df[numeric_cols].mean(),
            'Median': filtered_df[numeric_cols].median(),
            'Std Dev': filtered_df[numeric_cols].std(),
            'Min': filtered_df[numeric_cols].min(),
            'Max': filtered_df[numeric_cols].max(),
            'Non-Null %': (filtered_df[numeric_cols].notna().sum() / len(filtered_df) * 100)
        })
        
        st.dataframe(stats_df.style.format({
            'Mean': '{:,.2f}',
            'Median': '{:,.2f}',
            'Std Dev': '{:,.2f}',
            'Min': '{:,.2f}',
            'Max': '{:,.2f}',
            'Non-Null %': '{:.1f}%'
        }), use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🔄 Data Quality Check")
        
        quality_issues = []
        
        zero_revenue_2024 = len(filtered_df[filtered_df['MAT Q3 2024 USD MNF'] == 0])
        if zero_revenue_2024 > 0:
            quality_issues.append(f"{zero_revenue_2024} rows with zero 2024 revenue")
        
        price_discrepancy = len(filtered_df[
            (filtered_df['MAT Q3 2024 Unit Avg Price USD MNF'] > filtered_df['MAT Q3 2024 SU Avg Price USD MNF'] * 10)
        ])
        if price_discrepancy > 0:
            quality_issues.append(f"{price_discrepancy} rows with Unit price > 10x SU price")
        
        missing_country = filtered_df['Country'].isna().sum()
        if missing_country > 0:
            quality_issues.append(f"{missing_country} rows missing country data")
        
        duplicate_products = len(filtered_df) - len(filtered_df.drop_duplicates(subset=[
            'International Product', 'Country', 'Manufacturer'
        ]))
        if duplicate_products > 0:
            quality_issues.append(f"{duplicate_products} potential duplicate product entries")
        
        if quality_issues:
            st.warning("**Data Quality Issues Detected:**")
            for issue in quality_issues:
                st.markdown(f"- {issue}")
        else:
            st.success("✅ No major data quality issues detected")

if __name__ == "__main__":
    main()
