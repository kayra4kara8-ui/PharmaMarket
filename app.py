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
from scipy.stats import zscore, percentileofscore, spearmanr, kendalltau
import itertools
import functools
from collections import Counter, defaultdict
import time
import sys
import os

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Pharma Intelligence Suite | Q3 MAT Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_styles():
    st.markdown("""
    <style>
    .main { background-color: #0c0f1d; }
    .stApp { background: linear-gradient(180deg, #0c0f1d 0%, #13182e 50%, #1a2240 100%); }
    .sidebar .sidebar-content { background: linear-gradient(180deg, #13182e 0%, #1a2240 100%); }
    .st-emotion-cache-16idsys p { color: #e2e8f0; }
    
    .kpi-metric {
        background: linear-gradient(145deg, #1a2240 0%, #13182e 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .kpi-metric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
    }
    .kpi-title {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.2;
        margin-bottom: 6px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .kpi-subtitle {
        font-size: 14px;
        color: #cbd5e1;
        opacity: 0.9;
    }
    .kpi-trend {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 8px;
        backdrop-filter: blur(10px);
    }
    .trend-up { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3); }
    .trend-down { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }
    .trend-neutral { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid rgba(203, 213, 225, 0.3); }
    
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(56, 189, 248, 0.3) 50%, transparent 100%);
        margin: 30px 0;
    }
    
    .insight-panel {
        background: rgba(19, 24, 46, 0.7);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(56, 189, 248, 0.15);
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .insight-panel:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    .insight-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }
    .insight-icon {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
    }
    .insight-title {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
        flex: 1;
    }
    .insight-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-critical { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-success { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-info { background: rgba(56, 189, 248, 0.2); color: #60a5fa; border: 1px solid rgba(56, 189, 248, 0.3); }
    
    .data-grid {
        background: rgba(19, 24, 46, 0.7);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(56, 189, 248, 0.15);
        margin-top: 20px;
    }
    
    .manufacturer-score-card {
        background: linear-gradient(145deg, #1a2240 0%, #13182e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .manufacturer-score-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateX(4px);
    }
    .score-pill {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 18px;
        margin-right: 16px;
    }
    .score-excellent { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; }
    .score-good { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; }
    .score-fair { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; }
    .score-poor { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(19, 24, 46, 0.7);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(56, 189, 248, 0.15);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        background-color: transparent;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .filter-panel {
        background: rgba(19, 24, 46, 0.7);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(56, 189, 248, 0.15);
        margin-bottom: 24px;
    }
    
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        background-color: rgba(26, 34, 64, 0.7);
        border-color: rgba(56, 189, 248, 0.3);
        color: white;
        border-radius: 8px;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
    }
    
    .stSlider [data-baseweb="slider"] {
        color: #3b82f6;
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(56, 189, 248, 0.15);
    }
    
    .chart-container {
        background: rgba(19, 24, 46, 0.7);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(56, 189, 248, 0.15);
        margin-bottom: 24px;
    }
    
    .watermark {
        position: fixed;
        bottom: 20px;
        right: 20px;
        opacity: 0.1;
        font-size: 120px;
        font-weight: 900;
        color: #3b82f6;
        pointer-events: none;
        z-index: -1;
    }
    </style>
    <div class="watermark">PHARMA</div>
    """, unsafe_allow_html=True)

inject_custom_styles()

@st.cache_resource(show_spinner=False, max_entries=1)
def initialize_data_system():
    class PharmaDataSystem:
        def __init__(self):
            self.seed = 42
            np.random.seed(self.seed)
            self.initialize_dimensions()
            self.initialize_metrics()
            
        def initialize_dimensions(self):
            self.sources = ['IQVIA MAT', 'IMS Health', 'Intercontinental Medical', 'MarketTrack Pro', 'Evaluate Pharma']
            self.countries = ['USA', 'Germany', 'France', 'UK', 'Japan', 'Italy', 'Spain', 'Canada', 'Australia', 'Brazil', 
                            'China', 'India', 'Russia', 'Mexico', 'South Korea', 'Turkey', 'Saudi Arabia', 'South Africa']
            self.sectors = ['Oncology', 'Cardiology', 'Neurology', 'Endocrinology', 'Immunology', 'Respiratory', 
                          'Gastroenterology', 'Dermatology', 'Ophthalmology', 'Hematology', 'Infectious Diseases']
            self.panels = ['Retail Pharmacy', 'Hospital', 'Clinic', 'Combined Channels', 'Specialty Pharmacy']
            self.regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East & Africa']
            self.sub_regions = {
                'North America': ['USA & Canada'],
                'Europe': ['Western Europe', 'Eastern Europe', 'Nordic'],
                'Asia Pacific': ['Japan & Korea', 'China & Taiwan', 'ASEAN', 'ANZ'],
                'Latin America': ['Brazil & Southern Cone', 'Mexico & Central', 'Andean'],
                'Middle East & Africa': ['GCC', 'North Africa', 'Sub-Saharan Africa']
            }
            self.corporations = [
                'Pfizer Inc.', 'Novartis AG', 'Roche Holding', 'Merck & Co.', 'GlaxoSmithKline', 
                'Sanofi S.A.', 'AstraZeneca PLC', 'Johnson & Johnson', 'AbbVie Inc.', 'Bristol-Myers Squibb',
                'Eli Lilly', 'Amgen Inc.', 'Gilead Sciences', 'Takeda Pharmaceutical', 'Bayer AG',
                'Boehringer Ingelheim', 'Novo Nordisk', 'Astellas Pharma', 'Daiichi Sankyo', 'Biogen'
            ]
            self.manufacturers = [
                'Pfizer Manufacturing', 'Novartis Pharma', 'Roche Diagnostics', 'Merck Sharp & Dohme',
                'GSK Biologicals', 'Sanofi Pasteur', 'AstraZeneca Operations', 'Janssen Pharmaceuticals',
                'AbbVie Biologics', 'BMS Oncology', 'Lilly Biotechnology', 'Amgen Manufacturing',
                'Gilead Sciences Inc.', 'Takeda Oncology', 'Bayer Healthcare', 'Boehringer Ingelheim Pharma',
                'Novo Nordisk A/S', 'Astellas Manufacturing', 'Daiichi Sankyo Europe', 'Biogen International'
            ]
            self.molecules = [
                'Pembrolizumab', 'Nivolumab', 'Atezolizumab', 'Durvalumab', 'Ibrutinib', 'Venetoclax',
                'Acalabrutinib', 'Adalimumab', 'Etanercept', 'Infliximab', 'Ustekinumab', 'Secukinumab',
                'Empagliflozin', 'Dapagliflozin', 'Canagliflozin', 'Semaglutide', 'Liraglutide', 'Dulaglutide',
                'Apixaban', 'Rivaroxaban', 'Dabigatran', 'Edoxaban', 'Trastuzumab', 'Pertuzumab', 'Bevacizumab',
                'Rituximab', 'Brentuximab', 'Polatuzumab', 'Tocilizumab', 'Sarilumab', 'Dexamethasone',
                'Remdesivir', 'Molnupiravir', 'Paxlovid', 'Sotrovimab', 'Bamlanivimab', 'Casirivimab'
            ]
            self.chemical_salts = [
                'Sodium Chloride', 'Hydrochloride', 'Calcium Carbonate', 'Potassium Citrate',
                'Magnesium Stearate', 'Sodium Bicarbonate', 'Calcium Phosphate', 'Zinc Sulfate',
                'Ferrous Fumarate', 'Copper Gluconate', 'Manganese Sulfate', 'Selenium Methionine'
            ]
            self.specialty_status = ['Specialty', 'Non-Specialty', 'Orphan', 'Biologic']
            self.nfc_codes = [f'NFC{str(i).zfill(3)}' for i in range(1, 151)]
            self.pack_types = ['Bottle 30ct', 'Blister 28ct', 'Vial 10mL', 'Syringe 1mL', 'Pen Injector',
                             'Ampoule 2mL', 'Prefilled Syringe', 'Autoinjector', 'Inhaler 120d', 'Cream 30g']
            self.strengths = ['5mg', '10mg', '20mg', '40mg', '50mg', '100mg', '150mg', '200mg', '250mg', '500mg',
                            '1g', '2g', '5mcg', '10mcg', '20mcg', '50mcg', '100mcg', '200mcg', '500mcg']
            self.prescription_types = ['Rx Only', 'OTC', 'Hospital Only', 'Specialty Pharmacy', 'Mail Order']
            
        def initialize_metrics(self):
            self.metric_config = {
                'price_multipliers': {
                    'Specialty': 3.5,
                    'Orphan': 5.0,
                    'Biologic': 4.0,
                    'Non-Specialty': 1.0
                },
                'growth_trends': {
                    '2022-2023': {'min': -0.1, 'max': 0.4},
                    '2023-2024': {'min': -0.15, 'max': 0.5}
                },
                'volatility_factors': {
                    'Oncology': 0.3,
                    'Cardiology': 0.2,
                    'Neurology': 0.25,
                    'Immunology': 0.35
                }
            }
            
        def generate_corporate_data(self, n_rows=500000):
            data_records = []
            start_time = time.time()
            
            for i in range(n_rows):
                if i % 100000 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / i) * (n_rows - i)
                
                country = np.random.choice(self.countries)
                region = self.map_country_to_region(country)
                sub_region = np.random.choice(self.sub_regions.get(region, ['']))
                
                specialty = np.random.choice(self.specialty_status, p=[0.25, 0.5, 0.15, 0.1])
                sector = np.random.choice(self.sectors)
                
                price_multiplier = (
                    self.metric_config['price_multipliers'][specialty] *
                    (2.0 if sector == 'Oncology' else 1.0) *
                    (1.5 if country in ['USA', 'Germany', 'Japan'] else 1.0)
                )
                
                base_revenue = np.random.lognormal(8, 2) * price_multiplier
                base_units = np.random.lognormal(10, 1.5)
                
                growth_22_23 = np.random.uniform(
                    self.metric_config['growth_trends']['2022-2023']['min'],
                    self.metric_config['growth_trends']['2022-2023']['max']
                )
                
                growth_23_24 = np.random.uniform(
                    self.metric_config['growth_trends']['2023-2024']['min'],
                    self.metric_config['growth_trends']['2023-2024']['max']
                )
                
                volatility = self.metric_config['volatility_factors'].get(sector, 0.2)
                price_evolution = 1 + np.random.normal(0, volatility * 0.1, 3)
                
                record = {
                    'Source.Name': np.random.choice(self.sources),
                    'Country': country,
                    'Sector': sector,
                    'Panel': np.random.choice(self.panels),
                    'Region': region,
                    'Sub-Region': sub_region,
                    'Corporation': np.random.choice(self.corporations),
                    'Manufacturer': np.random.choice(self.manufacturers),
                    'Molecule List': ', '.join(np.random.choice(self.molecules, 
                                                               size=np.random.randint(1, 4), 
                                                               replace=False)),
                    'Molecule': np.random.choice(self.molecules),
                    'Chemical Salt': np.random.choice(self.chemical_salts),
                    'International Product': f'INT-{np.random.randint(10000, 99999)}',
                    'Specialty Product': specialty,
                    'NFC123': np.random.choice(self.nfc_codes),
                    'International Pack': np.random.choice(self.pack_types),
                    'International Strength': np.random.choice(self.strengths),
                    'International Size': np.random.randint(1, 100),
                    'International Volume': np.random.choice([1, 2.5, 5, 10, 20, 50, 100, 250]),
                    'International Prescription': np.random.choice(self.prescription_types),
                    
                    'MAT Q3 2022 USD MNF': base_revenue * (1 + np.random.uniform(-0.2, 0.2)),
                    'MAT Q3 2022 Standard Units': base_units * (1 + np.random.uniform(-0.1, 0.1)),
                    'MAT Q3 2022 Units': base_units * np.random.randint(1, 20) * (1 + np.random.uniform(-0.15, 0.15)),
                    'MAT Q3 2022 SU Avg Price USD MNF': price_multiplier * np.random.uniform(0.5, 2.0) * price_evolution[0],
                    'MAT Q3 2022 Unit Avg Price USD MNF': price_multiplier * np.random.uniform(0.1, 1.0) * price_evolution[0],
                    
                    'MAT Q3 2023 USD MNF': base_revenue * (1 + growth_22_23) * (1 + np.random.uniform(-0.15, 0.15)),
                    'MAT Q3 2023 Standard Units': base_units * (1 + growth_22_23 * 0.8) * (1 + np.random.uniform(-0.08, 0.08)),
                    'MAT Q3 2023 Units': base_units * np.random.randint(1, 20) * (1 + growth_22_23 * 0.8) * (1 + np.random.uniform(-0.12, 0.12)),
                    'MAT Q3 2023 SU Avg Price USD MNF': price_multiplier * np.random.uniform(0.5, 2.0) * price_evolution[1],
                    'MAT Q3 2023 Unit Avg Price USD MNF': price_multiplier * np.random.uniform(0.1, 1.0) * price_evolution[1],
                    
                    'MAT Q3 2024 USD MNF': base_revenue * (1 + growth_22_23) * (1 + growth_23_24) * (1 + np.random.uniform(-0.2, 0.2)),
                    'MAT Q3 2024 Standard Units': base_units * (1 + growth_22_23 * 0.8) * (1 + growth_23_24 * 0.8) * (1 + np.random.uniform(-0.1, 0.1)),
                    'MAT Q3 2024 Units': base_units * np.random.randint(1, 20) * (1 + growth_22_23 * 0.8) * (1 + growth_23_24 * 0.8) * (1 + np.random.uniform(-0.15, 0.15)),
                    'MAT Q3 2024 SU Avg Price USD MNF': price_multiplier * np.random.uniform(0.5, 2.0) * price_evolution[2],
                    'MAT Q3 2024 Unit Avg Price USD MNF': price_multiplier * np.random.uniform(0.1, 1.0) * price_evolution[2],
                }
                
                for key in ['MAT Q3 2022 USD MNF', 'MAT Q3 2023 USD MNF', 'MAT Q3 2024 USD MNF',
                          'MAT Q3 2022 SU Avg Price USD MNF', 'MAT Q3 2023 SU Avg Price USD MNF', 
                          'MAT Q3 2024 SU Avg Price USD MNF']:
                    record[key] = round(record[key], 2)
                
                for key in ['MAT Q3 2022 Standard Units', 'MAT Q3 2023 Standard Units', 'MAT Q3 2024 Standard Units',
                          'MAT Q3 2022 Units', 'MAT Q3 2023 Units', 'MAT Q3 2024 Units']:
                    record[key] = int(record[key])
                
                data_records.append(record)
                
                if i % 50000 == 0 and i > 0:
                    gc.collect()
            
            df = pd.DataFrame(data_records)
            
            for col in df.select_dtypes(include=[np.number]).columns:
                if np.random.random() < 0.25:
                    mask = np.random.random(len(df)) < 0.1
                    df.loc[mask, col] = np.nan
            
            numeric_cols = [
                'MAT Q3 2022 USD MNF', 'MAT Q3 2022 Standard Units', 'MAT Q3 2022 Units',
                'MAT Q3 2022 SU Avg Price USD MNF', 'MAT Q3 2022 Unit Avg Price USD MNF',
                'MAT Q3 2023 USD MNF', 'MAT Q3 2023 Standard Units', 'MAT Q3 2023 Units',
                'MAT Q3 2023 SU Avg Price USD MNF', 'MAT Q3 2023 Unit Avg Price USD MNF',
                'MAT Q3 2024 USD MNF', 'MAT Q3 2024 Standard Units', 'MAT Q3 2024 Units',
                'MAT Q3 2024 SU Avg Price USD MNF', 'MAT Q3 2024 Unit Avg Price USD MNF'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        
        def map_country_to_region(self, country):
            mapping = {
                'USA': 'North America',
                'Canada': 'North America',
                'Germany': 'Europe',
                'France': 'Europe',
                'UK': 'Europe',
                'Italy': 'Europe',
                'Spain': 'Europe',
                'Japan': 'Asia Pacific',
                'Australia': 'Asia Pacific',
                'China': 'Asia Pacific',
                'India': 'Asia Pacific',
                'South Korea': 'Asia Pacific',
                'Brazil': 'Latin America',
                'Mexico': 'Latin America',
                'Russia': 'Europe',
                'Turkey': 'Middle East & Africa',
                'Saudi Arabia': 'Middle East & Africa',
                'South Africa': 'Middle East & Africa'
            }
            return mapping.get(country, 'Other')
    
    return PharmaDataSystem()

@st.cache_data(show_spinner=False, ttl=3600, max_entries=1)
def load_pharma_data():
    with st.spinner('🧪 Generating 500,000+ pharmaceutical intelligence records...'):
        data_system = initialize_data_system()
        df = data_system.generate_corporate_data(500000)
        
        total_market_value = df['MAT Q3 2024 USD MNF'].sum()
        specialty_premium = (
            df[df['Specialty Product'] == 'Specialty']['MAT Q3 2024 SU Avg Price USD MNF'].mean() /
            df[df['Specialty Product'] == 'Non-Specialty']['MAT Q3 2024 SU Avg Price USD MNF'].mean()
            if df[df['Specialty Product'] == 'Non-Specialty']['MAT Q3 2024 SU Avg Price USD MNF'].mean() > 0 else 0
        )
        
        st.success(f'✅ Dataset loaded: {len(df):,} records | Market Value: ${total_market_value:,.0f} | Specialty Premium: {specialty_premium:.1f}x')
        
        return df

def compute_comprehensive_metrics(df):
    metrics = {}
    
    total_2022 = df['MAT Q3 2022 USD MNF'].sum()
    total_2023 = df['MAT Q3 2023 USD MNF'].sum()
    total_2024 = df['MAT Q3 2024 USD MNF'].sum()
    
    metrics['total_growth_2023'] = ((total_2023 - total_2022) / total_2022 * 100) if total_2022 > 0 else 0
    metrics['total_growth_2024'] = ((total_2024 - total_2023) / total_2023 * 100) if total_2023 > 0 else 0
    metrics['cagr_2yr'] = ((total_2024 / total_2022) ** 0.5 - 1) * 100 if total_2022 > 0 else 0
    
    specialty_df = df[df['Specialty Product'].isin(['Specialty', 'Orphan', 'Biologic'])]
    non_specialty_df = df[df['Specialty Product'] == 'Non-Specialty']
    
    metrics['specialty_growth_2024'] = ((specialty_df['MAT Q3 2024 USD MNF'].sum() - specialty_df['MAT Q3 2023 USD MNF'].sum()) /
                                       specialty_df['MAT Q3 2023 USD MNF'].sum() * 100) if specialty_df['MAT Q3 2023 USD MNF'].sum() > 0 else 0
    metrics['non_specialty_growth_2024'] = ((non_specialty_df['MAT Q3 2024 USD MNF'].sum() - non_specialty_df['MAT Q3 2023 USD MNF'].sum()) /
                                           non_specialty_df['MAT Q3 2023 USD MNF'].sum() * 100) if non_specialty_df['MAT Q3 2023 USD MNF'].sum() > 0 else 0
    
    price_growth_2024 = ((df['MAT Q3 2024 SU Avg Price USD MNF'].mean() - df['MAT Q3 2023 SU Avg Price USD MNF'].mean()) /
                        df['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if df['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0
    volume_growth_2024 = ((df['MAT Q3 2024 Standard Units'].sum() - df['MAT Q3 2023 Standard Units'].sum()) /
                         df['MAT Q3 2023 Standard Units'].sum() * 100) if df['MAT Q3 2023 Standard Units'].sum() > 0 else 0
    
    metrics['price_contribution'] = price_growth_2024
    metrics['volume_contribution'] = volume_growth_2024
    metrics['mix_contribution'] = metrics['total_growth_2024'] - price_growth_2024 - volume_growth_2024
    
    top_3_corp_share = (df.groupby('Corporation')['MAT Q3 2024 USD MNF'].sum().nlargest(3).sum() /
                       total_2024 * 100) if total_2024 > 0 else 0
    top_5_corp_share = (df.groupby('Corporation')['MAT Q3 2024 USD MNF'].sum().nlargest(5).sum() /
                       total_2024 * 100) if total_2024 > 0 else 0
    
    metrics['concentration_top3'] = top_3_corp_share
    metrics['concentration_top5'] = top_5_corp_share
    metrics['hhi_index'] = sum((df.groupby('Corporation')['MAT Q3 2024 USD MNF'].sum() / total_2024 * 100) ** 2) if total_2024 > 0 else 0
    
    exit_products = df[(df['MAT Q3 2023 USD MNF'] > 1000) & (df['MAT Q3 2024 USD MNF'] == 0)]
    launch_products = df[(df['MAT Q3 2023 USD MNF'] == 0) & (df['MAT Q3 2024 USD MNF'] > 1000)]
    
    metrics['product_exits'] = len(exit_products)
    metrics['exit_value'] = exit_products['MAT Q3 2023 USD MNF'].sum()
    metrics['product_launches'] = len(launch_products)
    metrics['launch_value'] = launch_products['MAT Q3 2024 USD MNF'].sum()
    
    fragility_score = 0
    fragility_factors = []
    
    if metrics['concentration_top3'] > 40:
        fragility_score += 25
        fragility_factors.append('High market concentration')
    if metrics['price_contribution'] > metrics['volume_contribution'] * 2:
        fragility_score += 20
        fragility_factors.append('Price-led growth dominance')
    if metrics['product_exits'] > 50:
        fragility_score += 15
        fragility_factors.append('High product attrition')
    if abs(metrics['specialty_growth_2024'] - metrics['non_specialty_growth_2024']) > 20:
        fragility_score += 20
        fragility_factors.append('Specialty/non-specialty divergence')
    if metrics['total_growth_2024'] > 25:
        fragility_score += 20
        fragility_factors.append('Unsustainable growth rate')
    
    metrics['fragility_score'] = min(100, fragility_score)
    metrics['fragility_factors'] = fragility_factors
    
    regional_concentration = df.groupby('Region')['MAT Q3 2024 USD MNF'].sum().nlargest(2).sum() / total_2024 * 100 if total_2024 > 0 else 0
    metrics['regional_concentration'] = regional_concentration
    
    price_elasticity = -spearmanr(df['MAT Q3 2024 SU Avg Price USD MNF'], df['MAT Q3 2024 Standard Units'])[0] if len(df) > 2 else 0
    metrics['price_elasticity'] = price_elasticity
    
    return metrics

def generate_executive_intelligence(metrics, df):
    intelligence_report = []
    
    intelligence_report.append("### 🎯 Executive Intelligence Summary")
    intelligence_report.append(f"**Market Performance:** {metrics['total_growth_2024']:+.1f}% growth in 2024 (2-Year CAGR: {metrics['cagr_2yr']:+.1f}%)")
    
    growth_composition = f"""
    **Growth Decomposition:**
    • Price Contribution: {metrics['price_contribution']:+.1f}%
    • Volume Contribution: {metrics['volume_contribution']:+.1f}%
    • Mix Contribution: {metrics['mix_contribution']:+.1f}%
    """
    
    if metrics['price_contribution'] > metrics['volume_contribution']:
        growth_composition += "\n• **🚨 Risk Alert:** Growth heavily price-dependent"
    else:
        growth_composition += "\n• **✅ Stability:** Volume-driven growth provides sustainability"
    
    intelligence_report.append(growth_composition)
    
    specialty_analysis = f"""
    **Specialty vs Non-Specialty Dynamics:**
    • Specialty Growth: {metrics['specialty_growth_2024']:+.1f}%
    • Non-Specialty Growth: {metrics['non_specialty_growth_2024']:+.1f}%
    • Growth Gap: {metrics['specialty_growth_2024'] - metrics['non_specialty_growth_2024']:+.1f}pp
    """
    
    if metrics['specialty_growth_2024'] - metrics['non_specialty_growth_2024'] > 15:
        specialty_analysis += "\n• **🚀 Opportunity:** Specialty driving market evolution"
    elif metrics['specialty_growth_2024'] - metrics['non_specialty_growth_2024'] < -10:
        specialty_analysis += "\n• **⚠️ Warning:** Non-specialty growth lagging"
    
    intelligence_report.append(specialty_analysis)
    
    concentration_risk = f"""
    **Market Concentration Analysis:**
    • Top 3 Corporations: {metrics['concentration_top3']:.1f}% market share
    • HHI Index: {metrics['hhi_index']:.0f} (Moderate concentration)
    • Regional Concentration: {metrics['regional_concentration']:.1f}% in top 2 regions
    """
    
    if metrics['concentration_top3'] > 50:
        concentration_risk += "\n• **🚨 High Risk:** Market oligopoly detected"
    elif metrics['concentration_top3'] > 35:
        concentration_risk += "\n• **⚠️ Moderate Risk:** Concentrated market structure"
    
    intelligence_report.append(concentration_risk)
    
    product_dynamics = f"""
    **Product Portfolio Dynamics:**
    • New Launches: {metrics['product_launches']} products (${metrics['launch_value']:,.0f})
    • Product Exits: {metrics['product_exits']} products (${metrics['exit_value']:,.0f})
    • Net Portfolio Change: {metrics['product_launches'] - metrics['product_exits']} products
    """
    
    if metrics['product_exits'] > metrics['product_launches']:
        product_dynamics += "\n• **📉 Contraction:** Portfolio shrinking faster than expanding"
    else:
        product_dynamics += "\n• **📈 Expansion:** Positive portfolio momentum"
    
    intelligence_report.append(product_dynamics)
    
    fragility_assessment = f"""
    **Growth Fragility Assessment:**
    • Fragility Score: {metrics['fragility_score']}/100
    • Price Elasticity: {metrics['price_elasticity']:.2f}
    """
    
    if metrics['fragility_score'] >= 70:
        fragility_assessment += "\n• **🚨 CRITICAL:** High fragility detected"
        fragility_assessment += "\n• Primary factors: " + ", ".join(metrics['fragility_factors'][:3])
    elif metrics['fragility_score'] >= 40:
        fragility_assessment += "\n• **⚠️ ELEVATED:** Moderate fragility present"
    else:
        fragility_assessment += "\n• **✅ STABLE:** Sustainable growth profile"
    
    intelligence_report.append(fragility_assessment)
    
    strategic_implications = """
    **Strategic Implications:**
    1. Monitor price-led growth sustainability
    2. Evaluate specialty portfolio expansion opportunities
    3. Assess concentration risk mitigation strategies
    4. Review product lifecycle management
    5. Consider geographic diversification initiatives
    """
    
    intelligence_report.append(strategic_implications)
    
    return "\n\n".join(intelligence_report)

def detect_market_structural_shifts(df):
    structural_shifts = []
    
    growth_engine_analysis = df.groupby(['Region', 'Specialty Product']).apply(
        lambda x: ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / 
                  x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0
    ).unstack()
    
    if 'Specialty' in growth_engine_analysis.columns and 'Non-Specialty' in growth_engine_analysis.columns:
        growth_engine_analysis['Specialty_Advantage'] = (
            growth_engine_analysis['Specialty'] - growth_engine_analysis['Non-Specialty']
        )
        
        regions_specialty_driven = growth_engine_analysis[
            growth_engine_analysis['Specialty_Advantage'] > 10
        ].index.tolist()
        
        if regions_specialty_driven:
            structural_shifts.append({
                'type': 'Growth Engine Reversal',
                'description': f'{len(regions_specialty_driven)} regions now specialty-driven',
                'regions': regions_specialty_driven,
                'severity': 'High' if len(regions_specialty_driven) > 2 else 'Medium'
            })
    
    channel_migration = df.groupby(['Panel', 'Specialty Product']).agg({
        'MAT Q3 2024 USD MNF': 'sum',
        'MAT Q3 2023 USD MNF': 'sum'
    }).unstack()
    
    if ('MAT Q3 2024 USD MNF', 'Specialty') in channel_migration.columns:
        specialty_growth_by_channel = (
            (channel_migration[('MAT Q3 2024 USD MNF', 'Specialty')] - 
             channel_migration[('MAT Q3 2023 USD MNF', 'Specialty')]) /
            channel_migration[('MAT Q3 2023 USD MNF', 'Specialty')] * 100
        )
        
        fastest_growing_channel = specialty_growth_by_channel.idxmax()
        fastest_growth_rate = specialty_growth_by_channel.max()
        
        if fastest_growth_rate > 30:
            structural_shifts.append({
                'type': 'Channel Migration',
                'description': f'Specialty migrating to {fastest_growing_channel} (+{fastest_growth_rate:.0f}%)',
                'channel': fastest_growing_channel,
                'severity': 'Medium'
            })
    
    regional_polarization = df.groupby('Region').apply(
        lambda x: ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / 
                  x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0
    )
    
    if len(regional_polarization) >= 3:
        fastest_region = regional_polarization.idxmax()
        slowest_region = regional_polarization.idxmin()
        growth_gap = regional_polarization.max() - regional_polarization.min()
        
        if growth_gap > 25:
            structural_shifts.append({
                'type': 'Regional Polarization',
                'description': f'{fastest_region} (+{regional_polarization.max():.0f}%) vs {slowest_region} ({regional_polarization.min():+.0f}%)',
                'gap': growth_gap,
                'severity': 'High' if growth_gap > 40 else 'Medium'
            })
    
    corporation_momentum = df.groupby('Corporation').apply(
        lambda x: pd.Series({
            'growth_2024': ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / 
                          x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0,
            'market_share_2024': (x['MAT Q3 2024 USD MNF'].sum() / df['MAT Q3 2024 USD MNF'].sum() * 100) 
                               if df['MAT Q3 2024 USD MNF'].sum() > 0 else 0
        })
    )
    
    rising_stars = corporation_momentum[
        (corporation_momentum['growth_2024'] > 20) & 
        (corporation_momentum['market_share_2024'] > 1)
    ].sort_values('growth_2024', ascending=False)
    
    if len(rising_stars) > 0:
        top_riser = rising_stars.index[0]
        structural_shifts.append({
            'type': 'New Market Leaders',
            'description': f'{top_riser} emerging as growth leader (+{rising_stars.iloc[0]["growth_2024"]:.0f}%)',
            'corporation': top_riser,
            'severity': 'Medium'
        })
    
    molecule_concentration = df.groupby('Molecule').apply(
        lambda x: x['MAT Q3 2024 USD MNF'].sum()
    ).nlargest(5).sum() / df['MAT Q3 2024 USD MNF'].sum() * 100
    
    if molecule_concentration > 30:
        structural_shifts.append({
            'type': 'Molecule Concentration',
            'description': f'Top 5 molecules represent {molecule_concentration:.1f}% of market',
            'concentration': molecule_concentration,
            'severity': 'Medium' if molecule_concentration > 40 else 'Low'
        })
    
    return structural_shifts

def perform_molecule_intelligence(df):
    molecule_insights = []
    
    molecule_performance = df.groupby('Molecule').apply(lambda x: pd.Series({
        'total_revenue_2024': x['MAT Q3 2024 USD MNF'].sum(),
        'growth_2024': ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / 
                       x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0,
        'growth_2023': ((x['MAT Q3 2023 USD MNF'].sum() - x['MAT Q3 2022 USD MNF'].sum()) / 
                       x['MAT Q3 2022 USD MNF'].sum() * 100) if x['MAT Q3 2022 USD MNF'].sum() > 0 else 0,
        'avg_price_2024': x['MAT Q3 2024 SU Avg Price USD MNF'].mean(),
        'price_change': ((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                        x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
        'geographic_spread': x['Country'].nunique(),
        'manufacturer_count': x['Manufacturer'].nunique(),
        'specialty_ratio': (x[x['Specialty Product'].isin(['Specialty', 'Orphan', 'Biologic'])]['MAT Q3 2024 USD MNF'].sum() / 
                          x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0
    })).reset_index()
    
    structural_growth_molecules = molecule_performance[
        (molecule_performance['growth_2023'] > 10) &
        (molecule_performance['growth_2024'] > 10) &
        (molecule_performance['total_revenue_2024'] > 1000000)
    ].sort_values('growth_2024', ascending=False)
    
    if len(structural_growth_molecules) > 0:
        top_structural = structural_growth_molecules.head(3)
        for _, row in top_structural.iterrows():
            molecule_insights.append({
                'type': 'Structural Growth Molecule',
                'molecule': row['Molecule'],
                'growth': f"+{row['growth_2024']:.1f}%",
                'revenue': f"${row['total_revenue_2024']:,.0f}",
                'momentum': 'Accelerating' if row['growth_2024'] > row['growth_2023'] else 'Stable',
                'priority': 'High'
            })
    
    relaunch_candidates = []
    for molecule in df['Molecule'].unique():
        molecule_data = df[df['Molecule'] == molecule]
        
        salt_analysis = molecule_data.groupby('Chemical Salt').agg({
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum'
        })
        
        if len(salt_analysis) >= 2:
            salt_growth = ((salt_analysis['MAT Q3 2024 USD MNF'] - salt_analysis['MAT Q3 2023 USD MNF']) / 
                          salt_analysis['MAT Q3 2023 USD MNF'] * 100)
            
            new_salts = salt_growth[salt_growth > 100].index.tolist()
            if new_salts:
                relaunch_candidates.append(f"{molecule} (New salts: {', '.join(new_salts)})")
        
        strength_analysis = molecule_data.groupby('International Strength').agg({
            'MAT Q3 2024 USD MNF': 'sum'
        }).sort_values('MAT Q3 2024 USD MNF', ascending=False)
        
        if len(strength_analysis) >= 3:
            concentration = strength_analysis.head(2)['MAT Q3 2024 USD MNF'].sum() / strength_analysis['MAT Q3 2024 USD MNF'].sum()
            if concentration < 0.6:
                relaunch_candidates.append(f"{molecule} (Diversified strength portfolio)")
    
    if relaunch_candidates:
        molecule_insights.append({
            'type': 'Relaunch Detection',
            'candidates': relaunch_candidates[:5],
            'count': len(relaunch_candidates),
            'priority': 'Medium'
        })
    
    saturation_analysis = molecule_performance[
        (molecule_performance['growth_2024'] < 5) &
        (molecule_performance['growth_2023'] < 5) &
        (molecule_performance['total_revenue_2024'] > 5000000) &
        (molecule_performance['price_change'] < 0)
    ]
    
    if len(saturation_analysis) > 0:
        saturated_molecules = saturation_analysis.nlargest(3, 'total_revenue_2024')
        molecule_insights.append({
            'type': 'Market Saturation',
            'molecules': saturated_molecules['Molecule'].tolist(),
            'avg_growth': saturated_molecules['growth_2024'].mean(),
            'priority': 'High'
        })
    
    commoditization_risk = molecule_performance[
        (molecule_performance['price_change'] < -10) &
        (molecule_performance['growth_2024'] > 0) &
        (molecule_performance['manufacturer_count'] > 3) &
        (molecule_performance['specialty_ratio'] < 30)
    ]
    
    if len(commoditization_risk) > 0:
        commoditized = commoditization_risk.nlargest(3, 'total_revenue_2024')
        molecule_insights.append({
            'type': 'Commoditization Risk',
            'molecules': commoditized['Molecule'].tolist(),
            'avg_price_decline': commoditized['price_change'].mean(),
            'priority': 'Critical'
        })
    
    geographic_expansion = molecule_performance[
        (molecule_performance['geographic_spread'] > 10) &
        (molecule_performance['growth_2024'] > 15)
    ]
    
    if len(geographic_expansion) > 0:
        expanding_molecules = geographic_expansion.nlargest(3, 'geographic_spread')
        molecule_insights.append({
            'type': 'Geographic Expansion',
            'molecules': expanding_molecules['Molecule'].tolist(),
            'avg_countries': expanding_molecules['geographic_spread'].mean(),
            'priority': 'Medium'
        })
    
    return molecule_insights

def calculate_manufacturer_scoring(df):
    manufacturer_scores = []
    
    manufacturer_metrics = df.groupby('Manufacturer').apply(lambda x: pd.Series({
        'total_revenue_2024': x['MAT Q3 2024 USD MNF'].sum(),
        'revenue_growth': ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / 
                         x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0,
        'volume_growth': ((x['MAT Q3 2024 Standard Units'].sum() - x['MAT Q3 2023 Standard Units'].sum()) / 
                         x['MAT Q3 2023 Standard Units'].sum() * 100) if x['MAT Q3 2023 Standard Units'].sum() > 0 else 0,
        'price_power': ((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                       x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
        'specialty_share': (x[x['Specialty Product'].isin(['Specialty', 'Orphan', 'Biologic'])]['MAT Q3 2024 USD MNF'].sum() / 
                          x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0,
        'geographic_diversity': x['Country'].nunique(),
        'product_concentration': (x.groupby('International Product')['MAT Q3 2024 USD MNF'].sum().nlargest(1).sum() / 
                                 x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0,
        'margin_erosion': ((x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() - x['MAT Q3 2023 Unit Avg Price USD MNF'].mean()) / 
                          x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() > 0 else 0,
        'volume_scale': x['MAT Q3 2024 Standard Units'].sum()
    })).reset_index()
    
    manufacturer_metrics['pricing_power_score'] = manufacturer_metrics['price_power'].apply(
        lambda x: 90 if x > 15 else 75 if x > 10 else 60 if x > 5 else 45 if x > 0 else 30 if x > -5 else 15
    )
    
    manufacturer_metrics['volume_scale_score'] = manufacturer_metrics['volume_scale'].apply(
        lambda x: 100 if x > 1000000 else 80 if x > 500000 else 60 if x > 100000 else 40 if x > 50000 else 20
    )
    
    manufacturer_metrics['growth_stability_score'] = manufacturer_metrics.apply(
        lambda row: 80 if row['revenue_growth'] > 20 and row['volume_growth'] > 10 else
                   60 if row['revenue_growth'] > 10 and row['volume_growth'] > 5 else
                   40 if row['revenue_growth'] > 0 else 20,
        axis=1
    )
    
    manufacturer_metrics['diversification_score'] = manufacturer_metrics.apply(
        lambda row: 90 if row['geographic_diversity'] > 15 and row['product_concentration'] < 30 else
                   70 if row['geographic_diversity'] > 10 and row['product_concentration'] < 50 else
                   50 if row['geographic_diversity'] > 5 else 30,
        axis=1
    )
    
    manufacturer_metrics['margin_health_score'] = manufacturer_metrics['margin_erosion'].apply(
        lambda x: 100 if x > 5 else 80 if x > 0 else 60 if x > -5 else 40 if x > -10 else 20
    )
    
    manufacturer_metrics['specialty_positioning_score'] = manufacturer_metrics['specialty_share'].apply(
        lambda x: 100 if x > 70 else 80 if x > 50 else 60 if x > 30 else 40 if x > 10 else 20
    )
    
    manufacturer_metrics['total_score'] = (
        manufacturer_metrics['pricing_power_score'] * 0.25 +
        manufacturer_metrics['volume_scale_score'] * 0.20 +
        manufacturer_metrics['growth_stability_score'] * 0.20 +
        manufacturer_metrics['diversification_score'] * 0.15 +
        manufacturer_metrics['margin_health_score'] * 0.10 +
        manufacturer_metrics['specialty_positioning_score'] * 0.10
    )
    
    manufacturer_metrics['overall_grade'] = manufacturer_metrics['total_score'].apply(
        lambda x: 'A' if x >= 85 else 'B' if x >= 70 else 'C' if x >= 55 else 'D' if x >= 40 else 'F'
    )
    
    manufacturer_metrics['risk_flags'] = manufacturer_metrics.apply(
        lambda row: [
            'Margin Erosion' if row['margin_erosion'] < -10 else None,
            'High Concentration' if row['product_concentration'] > 60 else None,
            'Low Growth' if row['revenue_growth'] < 0 else None,
            'Limited Geography' if row['geographic_diversity'] < 3 else None
        ],
        axis=1
    )
    
    manufacturer_metrics['risk_flags'] = manufacturer_metrics['risk_flags'].apply(
        lambda flags: [flag for flag in flags if flag is not None]
    )
    
    top_manufacturers = manufacturer_metrics.nlargest(15, 'total_score')
    
    for _, row in top_manufacturers.iterrows():
        score_card = {
            'manufacturer': row['Manufacturer'],
            'total_score': int(round(row['total_score'])),
            'grade': row['overall_grade'],
            'revenue': f"${row['total_revenue_2024']:,.0f}",
            'revenue_growth': f"{row['revenue_growth']:+.1f}%",
            'volume_growth': f"{row['volume_growth']:+.1f}%",
            'price_power': f"{row['price_power']:+.1f}%",
            'specialty_share': f"{row['specialty_share']:.1f}%",
            'diversification': row['geographic_diversity'],
            'concentration': f"{row['product_concentration']:.1f}%",
            'risk_flags': row['risk_flags'],
            'scores': {
                'pricing': row['pricing_power_score'],
                'volume': row['volume_scale_score'],
                'growth': row['growth_stability_score'],
                'diversification': row['diversification_score'],
                'margins': row['margin_health_score'],
                'specialty': row['specialty_positioning_score']
            }
        }
        manufacturer_scores.append(score_card)
    
    return manufacturer_scores

def analyze_pricing_mix_dynamics(df):
    pricing_insights = []
    
    price_divergence_analysis = df.groupby('Molecule').apply(lambda x: pd.Series({
        'su_price_growth': ((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                           x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
        'unit_price_growth': ((x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() - x['MAT Q3 2023 Unit Avg Price USD MNF'].mean()) / 
                             x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() > 0 else 0,
        'revenue_2024': x['MAT Q3 2024 USD MNF'].sum(),
        'volume_2024': x['MAT Q3 2024 Standard Units'].sum()
    }))
    
    significant_divergence = price_divergence_analysis[
        (abs(price_divergence_analysis['su_price_growth'] - price_divergence_analysis['unit_price_growth']) > 15) &
        (price_divergence_analysis['revenue_2024'] > 500000)
    ].sort_values('revenue_2024', ascending=False)
    
    if len(significant_divergence) > 0:
        top_divergence = significant_divergence.head(5)
        for idx, row in top_divergence.iterrows():
            divergence = row['su_price_growth'] - row['unit_price_growth']
            insight = {
                'type': 'Price Divergence',
                'molecule': idx,
                'su_growth': f"{row['su_price_growth']:+.1f}%",
                'unit_growth': f"{row['unit_price_growth']:+.1f}%",
                'gap': f"{divergence:+.1f}pp",
                'implication': 'Hidden discounting' if divergence > 0 else 'Pack size optimization'
            }
            pricing_insights.append(insight)
    
    pack_size_optimization = df.groupby(['International Pack', 'International Size']).apply(lambda x: pd.Series({
        'avg_unit_price': x['MAT Q3 2024 Unit Avg Price USD MNF'].mean(),
        'avg_su_price': x['MAT Q3 2024 SU Avg Price USD MNF'].mean(),
        'price_per_unit_size': x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() / x['International Size'].mean() 
                              if x['International Size'].mean() > 0 else 0,
        'total_units': x['MAT Q3 2024 Units'].sum(),
        'revenue': x['MAT Q3 2024 USD MNF'].sum(),
        'volume': x['MAT Q3 2024 Standard Units'].sum()
    })).reset_index()
    
    pack_size_optimization['efficiency_ratio'] = (
        pack_size_optimization['price_per_unit_size'] / 
        pack_size_optimization['price_per_unit_size'].median()
    )
    
    optimization_opportunities = pack_size_optimization[
        (pack_size_optimization['efficiency_ratio'] < 0.7) &
        (pack_size_optimization['revenue'] > 100000)
    ].sort_values('efficiency_ratio')
    
    if len(optimization_opportunities) > 0:
        top_opportunities = optimization_opportunities.head(5)
        opportunities_list = []
        for _, row in top_opportunities.iterrows():
            opportunities_list.append(
                f"{row['International Pack']} {row['International Size']}x "
                f"(Eff: {row['efficiency_ratio']:.2f}, Rev: ${row['revenue']:,.0f})"
            )
        
        pricing_insights.append({
            'type': 'Pack Size Optimization',
            'opportunities': opportunities_list,
            'count': len(optimization_opportunities),
            'avg_efficiency': optimization_opportunities['efficiency_ratio'].mean()
        })
    
    hidden_discounting_detection = df.groupby(['Manufacturer', 'Country', 'Molecule']).apply(lambda x: pd.Series({
        'unit_su_price_ratio': x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() / 
                               x['MAT Q3 2024 SU Avg Price USD MNF'].mean() 
                               if x['MAT Q3 2024 SU Avg Price USD MNF'].mean() > 0 else 0,
        'ratio_change_2023': (
            (x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() / x['MAT Q3 2024 SU Avg Price USD MNF'].mean()) - 
            (x['MAT Q3 2023 Unit Avg Price USD MNF'].mean() / x['MAT Q3 2023 SU Avg Price USD MNF'].mean())
        ) if x['MAT Q3 2024 SU Avg Price USD MNF'].mean() > 0 and x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
        'revenue_2024': x['MAT Q3 2024 USD MNF'].sum(),
        'price_variance': x['MAT Q3 2024 Unit Avg Price USD MNF'].std() / 
                         x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() 
                         if x['MAT Q3 2024 Unit Avg Price USD MNF'].mean() > 0 else 0
    })).reset_index()
    
    potential_discounting = hidden_discounting_detection[
        (hidden_discounting_detection['unit_su_price_ratio'] < 0.6) &
        (hidden_discounting_detection['revenue_2024'] > 250000) &
        (hidden_discounting_detection['price_variance'] > 0.25)
    ].sort_values('unit_su_price_ratio')
    
    if len(potential_discounting) > 0:
        top_discounting = potential_discounting.head(5)
        discounting_cases = []
        for _, row in top_discounting.iterrows():
            discounting_cases.append(
                f"{row['Manufacturer']} - {row['Country']} - {row['Molecule']} "
                f"(Ratio: {row['unit_su_price_ratio']:.2f}, Δ: {row['ratio_change_2023']:+.2f})"
            )
        
        pricing_insights.append({
            'type': 'Hidden Discounting',
            'cases': discounting_cases,
            'count': len(potential_discounting),
            'avg_ratio': potential_discounting['unit_su_price_ratio'].mean()
        })
    
    mix_shift_analysis = df.groupby(['Specialty Product', 'Region']).apply(lambda x: pd.Series({
        'share_2023': x['MAT Q3 2023 USD MNF'].sum() / 
                     df[df['Region'] == x.name[1]]['MAT Q3 2023 USD MNF'].sum() * 100 
                     if df[df['Region'] == x.name[1]]['MAT Q3 2023 USD MNF'].sum() > 0 else 0,
        'share_2024': x['MAT Q3 2024 USD MNF'].sum() / 
                     df[df['Region'] == x.name[1]]['MAT Q3 2024 USD MNF'].sum() * 100 
                     if df[df['Region'] == x.name[1]]['MAT Q3 2024 USD MNF'].sum() > 0 else 0
    })).reset_index()
    
    significant_mix_shifts = []
    for specialty in ['Specialty', 'Orphan', 'Biologic']:
        specialty_mix = mix_shift_analysis[mix_shift_analysis['Specialty Product'] == specialty]
        if not specialty_mix.empty:
            for _, row in specialty_mix.iterrows():
                shift = row['share_2024'] - row['share_2023']
                if abs(shift) > 5:
                    significant_mix_shifts.append({
                        'region': row['Region'],
                        'specialty_type': specialty,
                        'shift': shift,
                        'new_share': row['share_2024']
                    })
    
    if significant_mix_shifts:
        pricing_insights.append({
            'type': 'Mix Shift Analysis',
            'shifts': sorted(significant_mix_shifts, key=lambda x: abs(x['shift']), reverse=True)[:5],
            'total_shifts': len(significant_mix_shifts)
        })
    
    price_volume_tradeoff = df.groupby('Molecule').apply(lambda x: pd.Series({
        'price_growth': ((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                        x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
        'volume_growth': ((x['MAT Q3 2024 Standard Units'].sum() - x['MAT Q3 2023 Standard Units'].sum()) / 
                         x['MAT Q3 2023 Standard Units'].sum() * 100) if x['MAT Q3 2023 Standard Units'].sum() > 0 else 0,
        'revenue_2024': x['MAT Q3 2024 USD MNF'].sum()
    }))
    
    elastic_molecules = price_volume_tradeoff[
        (price_volume_tradeoff['price_growth'] > 10) &
        (price_volume_tradeoff['volume_growth'] > 0)
    ].sort_values('revenue_2024', ascending=False)
    
    if len(elastic_molecules) > 0:
        top_elastic = elastic_molecules.head(5)
        pricing_insights.append({
            'type': 'Price Elasticity Success',
            'molecules': [
                {
                    'name': idx,
                    'price_growth': f"{row['price_growth']:+.1f}%",
                    'volume_growth': f"{row['volume_growth']:+.1f}%",
                    'revenue': f"${row['revenue_2024']:,.0f}"
                }
                for idx, row in top_elastic.iterrows()
            ],
            'implication': 'Successful price increases with maintained volume'
        })
    
    return pricing_insights

def create_visualization_growth_decomposition(metrics):
    fig = go.Figure()
    
    components = ['Price', 'Volume', 'Mix']
    contributions = [metrics['price_contribution'], metrics['volume_contribution'], metrics['mix_contribution']]
    colors = ['#3b82f6', '#10b981', '#8b5cf6']
    
    for comp, contr, color in zip(components, contributions, colors):
        fig.add_trace(go.Bar(
            x=[comp],
            y=[contr],
            name=comp,
            marker_color=color,
            text=f'{contr:+.1f}%',
            textposition='auto',
            hovertemplate=f'<b>{comp} Contribution</b><br>{contr:+.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='2024 Growth Decomposition Analysis',
        yaxis_title='Contribution (%)',
        barmode='group',
        showlegend=True,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        hoverlabel=dict(bgcolor='#1a2240', font_color='white')
    )
    
    return fig

def create_visualization_specialty_premium(df):
    specialty_types = df['Specialty Product'].unique()
    avg_prices = []
    growth_rates = []
    
    for specialty in specialty_types:
        specialty_data = df[df['Specialty Product'] == specialty]
        avg_price = specialty_data['MAT Q3 2024 SU Avg Price USD MNF'].mean()
        growth = ((specialty_data['MAT Q3 2024 USD MNF'].sum() - specialty_data['MAT Q3 2023 USD MNF'].sum()) / 
                 specialty_data['MAT Q3 2023 USD MNF'].sum() * 100) if specialty_data['MAT Q3 2023 USD MNF'].sum() > 0 else 0
        
        avg_prices.append(avg_price)
        growth_rates.append(growth)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Price by Product Type', 'Growth Rate by Product Type'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig.add_trace(
        go.Bar(
            x=specialty_types,
            y=avg_prices,
            name='Avg Price',
            marker_color='#3b82f6',
            text=[f'${p:,.0f}' for p in avg_prices],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=specialty_types,
            y=growth_rates,
            name='Growth Rate',
            marker_color='#10b981',
            text=[f'{g:+.1f}%' for g in growth_rates],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Specialty Product Analysis',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=False
    )
    
    return fig

def create_visualization_market_concentration(df):
    corp_share = df.groupby('Corporation')['MAT Q3 2024 USD MNF'].sum().nlargest(10)
    other_share = df['MAT Q3 2024 USD MNF'].sum() - corp_share.sum()
    
    labels = list(corp_share.index) + ['Others']
    values = list(corp_share.values) + [other_share]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=px.colors.sequential.Plasma[:len(labels)],
        textinfo='label+percent',
        textposition='inside',
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.0f}<br>Share: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Top 10 Corporations Market Share (2024)',
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def create_visualization_regional_growth(df):
    regional_growth = df.groupby('Region').apply(lambda x: pd.Series({
        'growth_2024': ((x['MAT Q3 2024 USD MNF'].sum() - x['MAT Q3 2023 USD MNF'].sum()) / 
                       x['MAT Q3 2023 USD MNF'].sum() * 100) if x['MAT Q3 2023 USD MNF'].sum() > 0 else 0,
        'share_2024': (x['MAT Q3 2024 USD MNF'].sum() / df['MAT Q3 2024 USD MNF'].sum() * 100) 
                     if df['MAT Q3 2024 USD MNF'].sum() > 0 else 0,
        'specialty_share': (x[x['Specialty Product'].isin(['Specialty', 'Orphan', 'Biologic'])]['MAT Q3 2024 USD MNF'].sum() / 
                          x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0
    })).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Growth by Region', 'Market Share', 'Specialty Share', 'Growth vs Specialty Correlation'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    fig.add_trace(
        go.Bar(
            x=regional_growth['Region'],
            y=regional_growth['growth_2024'],
            name='Growth',
            marker_color='#3b82f6',
            text=regional_growth['growth_2024'].apply(lambda x: f'{x:+.1f}%'),
            textposition='auto'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=regional_growth['Region'],
            y=regional_growth['share_2024'],
            name='Share',
            marker_color='#10b981',
            text=regional_growth['share_2024'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=regional_growth['Region'],
            y=regional_growth['specialty_share'],
            name='Specialty Share',
            marker_color='#8b5cf6',
            text=regional_growth['specialty_share'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=regional_growth['specialty_share'],
            y=regional_growth['growth_2024'],
            mode='markers+text',
            marker=dict(
                size=regional_growth['share_2024'] * 2,
                color=regional_growth['growth_2024'],
                colorscale='Viridis',
                showscale=True
            ),
            text=regional_growth['Region'],
            textposition='top center',
            name='Correlation'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Regional Performance Analysis',
        height=800,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=False
    )
    
    return fig

def create_visualization_molecule_heatmap(df):
    top_molecules = df.groupby('Molecule')['MAT Q3 2024 USD MNF'].sum().nlargest(15).index
    top_countries = df.groupby('Country')['MAT Q3 2024 USD MNF'].sum().nlargest(10).index
    
    heatmap_data = df[
        (df['Molecule'].isin(top_molecules)) & 
        (df['Country'].isin(top_countries))
    ].groupby(['Molecule', 'Country'])['MAT Q3 2024 USD MNF'].sum().unstack()
    
    fig = go.Figure(data=go.Heatmap(
        z=np.log10(heatmap_data.values + 1),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis',
        colorbar=dict(title="Log10(Revenue)"),
        hovertext=heatmap_data.values,
        hovertemplate='<b>%{y} in %{x}</b><br>Revenue: $%{hovertext:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Molecule-Country Revenue Heatmap (Top 15×10)',
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_title='Country',
        yaxis_title='Molecule'
    )
    
    return fig

def create_visualization_price_volume_matrix(df):
    manufacturer_stats = df.groupby('Manufacturer').apply(lambda x: pd.Series({
        'price_growth': ((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                        x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
        'volume_growth': ((x['MAT Q3 2024 Standard Units'].sum() - x['MAT Q3 2023 Standard Units'].sum()) / 
                         x['MAT Q3 2023 Standard Units'].sum() * 100) if x['MAT Q3 2023 Standard Units'].sum() > 0 else 0,
        'revenue_2024': x['MAT Q3 2024 USD MNF'].sum(),
        'specialty_ratio': (x[x['Specialty Product'].isin(['Specialty', 'Orphan', 'Biologic'])]['MAT Q3 2024 USD MNF'].sum() / 
                          x['MAT Q3 2024 USD MNF'].sum() * 100) if x['MAT Q3 2024 USD MNF'].sum() > 0 else 0
    })).reset_index()
    
    top_manufacturers = manufacturer_stats.nlargest(20, 'revenue_2024')
    
    fig = px.scatter(
        top_manufacturers,
        x='price_growth',
        y='volume_growth',
        size='revenue_2024',
        color='specialty_ratio',
        hover_name='Manufacturer',
        hover_data=['revenue_2024', 'specialty_ratio'],
        title='Manufacturer Price vs Volume Growth Strategy',
        labels={
            'price_growth': 'Price Growth (%)',
            'volume_growth': 'Volume Growth (%)',
            'specialty_ratio': 'Specialty Share (%)',
            'revenue_2024': 'Revenue Size'
        },
        color_continuous_scale='Viridis'
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    
    fig.add_annotation(x=15, y=15, text="Premium Growers", showarrow=False, font=dict(color="#10b981", size=12))
    fig.add_annotation(x=-10, y=15, text="Volume Drivers", showarrow=False, font=dict(color="#3b82f6", size=12))
    fig.add_annotation(x=-10, y=-10, text="Declining", showarrow=False, font=dict(color="#ef4444", size=12))
    fig.add_annotation(x=15, y=-10, text="Price Increase/Volume Decline", showarrow=False, font=dict(color="#f59e0b", size=12))
    
    fig.update_layout(
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        hoverlabel=dict(bgcolor='#1a2240', font_color='white')
    )
    
    return fig

def render_kpi_metric(title, value, subtitle=None, trend=None, icon="📊"):
    trend_html = ""
    if trend is not None:
        trend_class = "trend-up" if trend > 0 else "trend-down" if trend < 0 else "trend-neutral"
        trend_symbol = "↗" if trend > 0 else "↘" if trend < 0 else "→"
        trend_html = f'<div class="kpi-trend {trend_class}">{trend_symbol} {abs(trend):.1f}%</div>'
    
    subtitle_html = f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ""
    
    return f'''
    <div class="kpi-metric">
        <div class="kpi-title">{icon} {title}</div>
        <div class="kpi-value">{value}</div>
        {subtitle_html}
        {trend_html}
    </div>
    '''

def render_insight_panel(title, content, insight_type="info", icon="💡", badge_text=None):
    badge_map = {
        'critical': 'badge-critical',
        'warning': 'badge-warning',
        'success': 'badge-success',
        'info': 'badge-info'
    }
    
    badge_class = badge_map.get(insight_type, 'badge-info')
    badge_html = f'<span class="insight-badge {badge_class}">{badge_text}</span>' if badge_text else ""
    
    return f'''
    <div class="insight-panel">
        <div class="insight-header">
            <div class="insight-icon">{icon}</div>
            <div class="insight-title">{title}</div>
            {badge_html}
        </div>
        <div class="insight-content">{content}</div>
    </div>
    '''

def render_manufacturer_score_card(score_data):
    score_class_map = {
        'A': 'score-excellent',
        'B': 'score-good',
        'C': 'score-fair',
        'D': 'score-poor',
        'F': 'score-poor'
    }
    
    risk_flags_html = ""
    if score_data['risk_flags']:
        flags = " • ".join(score_data['risk_flags'])
        risk_flags_html = f'<div style="margin-top: 8px; font-size: 12px; color: #f87171;">⚠️ {flags}</div>'
    
    scores_html = f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px;">
        <div style="text-align: center;">
            <div style="font-size: 11px; color: #94a3b8;">Pricing</div>
            <div style="font-weight: 700; color: #3b82f6;">{score_data['scores']['pricing']}</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 11px; color: #94a3b8;">Volume</div>
            <div style="font-weight: 700; color: #10b981;">{score_data['scores']['volume']}</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 11px; color: #94a3b8;">Growth</div>
            <div style="font-weight: 700; color: #8b5cf6;">{score_data['scores']['growth']}</div>
        </div>
    </div>
    """
    
    return f'''
    <div class="manufacturer-score-card">
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <div class="score-pill {score_class_map[score_data['grade']]}">
                {score_data['total_score']}
            </div>
            <div style="flex: 1;">
                <div style="font-weight: 800; font-size: 16px; color: white;">{score_data['manufacturer']}</div>
                <div style="display: flex; gap: 12px; margin-top: 4px;">
                    <span style="font-size: 12px; color: #94a3b8;">Grade: {score_data['grade']}</span>
                    <span style="font-size: 12px; color: #94a3b8;">Growth: {score_data['revenue_growth']}</span>
                    <span style="font-size: 12px; color: #94a3b8;">Specialty: {score_data['specialty_share']}</span>
                </div>
            </div>
        </div>
        {scores_html}
        {risk_flags_html}
    </div>
    '''

def main():
    st.title("💊 Pharma Intelligence Suite")
    st.markdown("### Q3 MAT Analytics Platform | 500,000+ Records | Advanced Market Intelligence")
    
    df = load_pharma_data()
    
    with st.sidebar:
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        st.markdown("## 🔍 Advanced Filters")
        
        filter_expander = st.expander("Dimension Filters", expanded=True)
        with filter_expander:
            all_countries = ['All Markets'] + sorted(df['Country'].unique().tolist())
            selected_country = st.selectbox("Country", all_countries, key='country_filter')
            
            all_corporations = ['All Corporations'] + sorted(df['Corporation'].unique().tolist())
            selected_corporation = st.selectbox("Corporation", all_corporations, key='corp_filter')
            
            all_manufacturers = ['All Manufacturers'] + sorted(df['Manufacturer'].unique().tolist())
            selected_manufacturer = st.selectbox("Manufacturer", all_manufacturers, key='manu_filter')
            
            all_molecules = ['All Molecules'] + sorted(df['Molecule'].unique().tolist())
            selected_molecule = st.selectbox("Molecule", all_molecules, key='molecule_filter')
            
            all_specialty = ['All Types'] + sorted(df['Specialty Product'].unique().tolist())
            selected_specialty = st.selectbox("Product Type", all_specialty, key='specialty_filter')
        
        metric_expander = st.expander("Performance Filters", expanded=True)
        with metric_expander:
            min_growth = st.slider("Minimum Growth Rate (%)", -50, 100, 0, 5, key='growth_filter')
            min_revenue = st.number_input("Minimum Revenue ($)", 0, 10000000, 0, 1000, key='revenue_filter')
            min_margin = st.slider("Minimum Price Margin (%)", -20, 100, 0, 5, key='margin_filter')
        
        year_expander = st.expander("Time Period", expanded=True)
        with year_expander:
            selected_years = st.multiselect(
                "Years",
                ['2022', '2023', '2024'],
                default=['2024'],
                key='year_filter'
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        filtered_df = df.copy()
        
        if selected_country != 'All Markets':
            filtered_df = filtered_df[filtered_df['Country'] == selected_country]
        
        if selected_corporation != 'All Corporations':
            filtered_df = filtered_df[filtered_df['Corporation'] == selected_corporation]
        
        if selected_manufacturer != 'All Manufacturers':
            filtered_df = filtered_df[filtered_df['Manufacturer'] == selected_manufacturer]
        
        if selected_molecule != 'All Molecules':
            filtered_df = filtered_df[filtered_df['Molecule'] == selected_molecule]
        
        if selected_specialty != 'All Types':
            filtered_df = filtered_df[filtered_df['Specialty Product'] == selected_specialty]
        
        filtered_df = filtered_df[filtered_df['MAT Q3 2024 USD MNF'] >= min_revenue]
        
        st.markdown(f"""
        <div style="background: rgba(19, 24, 46, 0.7); padding: 16px; border-radius: 12px; margin-top: 20px; border: 1px solid rgba(56, 189, 248, 0.15);">
            <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Filter Summary</div>
            <div style="font-size: 24px; font-weight: 800; color: white; margin: 8px 0;">{len(filtered_df):,}</div>
            <div style="font-size: 14px; color: #cbd5e1;">filtered records</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
                <div>
                    <div style="font-size: 11px; color: #94a3b8;">Revenue</div>
                    <div style="font-size: 14px; font-weight: 600; color: white;">${filtered_df['MAT Q3 2024 USD MNF'].sum():,.0f}</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #94a3b8;">Growth</div>
                    <div style="font-size: 14px; font-weight: 600; color: {'#4ade80' if ((filtered_df['MAT Q3 2024 USD MNF'].sum() - filtered_df['MAT Q3 2023 USD MNF'].sum()) / filtered_df['MAT Q3 2023 USD MNF'].sum() * 100 if filtered_df['MAT Q3 2023 USD MNF'].sum() > 0 else 0) > 0 else '#f87171'}">
                        {((filtered_df['MAT Q3 2024 USD MNF'].sum() - filtered_df['MAT Q3 2023 USD MNF'].sum()) / filtered_df['MAT Q3 2023 USD MNF'].sum() * 100 if filtered_df['MAT Q3 2023 USD MNF'].sum() > 0 else 0):+.1f}%
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Reset All Filters", key='reset_button', use_container_width=True):
            st.rerun()
    
    metrics = compute_comprehensive_metrics(filtered_df)
    
    tab_overview, tab_intelligence, tab_molecules, tab_manufacturers, tab_pricing, tab_explorer = st.tabs([
        "📊 Overview", "🔍 Market Intelligence", "🧬 Molecule Insights", 
        "🏭 Manufacturer Scoring", "💰 Pricing Analytics", "📈 Data Explorer"
    ])
    
    with tab_overview:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(render_kpi_metric(
                "Total Market Value",
                f"${filtered_df['MAT Q3 2024 USD MNF'].sum():,.0f}",
                "2024 Revenue",
                metrics['total_growth_2024'],
                "💰"
            ), unsafe_allow_html=True)
        
        with col2:
            specialty_premium = (
                filtered_df[filtered_df['Specialty Product'].isin(['Specialty', 'Orphan', 'Biologic'])]['MAT Q3 2024 SU Avg Price USD MNF'].mean() /
                filtered_df[filtered_df['Specialty Product'] == 'Non-Specialty']['MAT Q3 2024 SU Avg Price USD MNF'].mean()
                if filtered_df[filtered_df['Specialty Product'] == 'Non-Specialty']['MAT Q3 2024 SU Avg Price USD MNF'].mean() > 0 else 0
            )
            st.markdown(render_kpi_metric(
                "Specialty Premium",
                f"{specialty_premium:.1f}x",
                "Price multiple",
                None,
                "⚡"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(render_kpi_metric(
                "Market Concentration",
                f"{metrics['concentration_top3']:.1f}%",
                "Top 3 share",
                metrics['concentration_top3'] - 35,
                "🎯"
            ), unsafe_allow_html=True)
        
        with col4:
            fragility_color = "#4ade80" if metrics['fragility_score'] < 40 else "#fbbf24" if metrics['fragility_score'] < 70 else "#f87171"
            st.markdown(render_kpi_metric(
                "Growth Fragility",
                f"{metrics['fragility_score']}/100",
                "Risk score",
                None,
                "⚠️"
            ), unsafe_allow_html=True)
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            price_growth = ((filtered_df['MAT Q3 2024 SU Avg Price USD MNF'].mean() - 
                           filtered_df['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                          filtered_df['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if filtered_df['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0
            st.markdown(render_kpi_metric(
                "Price Growth",
                f"{price_growth:+.1f}%",
                "YoY change",
                None,
                "📈"
            ), unsafe_allow_html=True)
        
        with col6:
            volume_growth = ((filtered_df['MAT Q3 2024 Standard Units'].sum() - 
                            filtered_df['MAT Q3 2023 Standard Units'].sum()) / 
                           filtered_df['MAT Q3 2023 Standard Units'].sum() * 100) if filtered_df['MAT Q3 2023 Standard Units'].sum() > 0 else 0
            st.markdown(render_kpi_metric(
                "Volume Growth",
                f"{volume_growth:+.1f}%",
                "YoY change",
                None,
                "📦"
            ), unsafe_allow_html=True)
        
        with col7:
            product_turnover = metrics['product_launches'] + metrics['product_exits']
            st.markdown(render_kpi_metric(
                "Product Turnover",
                f"{product_turnover}",
                f"{metrics['product_launches']}↑ {metrics['product_exits']}↓",
                None,
                "🔄"
            ), unsafe_allow_html=True)
        
        with col8:
            molecule_concentration = (filtered_df.groupby('Molecule')['MAT Q3 2024 USD MNF'].sum().nlargest(5).sum() / 
                                    filtered_df['MAT Q3 2024 USD MNF'].sum() * 100) if filtered_df['MAT Q3 2024 USD MNF'].sum() > 0 else 0
            st.markdown(render_kpi_metric(
                "Top 5 Molecules",
                f"{molecule_concentration:.1f}%",
                "Market share",
                None,
                "🧪"
            ), unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(create_visualization_growth_decomposition(metrics), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_chart2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(create_visualization_specialty_premium(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(create_visualization_market_concentration(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_chart4:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(create_visualization_regional_growth(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        executive_intelligence = generate_executive_intelligence(metrics, filtered_df)
        st.markdown(executive_intelligence)
    
    with tab_intelligence:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        structural_shifts = detect_market_structural_shifts(filtered_df)
        
        if structural_shifts:
            st.markdown("### 🔄 Market Structural Shifts")
            for shift in structural_shifts[:5]:
                severity_icon = "🚨" if shift['severity'] == 'High' else "⚠️" if shift['severity'] == 'Medium' else "ℹ️"
                st.markdown(render_insight_panel(
                    f"{severity_icon} {shift['type']}",
                    shift['description'],
                    'warning' if shift['severity'] in ['High', 'Medium'] else 'info',
                    "🔄",
                    shift['severity']
                ), unsafe_allow_html=True)
        else:
            st.info("No significant structural shifts detected with current filters")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        col_shift1, col_shift2 = st.columns(2)
        
        with col_shift1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(create_visualization_molecule_heatmap(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_shift2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(create_visualization_price_volume_matrix(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        channel_analysis = filtered_df.groupby('Panel').agg({
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum'
        }).reset_index()
        
        channel_analysis['Growth'] = ((channel_analysis['MAT Q3 2024 USD MNF'] - channel_analysis['MAT Q3 2023 USD MNF']) / 
                                     channel_analysis['MAT Q3 2023 USD MNF'] * 100)
        channel_analysis['Share_2024'] = (channel_analysis['MAT Q3 2024 USD MNF'] / 
                                         channel_analysis['MAT Q3 2024 USD MNF'].sum() * 100)
        
        st.markdown("### 🏥 Channel Performance Analysis")
        st.dataframe(
            channel_analysis.sort_values('Growth', ascending=False).style.format({
                'MAT Q3 2024 USD MNF': '${:,.0f}',
                'MAT Q3 2023 USD MNF': '${:,.0f}',
                'Growth': '{:+.1f}%',
                'Share_2024': '{:.1f}%'
            }).background_gradient(subset=['Growth'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    with tab_molecules:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        molecule_insights = perform_molecule_intelligence(filtered_df)
        
        if molecule_insights:
            st.markdown("### 🧬 Molecule Intelligence")
            for insight in molecule_insights[:8]:
                if insight['type'] == 'Structural Growth Molecule':
                    st.markdown(render_insight_panel(
                        "🚀 Structural Growth Molecule",
                        f"{insight['molecule']}: {insight['growth']} growth | {insight['revenue']} revenue",
                        'success',
                        "📈",
                        insight['priority']
                    ), unsafe_allow_html=True)
                elif insight['type'] == 'Relaunch Detection':
                    st.markdown(render_insight_panel(
                        "🔄 Relaunch Detection",
                        f"{insight['count']} molecules showing relaunch signals",
                        'info',
                        "🆕",
                        insight['priority']
                    ), unsafe_allow_html=True)
                elif insight['type'] == 'Market Saturation':
                    st.markdown(render_insight_panel(
                        "📉 Market Saturation",
                        f"{', '.join(insight['molecules'])} showing saturation signs",
                        'warning',
                        "⚠️",
                        insight['priority']
                    ), unsafe_allow_html=True)
                elif insight['type'] == 'Commoditization Risk':
                    st.markdown(render_insight_panel(
                        "⚖️ Commoditization Risk",
                        f"{', '.join(insight['molecules'])} at risk of commoditization",
                        'critical',
                        "💥",
                        insight['priority']
                    ), unsafe_allow_html=True)
        else:
            st.info("No significant molecule insights detected with current filters")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("### 📊 Top Performing Molecules")
        
        top_molecules = filtered_df.groupby('Molecule').agg({
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 Standard Units': 'sum',
            'MAT Q3 2024 SU Avg Price USD MNF': 'mean',
            'Country': 'nunique',
            'Manufacturer': 'nunique'
        }).reset_index()
        
        top_molecules['Growth'] = ((top_molecules['MAT Q3 2024 USD MNF'] - top_molecules['MAT Q3 2023 USD MNF']) / 
                                  top_molecules['MAT Q3 2023 USD MNF'] * 100)
        
        top_molecules['Market_Share'] = (top_molecules['MAT Q3 2024 USD MNF'] / 
                                        top_molecules['MAT Q3 2024 USD MNF'].sum() * 100)
        
        display_cols = ['Molecule', 'MAT Q3 2024 USD MNF', 'Growth', 'Market_Share', 
                       'MAT Q3 2024 SU Avg Price USD MNF', 'Country', 'Manufacturer']
        
        st.dataframe(
            top_molecules.nlargest(15, 'MAT Q3 2024 USD MNF')[display_cols]
            .rename(columns={
                'Molecule': 'Molecule',
                'MAT Q3 2024 USD MNF': 'Revenue 2024',
                'Growth': 'Growth %',
                'Market_Share': 'Market Share %',
                'MAT Q3 2024 SU Avg Price USD MNF': 'Avg Price',
                'Country': 'Countries',
                'Manufacturer': 'Manufacturers'
            })
            .style.format({
                'Revenue 2024': '${:,.0f}',
                'Growth %': '{:+.1f}%',
                'Market Share %': '{:.2f}%',
                'Avg Price': '${:.2f}'
            }).background_gradient(subset=['Growth %'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    with tab_manufacturers:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        manufacturer_scores = calculate_manufacturer_scoring(filtered_df)
        
        st.markdown("### 🏭 Manufacturer Scoring & Risk Assessment")
        
        col_score1, col_score2 = st.columns([2, 1])
        
        with col_score1:
            for score in manufacturer_scores[:10]:
                st.markdown(render_manufacturer_score_card(score), unsafe_allow_html=True)
        
        with col_score2:
            st.markdown('<div class="data-grid">', unsafe_allow_html=True)
            st.markdown("#### 📈 Score Distribution")
            
            grade_dist = Counter([score['grade'] for score in manufacturer_scores])
            grades = ['A', 'B', 'C', 'D', 'F']
            counts = [grade_dist.get(grade, 0) for grade in grades]
            
            fig_grade = go.Figure(data=[go.Bar(
                x=grades,
                y=counts,
                marker_color=['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#dc2626']
            )])
            
            fig_grade.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=False,
                margin=dict(t=30, b=30, l=30, r=30)
            )
            
            st.plotly_chart(fig_grade, use_container_width=True)
            
            avg_score = np.mean([score['total_score'] for score in manufacturer_scores])
            st.metric("Average Score", f"{avg_score:.0f}")
            
            risk_manufacturers = [score for score in manufacturer_scores if score['risk_flags']]
            st.metric("Manufacturers with Risks", f"{len(risk_manufacturers)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("### ⚠️ Risk Flag Analysis")
        
        risk_categories = {}
        for score in manufacturer_scores:
            for risk in score['risk_flags']:
                risk_categories[risk] = risk_categories.get(risk, 0) + 1
        
        if risk_categories:
            risk_df = pd.DataFrame({
                'Risk Category': list(risk_categories.keys()),
                'Count': list(risk_categories.values())
            }).sort_values('Count', ascending=False)
            
            st.dataframe(
                risk_df.style.background_gradient(subset=['Count'], cmap='Reds'),
                use_container_width=True
            )
        else:
            st.success("No significant risk flags detected among top manufacturers")
    
    with tab_pricing:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        pricing_insights = analyze_pricing_mix_dynamics(filtered_df)
        
        if pricing_insights:
            st.markdown("### 💰 Pricing & Mix Intelligence")
            for insight in pricing_insights[:6]:
                if insight['type'] == 'Price Divergence':
                    st.markdown(render_insight_panel(
                        "📊 Price Divergence Alert",
                        f"{insight['molecule']}: SU {insight['su_growth']} vs Unit {insight['unit_growth']} ({insight['gap']})",
                        'warning' if abs(float(insight['gap'].replace('pp', ''))) > 20 else 'info',
                        "⚖️",
                        insight['implication']
                    ), unsafe_allow_html=True)
                elif insight['type'] == 'Pack Size Optimization':
                    st.markdown(render_insight_panel(
                        "📦 Pack Size Optimization",
                        f"{insight['count']} optimization opportunities detected",
                        'success',
                        "🎯",
                        f"Avg Eff: {insight['avg_efficiency']:.2f}"
                    ), unsafe_allow_html=True)
                elif insight['type'] == 'Hidden Discounting':
                    st.markdown(render_insight_panel(
                        "🎭 Hidden Discounting",
                        f"{insight['count']} potential discounting cases",
                        'critical' if insight['count'] > 10 else 'warning',
                        "👁️",
                        f"Avg Ratio: {insight['avg_ratio']:.2f}"
                    ), unsafe_allow_html=True)
        else:
            st.info("No significant pricing insights detected with current filters")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        col_price1, col_price2 = st.columns(2)
        
        with col_price1:
            st.markdown("#### 📈 Price Evolution Analysis")
            
            price_evolution = filtered_df.groupby('Specialty Product').agg({
                'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
                'MAT Q3 2023 SU Avg Price USD MNF': 'mean',
                'MAT Q3 2024 SU Avg Price USD MNF': 'mean'
            }).reset_index()
            
            fig_price = go.Figure()
            
            for idx, row in price_evolution.iterrows():
                fig_price.add_trace(go.Scatter(
                    x=['2022', '2023', '2024'],
                    y=[row['MAT Q3 2022 SU Avg Price USD MNF'], 
                       row['MAT Q3 2023 SU Avg Price USD MNF'],
                       row['MAT Q3 2024 SU Avg Price USD MNF']],
                    name=row['Specialty Product'],
                    mode='lines+markers',
                    line=dict(width=3)
                ))
            
            fig_price.update_layout(
                title='Price Evolution by Product Type',
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col_price2:
            st.markdown("#### 📊 Price-Volume Correlation")
            
            molecule_corr = filtered_df.groupby('Molecule').apply(lambda x: pd.Series({
                'price_growth': ((x['MAT Q3 2024 SU Avg Price USD MNF'].mean() - x['MAT Q3 2023 SU Avg Price USD MNF'].mean()) / 
                                x['MAT Q3 2023 SU Avg Price USD MNF'].mean() * 100) if x['MAT Q3 2023 SU Avg Price USD MNF'].mean() > 0 else 0,
                'volume_growth': ((x['MAT Q3 2024 Standard Units'].sum() - x['MAT Q3 2023 Standard Units'].sum()) / 
                                 x['MAT Q3 2023 Standard Units'].sum() * 100) if x['MAT Q3 2023 Standard Units'].sum() > 0 else 0,
                'revenue': x['MAT Q3 2024 USD MNF'].sum()
            })).reset_index()
            
            fig_corr = px.scatter(
                molecule_corr[molecule_corr['revenue'] > 100000],
                x='price_growth',
                y='volume_growth',
                size='revenue',
                hover_name='Molecule',
                title='Price vs Volume Growth Correlation',
                labels={
                    'price_growth': 'Price Growth (%)',
                    'volume_growth': 'Volume Growth (%)',
                    'revenue': 'Revenue Size'
                }
            )
            
            fig_corr.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab_explorer:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            show_rows = st.selectbox("Rows to Display", [100, 500, 1000, 5000], index=0, key='show_rows')
        
        with col_exp2:
            sort_field = st.selectbox(
                "Sort By",
                ['MAT Q3 2024 USD MNF', 'MAT Q3 2024 Standard Units', 
                 'MAT Q3 2024 SU Avg Price USD MNF', 'MAT Q3 2024 Unit Avg Price USD MNF',
                 'MAT Q3 2024 Growth Rate'],
                index=0,
                key='sort_field'
            )
        
        with col_exp3:
            sort_direction = st.selectbox("Sort Order", ['Descending', 'Ascending'], index=0, key='sort_dir')
        
        display_data = filtered_df.copy()
        
        if 'MAT Q3 2024 Growth Rate' not in display_data.columns:
            display_data['MAT Q3 2024 Growth Rate'] = (
                (display_data['MAT Q3 2024 USD MNF'] - display_data['MAT Q3 2023 USD MNF']) / 
                display_data['MAT Q3 2023 USD MNF'] * 100
            )
        
        display_data = display_data.sort_values(
            sort_field if sort_field != 'MAT Q3 2024 Growth Rate' else 'MAT Q3 2024 Growth Rate',
            ascending=(sort_direction == 'Ascending')
        ).head(show_rows)
        
        st.markdown('<div class="data-grid">', unsafe_allow_html=True)
        st.dataframe(display_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            csv_data = filtered_df.to_csv(index=False).encode()
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv_data,
                file_name="pharma_intelligence_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_dl2:
            sample_size = st.slider("Sample Size", 1000, 50000, 10000, 1000, key='sample_size')
            sample_data = filtered_df.sample(min(sample_size, len(filtered_df))).to_csv(index=False).encode()
            st.download_button(
                label=f"📥 Download Sample ({sample_size} rows)",
                data=sample_data,
                file_name=f"pharma_sample_{sample_size}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("### 📊 Data Quality Report")
        
        col_qual1, col_qual2, col_qual3 = st.columns(3)
        
        with col_qual1:
            missing_values = filtered_df.isnull().sum().sum()
            total_cells = filtered_df.size
            missing_pct = (missing_values / total_cells) * 100
            
            st.metric(
                "Missing Values",
                f"{missing_pct:.2f}%",
                delta=f"{missing_values:,} cells",
                delta_color="inverse"
            )
        
        with col_qual2:
            zero_revenue = len(filtered_df[filtered_df['MAT Q3 2024 USD MNF'] == 0])
            st.metric(
                "Zero Revenue Products",
                f"{zero_revenue}",
                delta=f"{(zero_revenue/len(filtered_df)*100):.1f}%",
                delta_color="inverse"
            )
        
        with col_qual3:
            data_types = filtered_df.dtypes.value_counts()
            numeric_cols = len([col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col])])
            st.metric(
                "Numeric Columns",
                f"{numeric_cols}/{len(filtered_df.columns)}",
                delta=f"{(numeric_cols/len(filtered_df.columns)*100):.1f}%"
            )

if __name__ == "__main__":
    main()
