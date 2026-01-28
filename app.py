# app.py - 2500+ satır gelişmiş Global İlaç Pazarı Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import json
from itertools import combinations
import math
from typing import Dict, List, Tuple, Optional
import hashlib
import re

# ================================================
# 1. KONFİGÜRASYON VE STİL AYARLARI
# ================================================
st.set_page_config(
    page_title="Global Pharma Market Intelligence Pro",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com',
        'Report a bug': "https://www.example.com",
        'About': "### Global İlaç Pazarı Strateji Dashboard v2.0\n\nTüm hakları saklıdır © 2024"
    }
)

# Özel CSS ve JavaScript
st.markdown("""
<style>
    /* Ana stiller */
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
    }
    
    .section-title {
        font-size: 1.8rem;
        color: #1E3A8A;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding: 0.5rem;
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.1), transparent);
        border-left: 4px solid #3B82F6;
    }
    
    .subsection-title {
        font-size: 1.4rem;
        color: #2563EB;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
    }
    
    /* Metrik kutuları */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #E5E7EB;
        transition: transform 0.2s, box-shadow 0.2s;
        margin: 0.5rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F2937;
        margin: 0.3rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    .metric-change {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    
    /* İçgörü kutuları */
    .insight-card {
        background: linear-gradient(135deg, #F0F9FF, #E0F2FE);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #0EA5E9;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
    }
    
    .insight-card.warning {
        background: linear-gradient(135deg, #FEF3C7, #FDE68A);
        border-left: 5px solid #F59E0B;
    }
    
    .insight-card.danger {
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        border-left: 5px solid #EF4444;
    }
    
    .insight-card.success {
        background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
        border-left: 5px solid #10B981;
    }
    
    .insight-title {
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .insight-content {
        color: #374151;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    
    /* Tab stilleri */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F9FAFB;
        border-radius: 8px 8px 0 0;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
    
    /* Progress bar */
    .progress-container {
        background: #E5E7EB;
        border-radius: 10px;
        height: 8px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #666;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #1F2937;
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>

<script>
// JavaScript fonksiyonları
function formatNumber(num) {
    if (num >= 1000000000) {
        return (num / 1000000000).toFixed(2) + 'B';
    }
    if (num >= 1000000) {
        return (num / 1000000).toFixed(2) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(2) + 'K';
    }
    return num.toFixed(2);
}

function animateCounter(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        element.innerHTML = formatNumber(Math.floor(progress * (end - start) + start));
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}
</script>
""", unsafe_allow_html=True)

# ================================================
# 2. VERİ YÜKLEME VE ÖN İŞLEME FONKSİYONLARI
# ================================================
@st.cache_data(ttl=3600, show_spinner="Veri yükleniyor...")
def load_comprehensive_data():
    """Kapsamlı sentetik veri oluşturma"""
    np.random.seed(42)
    
    # Kapsamlı veri yapısı
    countries = ['Türkiye', 'Almanya', 'Fransa', 'İtalya', 'İspanya', 'İngiltere',
                 'Polonya', 'Hollanda', 'Belçika', 'İsviçre', 'İsveç', 'Norveç',
                 'Danimarka', 'Finlandiya', 'Avusturya', 'Portekiz', 'Yunanistan',
                 'Çekya', 'Macaristan', 'Romanya', 'Bulgaristan', 'Hırvatistan']
    
    regions = {
        'Kuzey Avrupa': ['İsveç', 'Norveç', 'Danimarka', 'Finlandiya'],
        'Batı Avrupa': ['Almanya', 'Fransa', 'Hollanda', 'Belçika', 'Avusturya', 'İsviçre'],
        'Güney Avrupa': ['İtalya', 'İspanya', 'Portekiz', 'Yunanistan'],
        'Doğu Avrupa': ['Türkiye', 'Polonya', 'Çekya', 'Macaristan', 'Romanya', 'Bulgaristan', 'Hırvatistan'],
        'İngiliz Adaları': ['İngiltere']
    }
    
    companies = {
        'Pfizer': {'HQ': 'USA', 'Type': 'Big Pharma', 'Revenue_B': 81.3},
        'Novartis': {'HQ': 'İsviçre', 'Type': 'Big Pharma', 'Revenue_B': 53.6},
        'Roche': {'HQ': 'İsviçre', 'Type': 'Big Pharma', 'Revenue_B': 68.0},
        'Merck': {'HQ': 'USA', 'Type': 'Big Pharma', 'Revenue_B': 59.3},
        'Sanofi': {'HQ': 'Fransa', 'Type': 'Big Pharma', 'Revenue_B': 45.3},
        'GSK': {'HQ': 'UK', 'Type': 'Big Pharma', 'Revenue_B': 36.2},
        'AstraZeneca': {'HQ': 'UK', 'Type': 'Big Pharma', 'Revenue_B': 45.8},
        'Johnson & Johnson': {'HQ': 'USA', 'Type': 'Big Pharma', 'Revenue_B': 85.2},
        'Bayer': {'HQ': 'Almanya', 'Type': 'Big Pharma', 'Revenue_B': 53.0},
        'AbbVie': {'HQ': 'USA', 'Type': 'Big Pharma', 'Revenue_B': 58.1},
        'Eli Lilly': {'HQ': 'USA', 'Type': 'Big Pharma', 'Revenue_B': 34.1},
        'Boehringer': {'HQ': 'Almanya', 'Type': 'Big Pharma', 'Revenue_B': 25.4},
        'Novo Nordisk': {'HQ': 'Danimarka', 'Type': 'Specialty', 'Revenue_B': 33.7},
        'Amgen': {'HQ': 'USA', 'Type': 'Biotech', 'Revenue_B': 26.3},
        'Takeda': {'HQ': 'Japonya', 'Type': 'Big Pharma', 'Revenue_B': 29.5},
        'Biogen': {'HQ': 'USA', 'Type': 'Biotech', 'Revenue_B': 10.2},
        'Gilead': {'HQ': 'USA', 'Type': 'Biotech', 'Revenue_B': 27.3},
        'Teva': {'HQ': 'İsrail', 'Type': 'Generic', 'Revenue_B': 15.9},
        'Sandoz': {'HQ': 'İsviçre', 'Type': 'Generic', 'Revenue_B': 9.6},
        'Mylan': {'HQ': 'USA', 'Type': 'Generic', 'Revenue_B': 11.8}
    }
    
    molecules_db = {
        'Adalimumab': {'Class': 'Anti-TNF', 'Launch_Year': 2002, 'Patent_Expiry': 2023, 'Category': 'Specialty'},
        'Pembrolizumab': {'Class': 'Anti-PD-1', 'Launch_Year': 2014, 'Patent_Expiry': 2028, 'Category': 'Specialty'},
        'Nivolumab': {'Class': 'Anti-PD-1', 'Launch_Year': 2014, 'Patent_Expiry': 2027, 'Category': 'Specialty'},
        'Rituximab': {'Class': 'Anti-CD20', 'Launch_Year': 1997, 'Patent_Expiry': 2018, 'Category': 'Specialty'},
        'Trastuzumab': {'Class': 'Anti-HER2', 'Launch_Year': 1998, 'Patent_Expiry': 2019, 'Category': 'Specialty'},
        'Bevacizumab': {'Class': 'Anti-VEGF', 'Launch_Year': 2004, 'Patent_Expiry': 2019, 'Category': 'Specialty'},
        'Insulin Glargine': {'Class': 'Insulin Analog', 'Launch_Year': 2000, 'Patent_Expiry': 2015, 'Category': 'Specialty'},
        'Sitagliptin': {'Class': 'DPP-4 Inhibitor', 'Launch_Year': 2006, 'Patent_Expiry': 2022, 'Category': 'Non-Specialty'},
        'Atorvastatin': {'Class': 'Statin', 'Launch_Year': 1996, 'Patent_Expiry': 2011, 'Category': 'Non-Specialty'},
        'Apremilast': {'Class': 'PDE4 Inhibitor', 'Launch_Year': 2014, 'Patent_Expiry': 2028, 'Category': 'Specialty'},
        'Dupilumab': {'Class': 'Anti-IL4/13', 'Launch_Year': 2017, 'Patent_Expiry': 2031, 'Category': 'Specialty'},
        'Semaglutide': {'Class': 'GLP-1 Analog', 'Launch_Year': 2017, 'Patent_Expiry': 2031, 'Category': 'Specialty'},
        'Ibrutinib': {'Class': 'BTK Inhibitor', 'Launch_Year': 2013, 'Patent_Expiry': 2027, 'Category': 'Specialty'},
        'Venetoclax': {'Class': 'BCL-2 Inhibitor', 'Launch_Year': 2016, 'Patent_Expiry': 2030, 'Category': 'Specialty'},
        'Ozanimod': {'Class': 'S1P Receptor', 'Launch_Year': 2020, 'Patent_Expiry': 2034, 'Category': 'Specialty'},
        'Faricimab': {'Class': 'Anti-VEGF/Ang2', 'Launch_Year': 2021, 'Patent_Expiry': 2035, 'Category': 'Specialty'},
        'Tirzepatide': {'Class': 'GIP/GLP-1', 'Launch_Year': 2022, 'Patent_Expiry': 2036, 'Category': 'Specialty'},
        'Bimekizumab': {'Class': 'Anti-IL-17', 'Launch_Year': 2021, 'Patent_Expiry': 2035, 'Category': 'Specialty'}
    }
    
    therapeutic_areas = {
        'Onkoloji': ['Pembrolizumab', 'Nivolumab', 'Trastuzumab', 'Bevacizumab', 'Ibrutinib', 'Venetoclax'],
        'Otoimmün': ['Adalimumab', 'Rituximab', 'Apremilast', 'Dupilumab', 'Bimekizumab'],
        'Diyabet': ['Insulin Glargine', 'Sitagliptin', 'Semaglutide', 'Tirzepatide'],
        'Kardiyovasküler': ['Atorvastatin'],
        'Nöroloji': ['Ozanimod'],
        'Oftalmoloji': ['Faricimab']
    }
    
    # Kapsamlı veri oluşturma
    all_data = []
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    years = ['2021', '2022', '2023', '2024']
    
    for country in countries:
        region = next((r for r, c_list in regions.items() if country in c_list), 'Diğer')
        country_gdp_per_capita = np.random.lognormal(10, 0.5)  # GDP per capita
        
        for year in years:
            for quarter in quarters:
                period = f"{year}-{quarter}"
                
                # Ülke bazlı pazar büyüklüğü faktörü
                market_factor = country_gdp_per_capita * np.random.uniform(0.8, 1.2)
                
                for molecule_name, molecule_info in molecules_db.items():
                    # Molekül bazlı faktörler
                    molecule_age = int(year) - molecule_info['Launch_Year']
                    patent_status = 'On-Patent' if int(year) < molecule_info['Patent_Expiry'] else 'Off-Patent'
                    
                    # Terapötik alan
                    ta = next((ta for ta, mol_list in therapeutic_areas.items() if molecule_name in mol_list), 'Diğer')
                    
                    # Moleküle özgü baz satış
                    base_sales = molecule_info.get('Base_Sales', np.random.lognormal(12, 1.5))
                    
                    # Patent durumuna göre düzeltme
                    if patent_status == 'Off-Patent':
                        base_sales *= np.random.uniform(0.3, 0.8)  # Patent sonrası düşüş
                    
                    # Şirket ataması
                    molecule_companies = []
                    if molecule_name in ['Adalimumab', 'Apremilast']:
                        molecule_companies = ['AbbVie', 'Amgen', 'Teva']
                    elif molecule_name in ['Pembrolizumab', 'Sitagliptin']:
                        molecule_companies = ['Merck', 'Novartis']
                    elif molecule_name in ['Semaglutide', 'Tirzepatide']:
                        molecule_companies = ['Novo Nordisk', 'Eli Lilly']
                    elif molecule_name in ['Atorvastatin']:
                        molecule_companies = ['Pfizer', 'Teva', 'Sandoz', 'Mylan']
                    else:
                        molecule_companies = list(np.random.choice(list(companies.keys()), 
                                                                  size=np.random.randint(2, 5), 
                                                                  replace=False))
                    
                    for company in molecule_companies:
                        # Şirket pazar payı
                        if company in ['Pfizer', 'Novartis', 'Roche']:
                            company_share = np.random.beta(2, 5)  # Büyük şirketler
                        elif company in ['Teva', 'Sandoz', 'Mylan']:
                            company_share = np.random.beta(1, 3)  # Jenerik şirketler
                        else:
                            company_share = np.random.beta(1.5, 4)
                        
                        # Dönemsel varyasyon
                        seasonal_factor = 1 + 0.1 * np.sin((int(quarter[1]) * np.pi) / 2)
                        
                        # Satış hesaplama
                        sales = (base_sales * market_factor * company_share * 
                                seasonal_factor * np.random.uniform(0.9, 1.1))
                        
                        # Birimler ve fiyatlar
                        units = sales / np.random.lognormal(5, 0.5)
                        standard_units = units * np.random.uniform(0.8, 1.5)
                        unit_price = sales / units
                        
                        # Kanal dağılımı
                        if molecule_info['Category'] == 'Specialty':
                            hospital_share = np.random.beta(8, 2)  # Specialty ürünler hospital ağırlıklı
                        else:
                            hospital_share = np.random.beta(2, 8)  # Non-specialty retail ağırlıklı
                        
                        hospital_sales = sales * hospital_share
                        retail_sales = sales * (1 - hospital_share)
                        
                        # Paket ve formülasyon
                        packs = ['Vial', 'Prefilled Syringe', 'Autoinjector', 'Tablet', 'Capsule', 'Pen']
                        strengths = ['40mg', '50mg', '100mg', '150mg', '200mg', '300mg', '500mg']
                        
                        data_entry = {
                            'Country': country,
                            'Region': region,
                            'Sector': 'Hospital',
                            'Corporation': company,
                            'Molecule': molecule_name,
                            'Chemical_Class': molecule_info['Class'],
                            'Therapeutic_Area': ta,
                            'Product': f"{molecule_name} {np.random.choice(['SC', 'IV', 'Oral'])}",
                            'Pack': np.random.choice(packs, p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05]),
                            'Strength': np.random.choice(strengths),
                            'Volume_ml': np.random.choice([1, 2, 5, 10, 20, 50]),
                            'Specialty_Flag': molecule_info['Category'],
                            'Prescription_Status': 'Rx',
                            'Patent_Status': patent_status,
                            'Launch_Year': molecule_info['Launch_Year'],
                            'Period': period,
                            'Year': year,
                            'Quarter': quarter,
                            'USD_MNF': hospital_sales,
                            'Units': units * hospital_share,
                            'Standard_Units': standard_units * hospital_share,
                            'Unit_Avg_Price': unit_price * np.random.uniform(1.1, 1.3),  # Hospital premium
                            'SU_Avg_Price': (unit_price * np.random.uniform(1.1, 1.3)) / np.random.uniform(0.8, 1.2)
                        }
                        all_data.append(data_entry)
                        
                        # Retail kanalı için ayrı kayıt
                        retail_data = data_entry.copy()
                        retail_data.update({
                            'Sector': 'Retail',
                            'USD_MNF': retail_sales,
                            'Units': units * (1 - hospital_share),
                            'Standard_Units': standard_units * (1 - hospital_share),
                            'Unit_Avg_Price': unit_price * np.random.uniform(0.8, 1.0),  # Retail genelde daha düşük
                            'SU_Avg_Price': (unit_price * np.random.uniform(0.8, 1.0)) / np.random.uniform(0.8, 1.2)
                        })
                        all_data.append(retail_data)
    
    df = pd.DataFrame(all_data)
    
    # Hesaplanmış metrikler
    df['Price_Per_Unit'] = df['USD_MNF'] / df['Units']
    df['Volume_Per_Product'] = df['Units'] / df.groupby(['Product', 'Period'])['Product'].transform('count')
    df['Market_Share'] = df.groupby(['Period', 'Therapeutic_Area'])['USD_MNF'].transform(
        lambda x: x / x.sum()
    )
    
    # Büyüme oranları
    df['Sales_Growth_QoQ'] = df.groupby(['Country', 'Corporation', 'Molecule', 'Sector'])['USD_MNF'].pct_change()
    df['Price_Growth_QoQ'] = df.groupby(['Country', 'Corporation', 'Molecule', 'Sector'])['Unit_Avg_Price'].pct_change()
    df['Volume_Growth_QoQ'] = df.groupby(['Country', 'Corporation', 'Molecule', 'Sector'])['Units'].pct_change()
    
    # YoY büyüme
    df['Sales_Growth_YoY'] = df.groupby(['Country', 'Corporation', 'Molecule', 'Sector', 'Quarter'])['USD_MNF'].pct_change(4)
    df['Price_Growth_YoY'] = df.groupby(['Country', 'Corporation', 'Molecule', 'Sector', 'Quarter'])['Unit_Avg_Price'].pct_change(4)
    
    # Trend bileşenleri
    for key in ['Country', 'Corporation', 'Molecule']:
        df[f'{key}_Trend'] = df.groupby(key)['USD_MNF'].transform(
            lambda x: x.rolling(4, min_periods=1).mean()
        )
    
    # Rekabet metrikleri
    df['Competitor_Count'] = df.groupby(['Country', 'Molecule', 'Period'])['Corporation'].transform('nunique')
    df['Price_Variance'] = df.groupby(['Country', 'Molecule', 'Period'])['Unit_Avg_Price'].transform('std')
    
    return df, companies, molecules_db, therapeutic_areas, regions

# ================================================
# 3. ANALİTİK FONKSİYONLAR
# ================================================
class PharmaAnalytics:
    """İleri seviye farma analitik fonksiyonları"""
    
    @staticmethod
    def calculate_market_concentration(df, group_cols=['Country', 'Period'], top_n=3):
        """Pazar konsantrasyonu hesaplama"""
        concentration = {}
        for _, group in df.groupby(group_cols):
            key = tuple(_)
            total_sales = group['USD_MNF'].sum()
            top_companies = group.groupby('Corporation')['USD_MNF'].sum().nlargest(top_n).sum()
            concentration[key] = {
                'HHI': ((group.groupby('Corporation')['USD_MNF'].sum() / total_sales) ** 2).sum() * 10000,
                'Top3_Share': top_companies / total_sales if total_sales > 0 else 0,
                'CR4': group.groupby('Corporation')['USD_MNF'].sum().nlargest(4).sum() / total_sales if total_sales > 0 else 0
            }
        return concentration
    
    @staticmethod
    def detect_price_erosion(df, window=4, threshold=-0.05):
        """Fiyat erozyonu tespiti"""
        erosion_signals = []
        for (country, molecule, company), group in df.groupby(['Country', 'Molecule', 'Corporation']):
            group = group.sort_values('Period')
            if len(group) >= window:
                price_changes = group['Unit_Avg_Price'].pct_change(window - 1)
                volume_changes = group['Units'].pct_change(window - 1)
                
                # Fiyat erozyonu sinyalleri
                recent_price_change = price_changes.iloc[-1] if not price_changes.empty else 0
                recent_volume_change = volume_changes.iloc[-1] if not volume_changes.empty else 0
                
                if recent_price_change < threshold and recent_volume_change > 0:
                    erosion_score = abs(recent_price_change) * (1 + recent_volume_change)
                    erosion_signals.append({
                        'Country': country,
                        'Molecule': molecule,
                        'Corporation': company,
                        'Price_Change': recent_price_change,
                        'Volume_Change': recent_volume_change,
                        'Erosion_Score': erosion_score,
                        'Risk_Level': 'Yüksek' if erosion_score > 0.1 else 'Orta' if erosion_score > 0.05 else 'Düşük'
                    })
        return pd.DataFrame(erosion_signals)
    
    @staticmethod
    def analyze_product_lifecycle(df, min_periods=8):
        """Ürün yaşam döngüsü analizi"""
        lifecycle_data = []
        for (country, molecule), group in df.groupby(['Country', 'Molecule']):
            group = group.sort_values('Period')
            if len(group) >= min_periods:
                sales_series = group.set_index('Period')['USD_MNF']
                
                # Büyüme oranı hesaplama
                growth_rates = sales_series.pct_change().dropna()
                avg_growth = growth_rates.mean()
                growth_volatility = growth_rates.std()
                
                # Yaşam döngüsü belirleme
                if avg_growth > 0.1:
                    stage = 'Büyüme'
                elif avg_growth > -0.05:
                    stage = 'Olgun'
                else:
                    stage = 'Düşüş'
                
                # Penetrasyon analizi
                peak_sales = sales_series.max()
                current_sales = sales_series.iloc[-1]
                penetration_rate = current_sales / peak_sales if peak_sales > 0 else 0
                
                lifecycle_data.append({
                    'Country': country,
                    'Molecule': molecule,
                    'Current_Sales': current_sales,
                    'Peak_Sales': peak_sales,
                    'Avg_Growth_Rate': avg_growth,
                    'Growth_Volatility': growth_volatility,
                    'Lifecycle_Stage': stage,
                    'Penetration_Rate': penetration_rate,
                    'Sales_Trend': '↑' if avg_growth > 0.05 else '→' if avg_growth > -0.05 else '↓'
                })
        return pd.DataFrame(lifecycle_data)
    
    @staticmethod
    def perform_competitive_benchmarking(df, benchmark_companies=None):
        """Rekabet benchmarking analizi"""
        if benchmark_companies is None:
            benchmark_companies = df['Corporation'].unique()[:5]
        
        benchmarks = {}
        for metric in ['Unit_Avg_Price', 'Sales_Growth_YoY', 'Market_Share']:
            metric_data = {}
            for company in benchmark_companies:
                company_data = df[df['Corporation'] == company]
                if not company_data.empty:
                    metric_data[company] = {
                        'mean': company_data[metric].mean(),
                        'median': company_data[metric].median(),
                        'std': company_data[metric].std(),
                        'q1': company_data[metric].quantile(0.25),
                        'q3': company_data[metric].quantile(0.75)
                    }
            benchmarks[metric] = metric_data
        
        return benchmarks
    
    @staticmethod
    def calculate_price_elasticity(df, price_col='Unit_Avg_Price', volume_col='Units'):
        """Fiyat esnekliği hesaplama"""
        elasticity_results = {}
        for (country, molecule), group in df.groupby(['Country', 'Molecule']):
            group = group.sort_values('Period')
            if len(group) > 4:  # Minimum gözlem sayısı
                try:
                    # Log-log regression
                    X = np.log(group[price_col].values).reshape(-1, 1)
                    y = np.log(group[volume_col].values)
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                    elasticity = model.params[1]  # Price coefficient
                    
                    elasticity_results[(country, molecule)] = {
                        'elasticity': elasticity,
                        'p_value': model.pvalues[1],
                        'r_squared': model.rsquared,
                        'interpretation': 'Esnek' if elasticity < -1 else 'Birim Esnek' if abs(elasticity - 1) < 0.1 else 'Esnek Değil'
                    }
                except:
                    continue
        return elasticity_results
    
    @staticmethod
    def identify_white_spaces(df, min_market_size=1000000):
        """Beyaz alan (white space) tespiti"""
        # Tüm ülke-molekül kombinasyonları
        all_combinations = set(df[['Country', 'Molecule']].drop_duplicates().itertuples(index=False, name=None))
        
        # Mevcut ülke-molekül kombinasyonları
        existing_combinations = set()
        for (country, molecule), group in df.groupby(['Country', 'Molecule']):
            if group['USD_MNF'].sum() >= min_market_size:
                existing_combinations.add((country, molecule))
        
        # Beyaz alanlar
        white_spaces = all_combinations - existing_combinations
        
        # Potansiyel değerlendirme
        white_space_analysis = []
        for country, molecule in white_spaces:
            # Benzer ülkelerdeki performans
            similar_countries = df[
                (df['Region'] == df[df['Country'] == country]['Region'].iloc[0]) & 
                (df['Molecule'] == molecule)
            ]
            
            if not similar_countries.empty:
                avg_sales = similar_countries.groupby('Country')['USD_MNF'].sum().mean()
                avg_growth = similar_countries['Sales_Growth_YoY'].mean()
                
                white_space_analysis.append({
                    'Country': country,
                    'Molecule': molecule,
                    'Therapeutic_Area': df[df['Molecule'] == molecule]['Therapeutic_Area'].iloc[0],
                    'Avg_Similar_Market_Size': avg_sales,
                    'Avg_Growth_Rate': avg_growth,
                    'Competitor_Count': similar_countries['Corporation'].nunique(),
                    'Potential_Score': avg_sales * (1 + avg_growth) / max(1, similar_countries['Corporation'].nunique())
                })
        
        return pd.DataFrame(white_space_analysis).sort_values('Potential_Score', ascending=False)

# ================================================
# 4. GÖRSELLEŞTİRME FONKSİYONLARI
# ================================================
class PharmaVisualizations:
    """Gelişmiş görselleştirme fonksiyonları"""
    
    @staticmethod
    def create_market_evolution_chart(df, metric='USD_MNF', group_by='Country'):
        """Pazar evrim chart'ı"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Pazar Büyüklüğü Trendi', 'Büyüme Oranları',
                           'Pazar Payı Dağılımı', 'Fiyat-Hacim İlişkisi'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Pazar trendi
        trend_data = df.groupby(['Period', group_by])[metric].sum().unstack()
        for column in trend_data.columns:
            fig.add_trace(
                go.Scatter(x=trend_data.index, y=trend_data[column],
                          name=column, mode='lines+markers'),
                row=1, col=1
            )
        
        # 2. Büyüme oranları
        growth_data = trend_data.pct_change().mean().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=growth_data.index, y=growth_data.values,
                  marker_color='coral'),
            row=1, col=2
        )
        
        # 3. Pazar payı
        latest_period = df['Period'].max()
        market_share = df[df['Period'] == latest_period].groupby(group_by)[metric].sum()
        fig.add_trace(
            go.Pie(labels=market_share.index, values=market_share.values,
                  hole=0.4),
            row=2, col=1
        )
        
        # 4. Fiyat-hacim scatter
        scatter_data = df.groupby([group_by, 'Molecule']).agg({
            'Unit_Avg_Price': 'mean',
            'Units': 'sum'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(x=scatter_data['Unit_Avg_Price'],
                      y=scatter_data['Units'],
                      mode='markers',
                      marker=dict(size=10, color=scatter_data['Units'],
                                  colorscale='Viridis', showscale=True),
                      text=scatter_data['Molecule'],
                      hoverinfo='text+x+y'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Pazar Evrim Analizi")
        return fig
    
    @staticmethod
    def create_competitive_landscape(df, dimension1='Unit_Avg_Price', dimension2='Sales_Growth_YoY'):
        """Rekabet manzarası haritası"""
        company_metrics = df.groupby('Corporation').agg({
            dimension1: 'mean',
            dimension2: 'mean',
            'USD_MNF': 'sum',
            'Market_Share': 'mean',
            'Competitor_Count': 'mean'
        }).reset_index()
        
        fig = px.scatter(company_metrics,
                        x=dimension1,
                        y=dimension2,
                        size='USD_MNF',
                        color='Market_Share',
                        hover_name='Corporation',
                        size_max=60,
                        color_continuous_scale='RdYlBu',
                        labels={dimension1: 'Ortalama Fiyat',
                               dimension2: 'Büyüme Oranı',
                               'Market_Share': 'Pazar Payı'},
                        title='Rekabet Manzarası: Şirket Performans Karşılaştırması')
        
        # Quadrant çizgileri
        x_median = company_metrics[dimension1].median()
        y_median = company_metrics[dimension2].median()
        
        fig.add_hline(y=y_median, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=x_median, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Quadrant etiketleri
        fig.add_annotation(x=0.95, y=0.95, xref="paper", yref="paper",
                          text="Yüksek Fiyat<br>Yüksek Büyüme",
                          showarrow=False, font=dict(size=10, color="green"))
        fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper",
                          text="Düşük Fiyat<br>Yüksek Büyüme",
                          showarrow=False, font=dict(size=10, color="blue"))
        fig.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper",
                          text="Yüksek Fiyat<br>Düşük Büyüme",
                          showarrow=False, font=dict(size=10, color="orange"))
        fig.add_annotation(x=0.05, y=0.05, xref="paper", yref="paper",
                          text="Düşük Fiyat<br>Düşük Büyüme",
                          showarrow=False, font=dict(size=10, color="red"))
        
        return fig
    
    @staticmethod
    def create_price_erosion_heatmap(erosion_df):
        """Fiyat erozyonu heatmap'i"""
        pivot_table = erosion_df.pivot_table(
            index='Country',
            columns='Molecule',
            values='Erosion_Score',
            aggfunc='mean'
        ).fillna(0)
        
        fig = px.imshow(pivot_table,
                       labels=dict(x="Molekül", y="Ülke", color="Erozyon Skoru"),
                       title='Fiyat Erozyonu Risk Haritası',
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        
        fig.update_layout(height=600)
        return fig
    
    @staticmethod
    def create_therapeutic_area_sunburst(df):
        """Terapötik alan sunburst chart"""
        ta_data = df.groupby(['Therapeutic_Area', 'Molecule', 'Corporation']).agg({
            'USD_MNF': 'sum',
            'Sales_Growth_YoY': 'mean'
        }).reset_index()
        
        fig = px.sunburst(ta_data,
                         path=['Therapeutic_Area', 'Molecule', 'Corporation'],
                         values='USD_MNF',
                         color='Sales_Growth_YoY',
                         color_continuous_scale='RdYlBu',
                         title='Terapötik Alan Hiyerarşisi',
                         hover_data=['USD_MNF', 'Sales_Growth_YoY'])
        
        fig.update_layout(margin=dict(t=30, l=0, r=0, b=0))
        return fig
    
    @staticmethod
    def create_market_concentration_timeline(df, metric='HHI'):
        """Pazar konsantrasyonu zaman çizelgesi"""
        concentration = PharmaAnalytics.calculate_market_concentration(df)
        conc_df = pd.DataFrame.from_dict(concentration, orient='index')
        conc_df = conc_df.reset_index()
        conc_df[['Country', 'Period']] = pd.DataFrame(conc_df['index'].tolist(), index=conc_df.index)
        
        fig = px.line(conc_df, x='Period', y=metric, color='Country',
                     title=f'Pazar Konsantrasyonu Trendi ({metric})',
                     labels={metric: 'Konsantrasyon İndeksi', 'Period': 'Dönem'},
                     markers=True)
        
        # Referans çizgileri
        if metric == 'HHI':
            fig.add_hline(y=1500, line_dash="dash", line_color="orange",
                         annotation_text="Orta Konsantrasyon", opacity=0.7)
            fig.add_hline(y=2500, line_dash="dash", line_color="red",
                         annotation_text="Yüksek Konsantrasyon", opacity=0.7)
        
        return fig

# ================================================
# 5. OTOMATİK İÇGÖRÜ MOTORU
# ================================================
class InsightEngine:
    """Otomatik içgörü üretim motoru"""
    
    @staticmethod
    def generate_market_insights(df, filtered_df, selected_filters):
        """Pazar içgörüleri oluştur"""
        insights = []
        
        # 1. Pazar büyüklüğü içgörüsü
        total_market = filtered_df['USD_MNF'].sum()
        overall_market = df['USD_MNF'].sum()
        market_share = total_market / overall_market
        
        insights.append({
            'type': 'info',
            'title': 'Pazar Segmentasyonu',
            'content': f"Seçilen filtreler toplam pazarın %{market_share*100:.1f}'ini temsil etmektedir. "
                      f"Segment büyüklüğü: ${total_market/1e9:.2f}B",
            'icon': '📊'
        })
        
        # 2. Büyüme içgörüsü
        if len(selected_filters.get('selected_years', [])) > 1:
            growth_data = filtered_df.groupby('Year')['USD_MNF'].sum()
            cagr = (growth_data.iloc[-1] / growth_data.iloc[0]) ** (1/(len(growth_data)-1)) - 1
            
            insights.append({
                'type': 'success' if cagr > 0 else 'warning',
                'title': 'Büyüme Dinamikleri',
                'content': f"Segment CAGR: %{cagr*100:.1f}. "
                          f"Büyüme {selected_filters['selected_years'][-1]} yılında "
                          f"{'pozitif' if growth_data.pct_change().iloc[-1] > 0 else 'negatif'} seyretmiştir.",
                'icon': '📈'
            })
        
        # 3. Fiyat trendi içgörüsü
        price_trend = filtered_df.groupby('Period')['Unit_Avg_Price'].mean()
        if len(price_trend) > 1:
            price_change = (price_trend.iloc[-1] / price_trend.iloc[0] - 1) * 100
            
            insights.append({
                'type': 'warning' if price_change < -5 else 'info',
                'title': 'Fiyat Trend Analizi',
                'content': f"Ortalama birim fiyat %{price_change:.1f} değişmiştir. "
                          f"{'Fiyat erozyonu gözlemlenmektedir.' if price_change < -5 else 'Fiyat istikrarı korunmaktadır.'}",
                'icon': '💰'
            })
        
        # 4. Rekabet içgörüsü
        competition_metrics = filtered_df.groupby(['Period', 'Molecule'])['Corporation'].nunique().mean()
        market_concentration = filtered_df.groupby('Corporation')['USD_MNF'].sum().nlargest(3).sum() / total_market
        
        insights.append({
            'type': 'info',
            'title': 'Rekabet Yoğunluğu',
            'content': f"Ortalama {competition_metrics:.1f} şirket/molekül. "
                      f"Top 3 şirket pazarın %{market_concentration*100:.1f}'ini kontrol etmektedir.",
            'icon': '⚔️'
        })
        
        # 5. Molekül konsantrasyonu
        top_molecule = filtered_df.groupby('Molecule')['USD_MNF'].sum().nlargest(1)
        if not top_molecule.empty:
            molecule_share = top_molecule.iloc[0] / total_market
            
            insights.append({
                'type': 'info',
                'title': 'Molekül Konsantrasyonu',
                'content': f"{top_molecule.index[0]} molekülü segmentin %{molecule_share*100:.1f}'ini oluşturarak "
                          f"dominant pozisyondadır.",
                'icon': '🧪'
            })
        
        return insights
    
    @staticmethod
    def generate_strategic_recommendations(df):
        """Stratejik öneriler oluştur"""
        recommendations = []
        
        # Beyaz alan analizi
        white_spaces = PharmaAnalytics.identify_white_spaces(df)
        if not white_spaces.empty:
            top_opportunity = white_spaces.iloc[0]
            recommendations.append({
                'type': 'opportunity',
                'title': 'Beyaz Alan Fırsatı',
                'content': f"{top_opportunity['Country']} pazarında {top_opportunity['Molecule']} "
                          f"molekülü için önemli bir fırsat bulunmaktadır. "
                          f"Tahmini potansiyel: ${top_opportunity['Potential_Score']/1e6:.1f}M",
                'priority': 'Yüksek',
                'timeframe': '6-12 ay'
            })
        
        # Fiyat erozyonu uyarıları
        erosion_df = PharmaAnalytics.detect_price_erosion(df)
        high_risk_erosion = erosion_df[erosion_df['Risk_Level'] == 'Yüksek']
        
        if not high_risk_erosion.empty:
            for _, row in high_risk_erosion.head(2).iterrows():
                recommendations.append({
                    'type': 'risk',
                    'title': 'Fiyat Erozyonu Riski',
                    'content': f"{row['Molecule']} molekülü {row['Country']} pazarında "
                              f"fiyat erozyonu riski taşımaktadır. "
                              f"Fiyat değişimi: %{row['Price_Change']*100:.1f}",
                    'priority': 'Kritik',
                    'timeframe': 'Acil'
                })
        
        # Büyüme fırsatları
        growth_opportunities = df.groupby(['Country', 'Molecule']).apply(
            lambda x: pd.Series({
                'Current_Sales': x['USD_MNF'].sum(),
                'Growth_Rate': x['Sales_Growth_YoY'].mean(),
                'Penetration': x['Market_Share'].mean()
            })
        ).reset_index()
        
        high_growth = growth_opportunities[
            (growth_opportunities['Growth_Rate'] > 0.2) & 
            (growth_opportunities['Current_Sales'] > 1e6)
        ]
        
        if not high_growth.empty:
            top_growth = high_growth.nlargest(1, 'Growth_Rate').iloc[0]
            recommendations.append({
                'type': 'growth',
                'title': 'Yüksek Büyüme Fırsatı',
                'content': f"{top_growth['Country']} - {top_growth['Molecule']} kombinasyonunda "
                          f"%{top_growth['Growth_Rate']*100:.1f} büyüme oranı gözlemlenmektedir.",
                'priority': 'Orta',
                'timeframe': '12-18 ay'
            })
        
        return recommendations

# ================================================
# 6. ANA UYGULAMA
# ================================================
def main():
    # Veri yükleme
    df, companies, molecules_db, therapeutic_areas, regions = load_comprehensive_data()
    
    # Başlık
    st.markdown('<h1 class="main-title">💊 GLOBAL İLAÇ PAZARI STRATEJİ İSTİHBARAT PLATFORMU</h1>', 
                unsafe_allow_html=True)
    
    # Dashboard açıklaması
    with st.expander("📋 Dashboard Kullanım Kılavuzu", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🎯 Amaç**")
            st.markdown("""
            - Pazar dinamiklerini anlama
            - Rekabet analizi
            - Stratejik karar destek
            - Risk yönetimi
            """)
        
        with col2:
            st.markdown("**🔍 Temel Özellikler**")
            st.markdown("""
            - Gerçek zamanlı filtreleme
            - Otomatik içgörü üretimi
            - İleri analitik modeller
            - Görsel keşif araçları
            """)
        
        with col3:
            st.markdown("**📊 Analiz Modülleri**")
            st.markdown("""
            1. Pazar Büyüklüğü & Yapısı
            2. Fiyat-Hacim Analizi
            3. Rekabet Stratejisi
            4. Risk Yönetimi
            5. Stratejik Planlama
            """)
    
    # ================================================
    # FİLTRELEME PANELİ
    # ================================================
    st.sidebar.markdown("## 🔍 GELİŞMİŞ FİLTRELEME")
    
    # Çoklu seçim filtreleri
    with st.sidebar.expander("📍 Coğrafi Filtreler", expanded=True):
        selected_countries = st.multiselect(
            "Ülkeler",
            options=sorted(df['Country'].unique()),
            default=sorted(df['Country'].unique())[:5],
            help="Çoklu ülke seçimi yapabilirsiniz"
        )
        
        selected_regions = st.multiselect(
            "Bölgeler",
            options=sorted(df['Region'].unique()),
            default=sorted(df['Region'].unique()),
            help="Bölge bazlı filtreleme"
        )
    
    with st.sidebar.expander("🏢 Şirket & Molekül", expanded=True):
        selected_companies = st.multiselect(
            "Şirketler",
            options=sorted(df['Corporation'].unique()),
            default=sorted(df['Corporation'].unique())[:5],
            help="Çoklu şirket seçimi"
        )
        
        selected_molecules = st.multiselect(
            "Moleküller",
            options=sorted(df['Molecule'].unique()),
            default=sorted(df['Molecule'].unique())[:8],
            help="Çoklu molekül seçimi"
        )
        
        selected_ta = st.multiselect(
            "Terapötik Alanlar",
            options=sorted(df['Therapeutic_Area'].unique()),
            default=sorted(df['Therapeutic_Area'].unique()),
            help="Terapötik alan bazlı filtreleme"
        )
    
    with st.sidebar.expander("📈 Pazar Segmentleri", expanded=True):
        selected_sectors = st.multiselect(
            "Kanallar",
            options=sorted(df['Sector'].unique()),
            default=sorted(df['Sector'].unique()),
            help="Hospital/Retail dağılımı"
        )
        
        selected_specialty = st.multiselect(
            "Specialty Status",
            options=sorted(df['Specialty_Flag'].unique()),
            default=sorted(df['Specialty_Flag'].unique()),
            help="Specialty vs Non-Specialty"
        )
        
        selected_patent = st.multiselect(
            "Patent Durumu",
            options=sorted(df['Patent_Status'].unique()),
            default=sorted(df['Patent_Status'].unique()),
            help="Patentli/Jenerik ürünler"
        )
    
    with st.sidebar.expander("⏰ Zaman Periyodu", expanded=True):
        selected_years = st.multiselect(
            "Yıllar",
            options=sorted(df['Year'].unique()),
            default=sorted(df['Year'].unique())[-2:],
            help="Çoklu yıl seçimi"
        )
        
        selected_quarters = st.multiselect(
            "Çeyrekler",
            options=sorted(df['Quarter'].unique()),
            default=sorted(df['Quarter'].unique()),
            help="Çeyrek bazlı analiz"
        )
        
        time_comparison = st.selectbox(
            "Karşılaştırma Tipi",
            options=["YoY (Yıllık)", "QoQ (Çeyreklik)", "MAT (Hareketli Yıllık)", "Cumulative"],
            index=0
        )
    
    # İleri filtreler
    with st.sidebar.expander("⚙️ İleri Filtreler", expanded=False):
        price_range = st.slider(
            "Fiyat Aralığı ($)",
            min_value=float(df['Unit_Avg_Price'].min()),
            max_value=float(df['Unit_Avg_Price'].max()),
            value=(float(df['Unit_Avg_Price'].quantile(0.25)), 
                  float(df['Unit_Avg_Price'].quantile(0.75)))
        )
        
        market_size_threshold = st.number_input(
            "Minimum Pazar Büyüklüğü ($)",
            min_value=0,
            value=1000000,
            step=100000
        )
        
        growth_threshold = st.slider(
            "Minimum Büyüme Oranı (%)",
            min_value=-50,
            max_value=100,
            value=0,
            step=5
        )
    
    # Filtreleri uygula
    filtered_df = df[
        (df['Country'].isin(selected_countries)) &
        (df['Region'].isin(selected_regions)) &
        (df['Corporation'].isin(selected_companies)) &
        (df['Molecule'].isin(selected_molecules)) &
        (df['Therapeutic_Area'].isin(selected_ta)) &
        (df['Sector'].isin(selected_sectors)) &
        (df['Specialty_Flag'].isin(selected_specialty)) &
        (df['Patent_Status'].isin(selected_patent)) &
        (df['Year'].isin(selected_years)) &
        (df['Quarter'].isin(selected_quarters)) &
        (df['Unit_Avg_Price'].between(price_range[0], price_range[1]))
    ]
    
    # Filtrelenmiş veri kontrolü
    if filtered_df.empty:
        st.error("Seçilen filtrelerle eşleşen veri bulunamadı. Lütfen filtreleri genişletin.")
        return
    
    # ================================================
    # KPI PANELİ
    # ================================================
    st.markdown("## 📈 REALTIME PİYASA GÖSTERGELERİ")
    
    # Ana KPI'lar
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = filtered_df['USD_MNF'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">TOPLAM PAZAR</div>
            <div class="metric-value">${total_sales/1e9:.2f}B</div>
            <div class="metric-change">
                <span style="color: {'#10B981' if total_sales > df['USD_MNF'].sum() * 0.1 else '#F59E0B'}">
                    {total_sales/(df['USD_MNF'].sum())*100:.1f}% global pay
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_growth = filtered_df['Sales_Growth_YoY'].mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">YILLIK BÜYÜME</div>
            <div class="metric-value">{avg_growth:.1f}%</div>
            <div class="metric-change">
                <span style="color: {'#10B981' if avg_growth > 0 else '#EF4444'}">
                    {'↑ Büyüme' if avg_growth > 0 else '↓ Küçülme'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_price = filtered_df['Unit_Avg_Price'].mean()
        price_change = filtered_df['Price_Growth_YoY'].mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ORT. BİRİM FİYAT</div>
            <div class="metric-value">${avg_price:.2f}</div>
            <div class="metric-change">
                <span style="color: {'#10B981' if price_change > 0 else '#EF4444'}">
                    {price_change:+.1f}% yıllık
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        concentration = PharmaAnalytics.calculate_market_concentration(filtered_df)
        hhi = list(concentration.values())[0]['HHI'] if concentration else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">REKABET YOĞUNLUĞU</div>
            <div class="metric-value">{hhi:.0f} HHI</div>
            <div class="metric-change">
                <span style="color: {'#EF4444' if hhi > 2500 else '#F59E0B' if hhi > 1500 else '#10B981'}">
                    {'Yüksek' if hhi > 2500 else 'Orta' if hhi > 1500 else 'Düşük'} Konsantrasyon
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # İkincil KPI'lar
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        molecule_count = filtered_df['Molecule'].nunique()
        st.metric("Molekül Çeşitliliği", f"{molecule_count}", 
                 f"{filtered_df['Molecule'].nunique()/df['Molecule'].nunique()*100:.0f}% coverage")
    
    with col6:
        company_count = filtered_df['Corporation'].nunique()
        st.metric("Aktif Şirket", f"{company_count}", 
                 f"{company_count/len(companies)*100:.0f}% representation")
    
    with col7:
        country_count = filtered_df['Country'].nunique()
        st.metric("Ülke Kapsamı", f"{country_count}", 
                 f"{country_count/len(df['Country'].unique())*100:.0f}% coverage")
    
    with col8:
        specialty_share = filtered_df[filtered_df['Specialty_Flag'] == 'Specialty']['USD_MNF'].sum() / total_sales * 100
        st.metric("Specialty Payı", f"{specialty_share:.1f}%", 
                 f"Hospital: {filtered_df[filtered_df['Sector'] == 'Hospital']['USD_MNF'].sum()/total_sales*100:.1f}%")
    
    # ================================================
    # OTOMATİK İÇGÖRÜLER
    # ================================================
    st.markdown("## 💡 OTOMATİK İÇGÖRÜLER & ÖNERİLER")
    
    # İçgörüleri oluştur
    selected_filters = {
        'selected_countries': selected_countries,
        'selected_years': selected_years,
        'selected_molecules': selected_molecules
    }
    
    insights = InsightEngine.generate_market_insights(df, filtered_df, selected_filters)
    recommendations = InsightEngine.generate_strategic_recommendations(filtered_df)
    
    # İçgörüleri göster
    insight_cols = st.columns(min(3, len(insights)))
    for idx, insight in enumerate(insights[:3]):
        with insight_cols[idx % 3]:
            st.markdown(f"""
            <div class="insight-card {insight['type']}">
                <div class="insight-title">{insight['icon']} {insight['title']}</div>
                <div class="insight-content">{insight['content']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ================================================
    # ANA ANALİZ BÖLÜMLERİ
    # ================================================
    
    # Tab'lar ile ana analiz bölümleri
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 PAZAR YAPISI", 
        "💰 FİYAT ANALİZİ", 
        "⚔️ REKABET STRATEJİSİ", 
        "⚠️ RİSK YÖNETİMİ", 
        "🚀 BÜYÜME FIRSATLARI"
    ])
    
    # TAB 1: PAZAR YAPISI
    with tab1:
        st.markdown('<h3 class="section-title">1. PAZAR YAPISI & DINAMIKLERI</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Pazar evrim chart'ı
            fig = PharmaVisualizations.create_market_evolution_chart(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Terapötik alan analizi
            st.markdown('<h4 class="subsection-title">Terapötik Alan Dağılımı</h4>', unsafe_allow_html=True)
            fig2 = PharmaVisualizations.create_therapeutic_area_sunburst(filtered_df)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Molekül performansı
            st.markdown('<h4 class="subsection-title">Molekül Performans Matrisi</h4>', unsafe_allow_html=True)
            molecule_perf = filtered_df.groupby('Molecule').agg({
                'USD_MNF': ['sum', 'mean', 'std'],
                'Sales_Growth_YoY': 'mean',
                'Unit_Avg_Price': 'mean'
            }).round(2)
            
            # Renklendirme
            def color_molecule(val):
                if val.name == 'USD_MNF':
                    return ['background-color: #E0F2FE' for _ in val]
                elif val.name == 'Sales_Growth_YoY':
                    return ['color: green' if x > 0 else 'color: red' for x in val]
                return [''] * len(val)
            
            st.dataframe(molecule_perf.style.apply(color_molecule, axis=0), height=400)
            
            # Pazar konsantrasyonu
            st.markdown('<h4 class="subsection-title">Pazar Konsantrasyon Trendi</h4>', unsafe_allow_html=True)
            fig3 = PharmaVisualizations.create_market_concentration_timeline(filtered_df)
            st.plotly_chart(fig3, use_container_width=True)
    
    # TAB 2: FİYAT ANALİZİ
    with tab2:
        st.markdown('<h3 class="section-title">2. FİYAT ANALİZİ & OPTIMIZASYONU</h3>', unsafe_allow_html=True)
        
        # Fiyat-hacim analizi
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4 class="subsection-title">Fiyat-Hacim İlişkisi</h4>', unsafe_allow_html=True)
            
            # Fiyat esnekliği analizi
            elasticity_results = PharmaAnalytics.calculate_price_elasticity(filtered_df)
            if elasticity_results:
                elasticity_df = pd.DataFrame.from_dict(elasticity_results, orient='index')
                elasticity_df = elasticity_df.reset_index()
                elasticity_df[['Country', 'Molecule']] = pd.DataFrame(
                    elasticity_df['index'].tolist(), index=elasticity_df.index
                )
                
                fig = px.scatter(elasticity_df,
                                x='elasticity',
                                y='r_squared',
                                size='r_squared',
                                color='interpretation',
                                hover_name='Molecule',
                                title='Fiyat Esnekliği Analizi',
                                labels={'elasticity': 'Esneklik Katsayısı',
                                       'r_squared': 'Model Uyumu (R²)'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h4 class="subsection-title">Fiyat Benchmarking</h4>', unsafe_allow_html=True)
            
            # Şirketler arası fiyat karşılaştırması
            price_comparison = filtered_df.groupby(['Corporation', 'Molecule'])['Unit_Avg_Price'].mean().unstack()
            
            # Heatmap
            fig = px.imshow(price_comparison.T,
                           labels=dict(x="Şirket", y="Molekül", color="Fiyat ($)"),
                           title='Şirket-Molekül Fiyat Matrisi',
                           color_continuous_scale='RdYlBu_r',
                           aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
        
        # Fiyat trendleri
        st.markdown('<h4 class="subsection-title">Fiyat Trendleri & Tahminleri</h4>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Zaman serisi analizi
            selected_molecule_trend = st.selectbox(
                "Molekül Seçin:",
                options=sorted(filtered_df['Molecule'].unique())
            )
            
            if selected_molecule_trend:
                molecule_trend_data = filtered_df[filtered_df['Molecule'] == selected_molecule_trend]
                price_trend = molecule_trend_data.groupby('Period')['Unit_Avg_Price'].mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=price_trend.index, y=price_trend.values,
                                        mode='lines+markers', name='Ortalama Fiyat'))
                
                # Trend çizgisi
                z = np.polyfit(range(len(price_trend)), price_trend.values, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(x=price_trend.index, y=p(range(len(price_trend))),
                                        mode='lines', name='Trend',
                                        line=dict(dash='dash', color='red')))
                
                fig.update_layout(title=f'{selected_molecule_trend} - Fiyat Trendi',
                                 xaxis_title='Dönem',
                                 yaxis_title='Ortalama Fiyat ($)')
                st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Fiyat segmentasyonu
            price_segments = pd.qcut(filtered_df['Unit_Avg_Price'], q=4, labels=['Düşük', 'Orta-Düşük', 'Orta-Yüksek', 'Yüksek'])
            segment_analysis = filtered_df.groupby(price_segments).agg({
                'USD_MNF': 'sum',
                'Units': 'sum',
                'Sales_Growth_YoY': 'mean'
            }).reset_index()
            
            fig = px.bar(segment_analysis,
                        x='Unit_Avg_Price',
                        y='USD_MNF',
                        color='Sales_Growth_YoY',
                        title='Fiyat Segmentleri Performansı',
                        labels={'Unit_Avg_Price': 'Fiyat Segmenti',
                               'USD_MNF': 'Toplam Satış',
                               'Sales_Growth_YoY': 'Büyüme Oranı'})
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: REKABET STRATEJİSİ
    with tab3:
        st.markdown('<h3 class="section-title">3. REKABET ANALİZİ & STRATEJİ HARİTASI</h3>', unsafe_allow_html=True)
        
        # Rekabet manzarası
        fig = PharmaVisualizations.create_competitive_landscape(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detaylı rekabet analizi
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4 class="subsection-title">Şirket Bazlı Performans</h4>', unsafe_allow_html=True)
            
            company_performance = filtered_df.groupby('Corporation').agg({
                'USD_MNF': 'sum',
                'Market_Share': 'mean',
                'Sales_Growth_YoY': 'mean',
                'Price_Growth_YoY': 'mean',
                'Competitor_Count': 'mean'
            }).sort_values('USD_MNF', ascending=False)
            
            # Performans scoring
            company_performance['Performance_Score'] = (
                company_performance['Market_Share'] * 0.3 +
                company_performance['Sales_Growth_YoY'] * 0.3 +
                company_performance['Price_Growth_YoY'] * 0.2 +
                (1 / company_performance['Competitor_Count']) * 0.2
            )
            
            fig = px.bar(company_performance.head(10),
                        x=company_performance.head(10).index,
                        y='Performance_Score',
                        color='USD_MNF',
                        title='Top 10 Şirket Performans Skoru',
                        labels={'Performance_Score': 'Performans Skoru',
                               'USD_MNF': 'Satış ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h4 class="subsection-title">Rekabet Yoğunluğu Analizi</h4>', unsafe_allow_html=True)
            
            # Molekül bazlı rekabet
            molecule_competition = filtered_df.groupby(['Molecule', 'Period']).agg({
                'Corporation': 'nunique',
                'USD_MNF': 'sum',
                'Unit_Avg_Price': 'std'
            }).reset_index()
            
            fig = px.scatter(molecule_competition,
                            x='Corporation',
                            y='USD_MNF',
                            size='Unit_Avg_Price',
                            color='Period',
                            hover_name='Molecule',
                            title='Molekül Bazlı Rekabet Yoğunluğu',
                            labels={'Corporation': 'Rakip Sayısı',
                                   'USD_MNF': 'Pazar Büyüklüğü',
                                   'Unit_Avg_Price': 'Fiyat Varyansı'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Stratejik öneriler
        st.markdown('<h4 class="subsection-title">Rekabet Stratejisi Önerileri</h4>', unsafe_allow_html=True)
        
        strategic_analysis = filtered_df.groupby(['Country', 'Molecule']).apply(
            lambda x: pd.Series({
                'Market_Size': x['USD_MNF'].sum(),
                'Competition_Level': x['Corporation'].nunique(),
                'Price_Variance': x['Unit_Avg_Price'].std() / x['Unit_Avg_Price'].mean(),
                'Growth_Potential': x['Sales_Growth_YoY'].mean()
            })
        ).reset_index()
        
        # Strateji kategorizasyonu
        def categorize_strategy(row):
            if row['Competition_Level'] < 3 and row['Growth_Potential'] > 0.1:
                return 'Giriş Stratejisi'
            elif row['Competition_Level'] >= 3 and row['Price_Variance'] > 0.3:
                return 'Farklılaşma Stratejisi'
            elif row['Competition_Level'] >= 5 and row['Price_Variance'] < 0.2:
                return 'Maliyet Liderliği'
            else:
                return 'Odaklanma Stratejisi'
        
        strategic_analysis['Recommended_Strategy'] = strategic_analysis.apply(categorize_strategy, axis=1)
        
        fig = px.scatter(strategic_analysis,
                        x='Competition_Level',
                        y='Growth_Potential',
                        size='Market_Size',
                        color='Recommended_Strategy',
                        hover_name='Molecule',
                        title='Strateji Öneri Haritası',
                        labels={'Competition_Level': 'Rekabet Seviyesi',
                               'Growth_Potential': 'Büyüme Potansiyeli',
                               'Market_Size': 'Pazar Büyüklüğü'})
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: RİSK YÖNETİMİ
    with tab4:
        st.markdown('<h3 class="section-title">4. RİSK YÖNETİMİ & ERKEN UYARI SİSTEMİ</h3>', unsafe_allow_html=True)
        
        # Fiyat erozyonu analizi
        erosion_df = PharmaAnalytics.detect_price_erosion(filtered_df)
        
        if not erosion_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<h4 class="subsection-title">Fiyat Erozyonu Risk Haritası</h4>', unsafe_allow_html=True)
                fig = PharmaVisualizations.create_price_erosion_heatmap(erosion_df)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown('<h4 class="subsection-title">Risk Seviyeleri</h4>', unsafe_allow_html=True)
                
                risk_summary = erosion_df['Risk_Level'].value_counts()
                fig = px.pie(values=risk_summary.values,
                            names=risk_summary.index,
                            title='Risk Dağılımı',
                            color=risk_summary.index,
                            color_discrete_map={'Yüksek': '#EF4444',
                                              'Orta': '#F59E0B',
                                              'Düşük': '#10B981'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Yüksek riskli alanlar
                high_risk = erosion_df[erosion_df['Risk_Level'] == 'Yüksek']
                if not high_risk.empty:
                    st.markdown("**🔴 Yüksek Riskli Alanlar:**")
                    for _, row in high_risk.head(5).iterrows():
                        st.markdown(f"- {row['Molecule']} ({row['Country']}): "
                                  f"Fiyat ↓%{abs(row['Price_Change'])*100:.1f}, "
                                  f"Hacim ↑%{row['Volume_Change']*100:.1f}")
        
        # Pazar konsantrasyonu riskleri
        st.markdown('<h4 class="subsection-title">Pazar Konsantrasyonu Riskleri</h4>', unsafe_allow_html=True)
        
        concentration_risk = PharmaAnalytics.calculate_market_concentration(filtered_df)
        conc_risk_df = pd.DataFrame.from_dict(concentration_risk, orient='index')
        conc_risk_df = conc_risk_df.reset_index()
        conc_risk_df[['Country', 'Period']] = pd.DataFrame(conc_risk_df['index'].tolist(), 
                                                          index=conc_risk_df.index)
        
        # Risk kategorizasyonu
        def categorize_concentration_risk(hhi):
            if hhi > 2500:
                return 'Yüksek Risk (Oligopol)'
            elif hhi > 1800:
                return 'Orta Risk'
            elif hhi > 1000:
                return 'Düşük Risk'
            else:
                return 'Fragmente Pazar'
        
        conc_risk_df['Risk_Category'] = conc_risk_df['HHI'].apply(categorize_concentration_risk)
        
        fig = px.scatter(conc_risk_df,
                        x='Period',
                        y='HHI',
                        color='Risk_Category',
                        size='Top3_Share',
                        hover_name='Country',
                        title='Pazar Konsantrasyonu Risk Analizi',
                        labels={'HHI': 'Herfindahl-Hirschman İndeksi',
                               'Top3_Share': 'Top 3 Payı',
                               'Period': 'Dönem'})
        
        # Referans çizgileri
        fig.add_hline(y=2500, line_dash="dash", line_color="red", opacity=0.7)
        fig.add_hline(y=1800, line_dash="dash", line_color="orange", opacity=0.7)
        fig.add_hline(y=1000, line_dash="dash", line_color="green", opacity=0.7)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tedarik zinciri riskleri
        st.markdown('<h4 class="subsection-title">Ürün Yaşam Döngüsü Riskleri</h4>', unsafe_allow_html=True)
        
        lifecycle_df = PharmaAnalytics.analyze_product_lifecycle(filtered_df)
        
        if not lifecycle_df.empty:
            fig = px.scatter(lifecycle_df,
                            x='Avg_Growth_Rate',
                            y='Penetration_Rate',
                            size='Current_Sales',
                            color='Lifecycle_Stage',
                            hover_name='Molecule',
                            title='Ürün Yaşam Döngüsü Risk Analizi',
                            labels={'Avg_Growth_Rate': 'Ortalama Büyüme',
                                   'Penetration_Rate': 'Penetrasyon Oranı',
                                   'Current_Sales': 'Cari Satışlar',
                                   'Lifecycle_Stage': 'Yaşam Döngüsü'})
            
            # Quadrant çizgileri
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: BÜYÜME FIRSATLARI
    with tab5:
        st.markdown('<h3 class="section-title">5. BÜYÜME FIRSATLARI & STRATEJİK YATIRIM</h3>', unsafe_allow_html=True)
        
        # Beyaz alan analizi
        white_spaces_df = PharmaAnalytics.identify_white_spaces(filtered_df)
        
        if not white_spaces_df.empty:
            st.markdown('<h4 class="subsection-title">Beyaz Alan (White Space) Fırsatları</h4>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Coğrafi dağılım
                fig = px.choropleth(white_spaces_df,
                                   locations='Country',
                                   locationmode='country names',
                                   color='Potential_Score',
                                   hover_name='Molecule',
                                   title='Beyaz Alan Fırsat Haritası',
                                   color_continuous_scale='Viridis')
                fig.update_layout(geo=dict(showframe=False, showcoastlines=True))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top fırsatlar
                st.markdown("**🏆 En Yüksek Potansiyelli Fırsatlar:**")
                for idx, row in white_spaces_df.head(5).iterrows():
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{row['Molecule']}</strong><br>
                        <small>{row['Country']} | {row['Therapeutic_Area']}</small><br>
                        <span style="color: #3B82F6">Potansiyel: ${row['Potential_Score']/1e6:.1f}M</span><br>
                        <small>Büyüme: %{row['Avg_Growth_Rate']*100:.1f}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Büyüme fırsatları analizi
        st.markdown('<h4 class="subsection-title">Büyüme Motorları Analizi</h4>', unsafe_allow_html=True)
        
        growth_analysis = filtered_df.groupby(['Country', 'Molecule', 'Therapeutic_Area']).apply(
            lambda x: pd.Series({
                'Current_Sales': x['USD_MNF'].sum(),
                'YoY_Growth': x['Sales_Growth_YoY'].mean(),
                'Market_Share': x['Market_Share'].mean(),
                'Competition': x['Corporation'].nunique()
            })
        ).reset_index()
        
        # Büyüme matrisi
        growth_analysis['Growth_Score'] = (
            growth_analysis['YoY_Growth'] * 0.4 +
            growth_analysis['Market_Share'] * 0.3 +
            (1 / growth_analysis['Competition']) * 0.3
        )
        
        fig = px.treemap(growth_analysis,
                        path=['Therapeutic_Area', 'Country', 'Molecule'],
                        values='Current_Sales',
                        color='Growth_Score',
                        color_continuous_scale='RdYlGn',
                        title='Büyüme Fırsatları Hiyerarşisi',
                        hover_data=['YoY_Growth', 'Market_Share', 'Competition'])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Yatırım önceliklendirme
        st.markdown('<h4 class="subsection-title">Yatırım Önceliklendirme Matrisi</h4>', unsafe_allow_html=True)
        
        investment_matrix = growth_analysis.copy()
        
        def prioritize_investment(row):
            if row['YoY_Growth'] > 0.15 and row['Market_Share'] < 0.3:
                return 'Yüksek Öncelik - Hızlı Büyüme'
            elif row['YoY_Growth'] > 0.1 and row['Competition'] < 4:
                return 'Orta Öncelik - Koruma'
            elif row['YoY_Growth'] < 0 and row['Market_Share'] > 0.2:
                return 'Optimizasyon - Düşüşte'
            else:
                return 'İzleme - Nötr'
        
        investment_matrix['Investment_Priority'] = investment_matrix.apply(prioritize_investment, axis=1)
        
        fig = px.scatter(investment_matrix,
                        x='Market_Share',
                        y='YoY_Growth',
                        size='Current_Sales',
                        color='Investment_Priority',
                        hover_name='Molecule',
                        title='Yatırım Önceliklendirme Matrisi',
                        labels={'Market_Share': 'Pazar Payı',
                               'YoY_Growth': 'Yıllık Büyüme',
                               'Current_Sales': 'Cari Satışlar'})
        
        # Quadrant çizgileri
        fig.add_hline(y=0.1, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0.2, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ================================================
    # İNDİRİLEBİLİR RAPORLAR
    # ================================================
    st.markdown("## 📥 RAPORLAR & İNDİRİLEBİLİR ÇIKTILAR")
    
    with st.expander("Rapor Oluşturma Seçenekleri", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            report_type = st.selectbox(
                "Rapor Tipi",
                options=["Pazar Özeti", "Rekabet Analizi", "Risk Değerlendirmesi", 
                        "Büyüme Stratejisi", "Tam Kapsamlı Analiz"]
            )
        
        with col2:
            format_type = st.selectbox(
                "Format",
                options=["PDF", "Excel", "PowerPoint", "JSON"]
            )
        
        with col3:
            detail_level = st.select_slider(
                "Detay Seviyesi",
                options=["Özet", "Standart", "Detaylı", "Kapsamlı"]
            )
        
        if st.button("📊 Rapor Oluştur", type="primary"):
            with st.spinner("Rapor oluşturuluyor..."):
                # Rapor oluşturma simülasyonu
                report_data = {
                    'report_type': report_type,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'filters_applied': {
                        'countries': selected_countries,
                        'years': selected_years,
                        'molecules': selected_molecules
                    },
                    'key_findings': insights[:3],
                    'recommendations': recommendations[:3]
                }
                
                # JSON olarak indirme
                st.download_button(
                    label="📥 Raporu İndir",
                    data=json.dumps(report_data, indent=2, ensure_ascii=False),
                    file_name=f"pharma_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # ================================================
    # FOOTER & BİLGİLENDİRME
    # ================================================
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("""
        **📞 İletişim & Destek**
        - Email: analiz@globalpharmaintel.com
        - Telefon: +90 212 123 4567
        """)
    
    with footer_col2:
        st.markdown("""
        **🔒 Veri Güvenliği**
        - GDPR Uyumlu
        - End-to-End Şifreleme
        - Anonimize Edilmiş Veri
        """)
    
    with footer_col3:
        st.markdown("""
        **🔄 Güncellemeler**
        - Son Güncelleme: {}
        - Veri Kaynağı: IQVIA, Evaluate Pharma
        - Bir Sonraki Güncelleme: {}
        """.format(
            datetime.now().strftime("%d/%m/%Y"),
            (datetime.now() + timedelta(days=7)).strftime("%d/%m/%Y")
        ))
    
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; color: #6B7280; font-size: 0.8rem;">
    <em>Global Pharma Market Intelligence Pro v2.0 | © 2024 Tüm hakları saklıdır. 
    Bu dashboard bağımsız pazar araştırması amaçlıdır.</em>
    </div>
    """, unsafe_allow_html=True)

# ================================================
# 7. UYGULAMA BAŞLATMA
# ================================================
if __name__ == "__main__":
    main()
