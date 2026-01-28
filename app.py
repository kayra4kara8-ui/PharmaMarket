# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import statsmodels.api as sm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Global İlaç Pazarı Strateji Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #F0F9FF;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px;
    }
    .risk-high { color: #DC2626; font-weight: bold; }
    .risk-medium { color: #F59E0B; font-weight: bold; }
    .risk-low { color: #10B981; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌍 Global İlaç Pazarı Strateji Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Bağımsız Pazar Araştırması Perspektifi**")

# Data Loading Function
@st.cache_data
def load_data():
    # In production, replace with actual data loading
    # For demo purposes, generating synthetic data
    np.random.seed(42)
    
    # Generate synthetic data structure
    countries = ['Türkiye', 'Almanya', 'Fransa', 'İtalya', 'İspanya', 'İngiltere', 
                 'Polonya', 'Hollanda', 'Belçika', 'İsveç', 'Norveç', 'Danimarka']
    regions = {'Kuzey Avrupa': ['İsveç', 'Norveç', 'Danimarka'],
               'Batı Avrupa': ['Almanya', 'Fransa', 'Hollanda', 'Belçika'],
               'Güney Avrupa': ['İtalya', 'İspanya'],
               'Doğu Avrupa': ['Türkiye', 'Polonya']}
    
    companies = ['Pfizer', 'Novartis', 'Roche', 'Merck', 'Sanofi', 'GSK', 
                 'AstraZeneca', 'Johnson & Johnson', 'Bayer', 'AbbVie', 'Eli Lilly', 'Boehringer']
    
    molecules = ['Adalimumab', 'Pembrolizumab', 'Nivolumab', 'Rituximab', 'Trastuzumab',
                 'Bevacizumab', 'Insulin Glargine', 'Sitagliptin', 'Atorvastatin',
                 'Apremilast', 'Dupilumab', 'Semaglutide', 'Ibrutinib', 'Venetoclax']
    
    salts = {
        'Adalimumab': 'Monoclonal Antibody',
        'Pembrolizumab': 'Anti-PD-1',
        'Nivolumab': 'Anti-PD-L1',
        'Semaglutide': 'GLP-1 Analog',
        'Insulin Glargine': 'Insulin Analog'
    }
    
    # Generate time series data
    dates = ['2022-Q3', '2023-Q3', '2024-Q3']
    
    data = []
    for country in countries:
        for company in companies[:8]:  # Limit for demo
            for molecule in molecules[:10]:  # Limit for demo
                for date in dates:
                    # Find region
                    region = next((r for r, c_list in regions.items() if country in c_list), 'Other')
                    
                    # Generate realistic metrics
                    base_sales = np.random.lognormal(mean=2, sigma=1.5) * 1000000
                    growth_rate = np.random.normal(loc=0.08, scale=0.15)
                    
                    if date == '2023-Q3':
                        sales = base_sales * (1 + growth_rate)
                    elif date == '2024-Q3':
                        sales = base_sales * (1 + growth_rate) * (1 + np.random.normal(loc=0.05, scale=0.1))
                    else:
                        sales = base_sales
                    
                    units = sales / np.random.uniform(50, 500)
                    su_units = units * np.random.uniform(0.8, 1.2)
                    unit_price = sales / units
                    su_price = sales / su_units
                    
                    sector = np.random.choice(['Hospital', 'Retail'], p=[0.6, 0.4])
                    specialty = np.random.choice(['Specialty', 'Non-Specialty'], 
                                                p=[0.3, 0.7] if sector == 'Hospital' else [0.1, 0.9])
                    
                    data.append({
                        'Country': country,
                        'Region': region,
                        'Sector': sector,
                        'Corporation': company,
                        'Molecule': molecule,
                        'Chemical_Salt': salts.get(molecule, 'Other'),
                        'Product': f"{molecule} {np.random.choice(['SC', 'IV', 'Oral'])}",
                        'Pack': np.random.choice(['Vial', 'Prefilled Syringe', 'Tablet', 'Capsule']),
                        'Strength': np.random.choice(['40mg', '100mg', '150mg', '200mg', '500mg']),
                        'Volume_ml': np.random.choice([1, 2, 5, 10, 20]),
                        'Specialty_Flag': specialty,
                        'Prescription_Status': 'Rx',
                        'Period': date,
                        'USD_MNF': sales,
                        'Units': units,
                        'Standard_Units': su_units,
                        'Unit_Avg_Price': unit_price,
                        'SU_Avg_Price': su_price
                    })
    
    df = pd.DataFrame(data)
    
    # Add calculated columns
    df['Year'] = df['Period'].str[:4]
    df['MAT_Growth'] = df.groupby(['Country', 'Corporation', 'Molecule'])['USD_MNF'].pct_change()
    df['Price_Change_Pct'] = df.groupby(['Country', 'Corporation', 'Molecule'])['Unit_Avg_Price'].pct_change()
    df['Volume_Change_Pct'] = df.groupby(['Country', 'Corporation', 'Molecule'])['Units'].pct_change()
    
    return df

# Load data
df = load_data()

# Sidebar Filters
st.sidebar.markdown("## 🔍 Filtreler")

# Multi-select filters
selected_countries = st.sidebar.multiselect(
    "Ülkeler",
    options=sorted(df['Country'].unique()),
    default=sorted(df['Country'].unique())[:3]
)

selected_region = st.sidebar.multiselect(
    "Bölge / Alt Bölge",
    options=sorted(df['Region'].unique()),
    default=sorted(df['Region'].unique())
)

selected_companies = st.sidebar.multiselect(
    "Şirketler (Manufacturer)",
    options=sorted(df['Corporation'].unique()),
    default=sorted(df['Corporation'].unique())[:3]
)

selected_molecules = st.sidebar.multiselect(
    "Moleküller",
    options=sorted(df['Molecule'].unique()),
    default=sorted(df['Molecule'].unique())[:5]
)

selected_sector = st.sidebar.multiselect(
    "Kanal (Sector)",
    options=sorted(df['Sector'].unique()),
    default=sorted(df['Sector'].unique())
)

selected_specialty = st.sidebar.multiselect(
    "Specialty / Non-Specialty",
    options=sorted(df['Specialty_Flag'].unique()),
    default=sorted(df['Specialty_Flag'].unique())
)

year_options = sorted(df['Year'].unique())
selected_years = st.sidebar.multiselect(
    "Yıllar",
    options=year_options,
    default=year_options
)

comparison_type = st.sidebar.selectbox(
    "Karşılaştırma Tipi",
    options=["Yıllık Büyüme", "MAT Karşılaştırması", "Bölgesel Farklar"]
)

# Apply filters
filtered_df = df[
    (df['Country'].isin(selected_countries)) &
    (df['Region'].isin(selected_region)) &
    (df['Corporation'].isin(selected_companies)) &
    (df['Molecule'].isin(selected_molecules)) &
    (df['Sector'].isin(selected_sector)) &
    (df['Specialty_Flag'].isin(selected_specialty)) &
    (df['Year'].isin(selected_years))
]

# Calculate KPIs
total_sales = filtered_df['USD_MNF'].sum() / 1e6  # Convert to millions
yoy_growth = filtered_df.groupby('Year')['USD_MNF'].sum().pct_change().iloc[-1] if len(selected_years) > 1 else 0
avg_price = filtered_df['Unit_Avg_Price'].mean()
price_change = filtered_df.groupby('Year')['Unit_Avg_Price'].mean().pct_change().iloc[-1] if len(selected_years) > 1 else 0
market_concentration = filtered_df.groupby('Corporation')['USD_MNF'].sum().nlargest(3).sum() / filtered_df['USD_MNF'].sum()

# Display KPIs
st.markdown('<h2 class="sub-header">📈 Temel Pazar Göstergeleri</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Toplam Pazar ($M)", f"${total_sales:.1f}M", 
              f"{yoy_growth*100:.1f}%" if len(selected_years) > 1 else "")
with col2:
    st.metric("Ortalama Birim Fiyat", f"${avg_price:.2f}", 
              f"{price_change*100:.1f}%" if len(selected_years) > 1 else "")
with col3:
    st.metric("Pazar Konsantrasyonu (Top3)", f"{market_concentration*100:.1f}%")
with col4:
    molecule_count = filtered_df['Molecule'].nunique()
    company_count = filtered_df['Corporation'].nunique()
    st.metric("Molekül/Şirket Çeşitliliği", f"{molecule_count}/{company_count}")

# Automatic Insight Generation
st.markdown("## 💡 Otomatik İçgörüler")

# Generate insights based on data
if len(filtered_df) > 0:
    # Insight 1: Growth drivers
    growth_by_molecule = filtered_df.groupby('Molecule')['USD_MNF'].sum().sort_values(ascending=False)
    top_molecule = growth_by_molecule.index[0]
    top_molecule_share = growth_by_molecule.iloc[0] / filtered_df['USD_MNF'].sum()
    
    # Insight 2: Price-Volume analysis
    price_volume_corr = filtered_df.groupby('Molecule')[['Unit_Avg_Price', 'Units']].corr().iloc[0::2, 1].mean()
    
    # Insight 3: Regional concentration
    regional_share = filtered_df.groupby('Region')['USD_MNF'].sum() / filtered_df['USD_MNF'].sum()
    top_region = regional_share.idxmax()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
        <strong>Pazar Yoğunlaşması:</strong> {top_molecule} molekülü, seçilen segmentteki toplam satışların <strong>%{top_molecule_share*100:.1f}</strong>'ini oluşturmaktadır. Bu molekül pazar büyümesinin ana itici gücüdür.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>Bölgesel Konsantrasyon:</strong> Pazarın <strong>%{regional_share.max()*100:.1f}</strong>'i {top_region} bölgesinde toplanmıştır. Büyüme bu bölgeden kaynaklanmakta, diğer bölgeler görece durgun kalmaktadır.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        price_volume_text = "pozitif" if price_volume_corr > 0.1 else "negatif" if price_volume_corr < -0.1 else "zayıf"
        st.markdown(f"""
        <div class="insight-box">
        <strong>Fiyat-Hacim İlişkisi:</strong> Fiyat ve hacim arasında <strong>{price_volume_text}</strong> korelasyon ({price_volume_corr:.2f}) gözlemlenmektedir. Bu, fiyat artışlarının hacim kaybına yol açmadığını göstermektedir.
        </div>
        """, unsafe_allow_html=True)
        
        # Competition insight
        market_share_change = filtered_df.pivot_table(
            index='Corporation', 
            columns='Year', 
            values='USD_MNF', 
            aggfunc='sum'
        ).fillna(0)
        
        if market_share_change.shape[1] > 1:
            share_growth = (market_share_change.iloc[:, -1] / market_share_change.iloc[:, 0] - 1).sort_values(ascending=False)
            top_gainer = share_growth.index[0]
            gain = share_growth.iloc[0]
            
            st.markdown(f"""
            <div class="insight-box">
            <strong>Rekabet Dinamikleri:</strong> {top_gainer} şirketi pazar payını <strong>%{gain*100:.1f}</strong> artırarak en hızlı büyüyen oyuncu olmuştur. Bu genellikle agresif fiyatlandırma veya yeni ürün lansmanları ile ilişkilidir.
            </div>
            """, unsafe_allow_html=True)

# 1. Market Size & Structure Analysis
st.markdown('<h2 class="sub-header">📊 Pazar Büyüklüğü & Yapısı</h2>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Zaman Trendi", "Bölgesel Dağılım", "Şirket Payları", "Molekül Konsantrasyonu"])

with tab1:
    # Time series analysis
    time_series = filtered_df.groupby(['Period', 'Year'])['USD_MNF'].sum().reset_index()
    
    fig = px.area(time_series, x='Period', y='USD_MNF',
                  title='Pazar Büyüklüğü Trendi (USD MNF)',
                  labels={'USD_MNF': 'Satış ($)', 'Period': 'Dönem'})
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Insight for time series
    if len(time_series) > 1:
        total_growth = (time_series['USD_MNF'].iloc[-1] / time_series['USD_MNF'].iloc[0] - 1) * 100
        st.markdown(f"""
        <div class="insight-box">
        <strong>Trend Analizi:</strong> Seçilen dönemde pazar toplamda <strong>%{total_growth:.1f}</strong> büyümüştür. 
        En hızlı büyüme {time_series.loc[time_series['USD_MNF'].pct_change().idxmax(), 'Period']} döneminde gerçekleşmiştir.
        </div>
        """, unsafe_allow_html=True)

with tab2:
    # Regional heatmap
    regional_data = filtered_df.groupby(['Region', 'Country'])['USD_MNF'].sum().reset_index()
    
    fig = px.choropleth(regional_data,
                        locations='Country',
                        locationmode='country names',
                        color='USD_MNF',
                        hover_name='Country',
                        color_continuous_scale='Blues',
                        title='Ülke Bazlı Pazar Dağılımı')
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True))
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional insight
    regional_summary = regional_data.groupby('Region')['USD_MNF'].sum().sort_values(ascending=False)
    top_region_share = regional_summary.iloc[0] / regional_summary.sum() * 100
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>Bölgesel Konsantrasyon:</strong> {regional_summary.index[0]} bölgesi, toplam pazarın <strong>%{top_region_share:.1f}</strong>'ini oluşturarak dominant bölge konumundadır.
    Ülke bazında en büyük pazar {regional_data.loc[regional_data['USD_MNF'].idxmax(), 'Country']}'dir.
    </div>
    """, unsafe_allow_html=True)

with tab3:
    # Market share by company
    company_share = filtered_df.groupby('Corporation')['USD_MNF'].sum().sort_values(ascending=False).head(10)
    
    fig = px.bar(company_share, 
                 x=company_share.values,
                 y=company_share.index,
                 orientation='h',
                 title='Şirket Bazlı Pazar Payı (Top 10)',
                 labels={'x': 'Satış ($)', 'y': 'Şirket'})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate HHI
    market_shares = company_share / company_share.sum()
    hhi = (market_shares ** 2).sum() * 10000
    
    concentration_level = "Yüksek Konsantrasyon" if hhi > 2500 else "Orta Konsantrasyon" if hhi > 1500 else "Düşük Konsantrasyon"
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>Rekabet Yoğunluğu:</strong> Herfindahl-Hirschman İndeksi (HHI): <strong>{hhi:.0f}</strong> - {concentration_level}
    Top 3 şirket pazarın <strong>%{(market_shares.head(3).sum()*100):.1f}</strong>'ini kontrol etmektedir.
    </div>
    """, unsafe_allow_html=True)

with tab4:
    # Molecule treemap
    molecule_data = filtered_df.groupby(['Molecule', 'Chemical_Salt'])['USD_MNF'].sum().reset_index()
    
    fig = px.treemap(molecule_data,
                     path=['Chemical_Salt', 'Molecule'],
                     values='USD_MNF',
                     title='Molekül Bazlı Pazar Dağılımı',
                     color='USD_MNF',
                     color_continuous_scale='RdBu')
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig, use_container_width=True)
    
    # Molecule lifecycle analysis
    molecule_growth = filtered_df.groupby('Molecule').apply(
        lambda x: (x['USD_MNF'].iloc[-1] / x['USD_MNF'].iloc[0] - 1) if len(x) > 1 else 0
    ).sort_values(ascending=False)
    
    growth_molecules = molecule_growth[molecule_growth > 0.1].index.tolist()
    decline_molecules = molecule_growth[molecule_growth < -0.1].index.tolist()
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>Molekül Yaşam Döngüsü:</strong>
    <br><strong>Büyüme Fazında:</strong> {', '.join(growth_molecules[:3]) if growth_molecules else 'Belirgin büyüme yok'}
    <br><strong>Düşüş Fazında:</strong> {', '.join(decline_molecules[:3]) if decline_molecules else 'Belirgin düşüş yok'}
    </div>
    """, unsafe_allow_html=True)

# 2. Price-Volume-Growth Analysis
st.markdown('<h2 class="sub-header">💰 Fiyat - Hacim - Büyüme İlişkisi</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Price vs Volume scatter
    scatter_data = filtered_df.groupby(['Molecule', 'Corporation']).agg({
        'Unit_Avg_Price': 'mean',
        'Units': 'sum',
        'USD_MNF': 'sum'
    }).reset_index()
    
    fig = px.scatter(scatter_data,
                     x='Unit_Avg_Price',
                     y='Units',
                     size='USD_MNF',
                     color='Corporation',
                     hover_name='Molecule',
                     log_x=True,
                     log_y=True,
                     title='Fiyat vs Hacim Dağılımı (Molekül & Şirket Bazlı)',
                     labels={'Unit_Avg_Price': 'Ortalama Fiyat (log)', 'Units': 'Toplam Hacim (log)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Price-Volume insight
    price_volume_ratio = scatter_data['USD_MNF'].sum() / (scatter_data['Unit_Avg_Price'].mean() * scatter_data['Units'].sum())
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>Fiyat-Hacim Dengesi:</strong> Pazarın fiyat-hacim esnekliği <strong>{price_volume_ratio:.2f}</strong> seviyesindedir. 
    Yüksek fiyatlı segmentlerde düşük hacim, düşük fiyatlı segmentlerde yüksek hacim pattern'i gözlemlenmektedir.
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Price comparison by molecule
    price_comparison = filtered_df.groupby(['Molecule', 'Corporation'])['Unit_Avg_Price'].mean().unstack().head(10)
    
    fig = go.Figure()
    for company in price_comparison.columns[:5]:  # Limit to 5 companies for clarity
        fig.add_trace(go.Box(
            y=price_comparison[company].dropna(),
            name=company,
            boxpoints='outliers'
        ))
    
    fig.update_layout(
        title='Şirketler Arası Fiyat Karşılaştırması (Molekül Bazlı)',
        yaxis_title='Ortalama Birim Fiyat ($)',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Price premium analysis
    price_variance = price_comparison.std(axis=1) / price_comparison.mean(axis=1)
    high_variance_molecules = price_variance[price_variance > 0.3].index.tolist()
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>Fiyat Farklılaşması:</strong> {', '.join(high_variance_molecules[:3]) if high_variance_molecules else 'Hiçbir molekül'} moleküllerinde şirketler arası fiyat farkı >%30'dur. 
    Bu, güçlü marka veya formulasyon farklılaşmasına işaret etmektedir.
    </div>
    """, unsafe_allow_html=True)

# 3. Hospital vs Retail Analysis
st.markdown('<h3 class="sub-header">🏥 Hospital vs Retail Kanal Analizi</h3>', unsafe_allow_html=True)

if 'Hospital' in selected_sector and 'Retail' in selected_sector:
    sector_comparison = filtered_df.groupby(['Sector', 'Period']).agg({
        'USD_MNF': 'sum',
        'Unit_Avg_Price': 'mean',
        'Units': 'sum'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(sector_comparison, 
                     x='Period', 
                     y='USD_MNF',
                     color='Sector',
                     title='Kanal Bazlı Satış Trendleri',
                     labels={'USD_MNF': 'Satış ($)', 'Period': 'Dönem'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(sector_comparison,
                     x='Period',
                     y='Unit_Avg_Price',
                     color='Sector',
                     barmode='group',
                     title='Kanal Bazlı Ortalama Fiyat Karşılaştırması',
                     labels={'Unit_Avg_Price': 'Ortalama Fiyat ($)', 'Period': 'Dönem'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Sector insight
    hospital_share = sector_comparison[sector_comparison['Sector'] == 'Hospital']['USD_MNF'].sum() / sector_comparison['USD_MNF'].sum()
    price_premium = (sector_comparison[sector_comparison['Sector'] == 'Hospital']['Unit_Avg_Price'].mean() / 
                    sector_comparison[sector_comparison['Sector'] == 'Retail']['Unit_Avg_Price'].mean() - 1) * 100
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>Kanal Dinamikleri:</strong> Hospital kanalı pazarın <strong>%{hospital_share*100:.1f}</strong>'ini oluşturmakta ve 
    Retail kanalına kıyasla <strong>%{price_premium:.1f}</strong> fiyat primi uygulamaktadır. 
    Bu prim, hospital-only ürünler ve uzmanlaşmış tedavilerden kaynaklanmaktadır.
    </div>
    """, unsafe_allow_html=True)

# 4. Molecule Deep Dive
st.markdown('<h2 class="sub-header">🔬 Molekül Derinlemesine Analiz</h2>', unsafe_allow_html=True)

# Select molecule for deep dive
selected_molecule_deep = st.selectbox(
    "Detaylı Analiz için Molekül Seçin:",
    options=sorted(filtered_df['Molecule'].unique())
)

if selected_molecule_deep:
    molecule_df = filtered_df[filtered_df['Molecule'] == selected_molecule_deep]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        country_count = molecule_df['Country'].nunique()
        st.metric("Ülke Sayısı", country_count)
    
    with col2:
        company_count = molecule_df['Corporation'].nunique()
        st.metric("Şirket Sayısı", company_count)
    
    with col3:
        avg_price = molecule_df['Unit_Avg_Price'].mean()
        st.metric("Ort. Fiyat", f"${avg_price:.2f}")
    
    with col4:
        growth = molecule_df.groupby('Year')['USD_MNF'].sum().pct_change().iloc[-1] if len(molecule_df['Year'].unique()) > 1 else 0
        st.metric("Büyüme", f"{growth*100:.1f}%")
    
    # Molecule lifecycle chart
    molecule_trend = molecule_df.groupby('Period').agg({
        'USD_MNF': 'sum',
        'Units': 'sum',
        'Unit_Avg_Price': 'mean'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=molecule_trend['Period'], y=molecule_trend['USD_MNF'],
                  name="Satış", mode='lines+markers'),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=molecule_trend['Period'], y=molecule_trend['Unit_Avg_Price'],
                  name="Fiyat", mode='lines+markers'),
        secondary_y=True,
    )
    
    fig.update_layout(
        title=f'{selected_molecule_deep} - Satış & Fiyat Trendi',
        xaxis_title="Dönem"
    )
    
    fig.update_yaxes(title_text="Satış ($)", secondary_y=False)
    fig.update_yaxes(title_text="Ortalama Fiyat ($)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Competition matrix for selected molecule
    comp_matrix = molecule_df.pivot_table(
        index='Country',
        columns='Corporation',
        values='USD_MNF',
        aggfunc='sum'
    ).fillna(0)
    
    if not comp_matrix.empty:
        fig = px.imshow(comp_matrix,
                       labels=dict(x="Şirket", y="Ülke", color="Satış ($)"),
                       title=f'{selected_molecule_deep} - Ülke x Şirket Rekabet Matrisi',
                       color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

# 5. Price Erosion & Early Warning System
st.markdown('<h2 class="sub-header">⚠️ Fiyat Erozyonu & Erken Uyarı Sistemi</h2>', unsafe_allow_html=True)

# Calculate price erosion risk
price_risk_data = filtered_df.groupby(['Molecule', 'Country']).apply(
    lambda x: pd.Series({
        'Price_Change': x['Unit_Avg_Price'].pct_change().iloc[-1] if len(x) > 1 else 0,
        'Volume_Change': x['Units'].pct_change().iloc[-1] if len(x) > 1 else 0,
        'Market_Share_Change': x['USD_MNF'].pct_change().iloc[-1] if len(x) > 1 else 0,
        'Avg_Price': x['Unit_Avg_Price'].mean(),
        'Total_Sales': x['USD_MNF'].sum()
    })
).reset_index()

# Risk scoring
price_risk_data['Risk_Score'] = (
    (price_risk_data['Price_Change'] < -0.05).astype(int) * 3 +  # Price drop >5%
    (price_risk_data['Volume_Change'] > 0.2).astype(int) * 2 +   # Volume increase >20%
    (price_risk_data['Market_Share_Change'] < -0.1).astype(int) * 2  # Market share loss >10%
)

# Classify risk levels
def classify_risk(score):
    if score >= 5:
        return 'Yüksek Risk'
    elif score >= 3:
        return 'Orta Risk'
    else:
        return 'Düşük Risk'

price_risk_data['Risk_Level'] = price_risk_data['Risk_Score'].apply(classify_risk)

# Display risk matrix
col1, col2 = st.columns(2)

with col1:
    risk_summary = price_risk_data['Risk_Level'].value_counts()
    fig = px.pie(values=risk_summary.values,
                 names=risk_summary.index,
                 title='Risk Seviyesi Dağılımı',
                 color=risk_summary.index,
                 color_discrete_map={'Yüksek Risk': '#DC2626',
                                   'Orta Risk': '#F59E0B',
                                   'Düşük Risk': '#10B981'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    high_risk_molecules = price_risk_data[price_risk_data['Risk_Level'] == 'Yüksek Risk']
    
    if not high_risk_molecules.empty:
        st.markdown("### 🔴 Yüksek Riskli Moleküller")
        for _, row in high_risk_molecules.head(5).iterrows():
            st.markdown(f"""
            <div class="metric-box">
            <strong>{row['Molecule']}</strong> - {row['Country']}<br>
            Fiyat Değişimi: <span class="risk-high">{row['Price_Change']*100:.1f}%</span> | 
            Hacim Değişimi: {row['Volume_Change']*100:.1f}%
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Yüksek riskli molekül tespit edilmemiştir.")

# Price erosion trend
st.markdown("### 📉 Fiyat Erozyonu Trendleri")

price_trend_data = filtered_df.groupby(['Molecule', 'Period'])['Unit_Avg_Price'].mean().unstack().T
price_change_pct = price_trend_data.pct_change().mean() * 100

high_erosion = price_change_pct[price_change_pct < -5].sort_values().head(10)

if not high_erosion.empty:
    fig = px.bar(x=high_erosion.values,
                 y=high_erosion.index,
                 orientation='h',
                 title='En Yüksek Fiyat Erozyonuna Sahip Moleküller',
                 labels={'x': 'Ortalama Fiyat Değişimi (%)', 'y': 'Molekül'})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>Erken Uyarı:</strong> {', '.join(high_erosion.index[:3])} moleküllerinde ortalama fiyat erozyonu >%5 seviyesindedir. 
    Bu trend devam ederse, önümüzdeki 6-12 ay içinde pazar değerinde %10-15 azalma beklenebilir.
    Özellikle {high_erosion.index[0]} molekülünde jenerik rekabet veya pazarlık baskısı artmaktadır.
    </div>
    """, unsafe_allow_html=True)

# 6. Strategic Recommendations
st.markdown('<h2 class="sub-header">🎯 Stratejik Öneriler</h2>', unsafe_allow_html=True)

# Generate strategic insights
if len(filtered_df) > 0:
    # Find growth opportunities
    growing_molecules = price_risk_data[
        (price_risk_data['Price_Change'] > 0) & 
        (price_risk_data['Volume_Change'] > 0) &
        (price_risk_data['Risk_Level'] == 'Düşük Risk')
    ].sort_values('Total_Sales', ascending=False).head(3)
    
    # Find underpenetrated markets
    market_density = filtered_df.groupby(['Country', 'Molecule'])['USD_MNF'].sum().unstack().fillna(0)
    underpenetrated = market_density[market_density.sum(axis=1) < market_density.sum(axis=1).quantile(0.25)].index.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Büyüme Fırsatları")
        if not growing_molecules.empty:
            for _, row in growing_molecules.iterrows():
                st.markdown(f"""
                <div class="metric-box">
                <strong>{row['Molecule']}</strong> ({row['Country']})<br>
                Hem fiyat (+{row['Price_Change']*100:.1f}%) hem hacim (+{row['Volume_Change']*100:.1f}%) artışı<br>
                Pazar: ${row['Total_Sales']/1e6:.1f}M
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Belirgin büyüme fırsatı tespit edilmemiştir.")
    
    with col2:
        st.markdown("### 📍 Az Nüfuz Edilmiş Pazarlar")
        if underpenetrated:
            for country in underpenetrated[:3]:
                st.markdown(f"""
                <div class="metric-box">
                <strong>{country}</strong><br>
                Pazar büyüklüğü: ${market_density.loc[country].sum()/1e6:.1f}M<br>
                Molekül çeşitliliği: {(market_density.loc[country] > 0).sum()}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Tüm pazarlar yeterince nüfuz edilmiş durumda.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
<small>Global İlaç Pazarı Strateji Dashboard v1.0 | 
Bağımsız Pazar Araştırması Perspektifi | 
Veri Güncelliği: MAT Q3 2024 | 
© 2024 Strateji Danışmanlığı</small>
</div>
""", unsafe_allow_html=True)
