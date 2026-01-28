# app.py - Global İlaç Pazarı Dashboard (Streamlit Uyumlu)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy import stats
import statsmodels.api as sm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import json
import math

# ================================================
# 1. KONFİGÜRASYON VE STİL AYARLARI
# ================================================
st.set_page_config(
    page_title="Global Pharma Market Intelligence Pro",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS
st.markdown("""
<style>
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
    
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #E5E7EB;
        transition: transform 0.2s, box-shadow 0.2s;
        margin: 0.5rem 0;
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
</style>
""", unsafe_allow_html=True)

# ================================================
# 2. VERİ YÜKLEME VE ÖN İŞLEME FONKSİYONLARI
# ================================================
@st.cache_data(ttl=3600)
def load_comprehensive_data():
    """Kapsamlı sentetik veri oluşturma"""
    np.random.seed(42)
    
    countries = ['Türkiye', 'Almanya', 'Fransa', 'İtalya', 'İspanya', 'İngiltere',
                 'Polonya', 'Hollanda', 'Belçika', 'İsviçre', 'İsveç', 'Norveç',
                 'Danimarka', 'Finlandiya', 'Avusturya', 'Portekiz', 'Yunanistan']
    
    companies = ['Pfizer', 'Novartis', 'Roche', 'Merck', 'Sanofi', 'GSK', 
                 'AstraZeneca', 'Johnson & Johnson', 'Bayer', 'AbbVie']
    
    molecules = ['Adalimumab', 'Pembrolizumab', 'Nivolumab', 'Rituximab', 
                 'Trastuzumab', 'Bevacizumab', 'Insulin Glargine', 'Sitagliptin',
                 'Atorvastatin', 'Apremilast', 'Dupilumab', 'Semaglutide']
    
    therapeutic_areas = ['Onkoloji', 'Otoimmün', 'Diyabet', 'Kardiyovasküler']
    
    # Molekül bilgileri
    molecules_db = {}
    for mol in molecules:
        if 'umab' in mol:
            molecules_db[mol] = {'Category': 'Specialty', 'Class': 'Biyolojik'}
        elif 'tinib' in mol or 'mab' in mol:
            molecules_db[mol] = {'Category': 'Specialty', 'Class': 'Hedefe Yönelik'}
        else:
            molecules_db[mol] = {'Category': 'Non-Specialty', 'Class': 'Kimyasal'}
    
    # Veri oluşturma
    all_data = []
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    years = ['2021', '2022', '2023', '2024']
    
    for country in countries:
        for year in years:
            for quarter in quarters:
                period = f"{year}-{quarter}"
                
                for molecule in molecules:
                    # Molekül için şirket atama
                    if molecule in ['Adalimumab', 'Apremilast']:
                        molecule_companies = ['AbbVie', 'Amgen']
                    elif molecule in ['Pembrolizumab', 'Sitagliptin']:
                        molecule_companies = ['Merck']
                    elif molecule in ['Semaglutide']:
                        molecule_companies = ['Novo Nordisk']
                    else:
                        molecule_companies = list(np.random.choice(companies, size=2, replace=False))
                    
                    for company in molecule_companies:
                        # Satış değeri oluştur
                        base_sales = np.random.lognormal(10, 1) * 1000
                        
                        # Kanal dağılımı
                        if molecules_db[molecule]['Category'] == 'Specialty':
                            hospital_share = np.random.beta(8, 2)
                        else:
                            hospital_share = np.random.beta(2, 8)
                        
                        hospital_sales = base_sales * hospital_share
                        retail_sales = base_sales * (1 - hospital_share)
                        
                        # Terapötik alan atama
                        if molecule in ['Adalimumab', 'Rituximab', 'Apremilast', 'Dupilumab']:
                            ta = 'Otoimmün'
                        elif molecule in ['Pembrolizumab', 'Nivolumab', 'Trastuzumab', 'Bevacizumab']:
                            ta = 'Onkoloji'
                        elif molecule in ['Insulin Glargine', 'Sitagliptin', 'Semaglutide']:
                            ta = 'Diyabet'
                        else:
                            ta = 'Kardiyovasküler'
                        
                        # Hospital kanalı
                        all_data.append({
                            'Country': country,
                            'Corporation': company,
                            'Molecule': molecule,
                            'Therapeutic_Area': ta,
                            'Product': f"{molecule} {np.random.choice(['SC', 'IV', 'Oral'])}",
                            'Specialty_Flag': molecules_db[molecule]['Category'],
                            'Period': period,
                            'Year': year,
                            'Quarter': quarter,
                            'Sector': 'Hospital',
                            'USD_MNF': hospital_sales,
                            'Units': hospital_sales / np.random.uniform(50, 200),
                            'Unit_Avg_Price': np.random.uniform(100, 500)
                        })
                        
                        # Retail kanalı
                        all_data.append({
                            'Country': country,
                            'Corporation': company,
                            'Molecule': molecule,
                            'Therapeutic_Area': ta,
                            'Product': f"{molecule} {np.random.choice(['SC', 'IV', 'Oral'])}",
                            'Specialty_Flag': molecules_db[molecule]['Category'],
                            'Period': period,
                            'Year': year,
                            'Quarter': quarter,
                            'Sector': 'Retail',
                            'USD_MNF': retail_sales,
                            'Units': retail_sales / np.random.uniform(30, 150),
                            'Unit_Avg_Price': np.random.uniform(80, 400)
                        })
    
    df = pd.DataFrame(all_data)
    
    # Ek hesaplamalar
    df['Market_Share'] = df.groupby(['Period', 'Therapeutic_Area'])['USD_MNF'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0
    )
    
    # Büyüme oranları (basitleştirilmiş)
    df['Sales_Growth_YoY'] = df.groupby(['Country', 'Corporation', 'Molecule', 'Sector', 'Quarter'])['USD_MNF'].transform(
        lambda x: x.pct_change(4) if len(x) > 4 else 0
    )
    
    # Patent durumu (basitleştirilmiş)
    df['Patent_Status'] = np.where(df['Molecule'].str.contains('umab'), 'On-Patent', 'Off-Patent')
    
    # Region bilgisi
    def assign_region(country):
        if country in ['Almanya', 'Fransa', 'Hollanda', 'Belçika', 'Avusturya', 'İsviçre']:
            return 'Batı Avrupa'
        elif country in ['İtalya', 'İspanya', 'Portekiz', 'Yunanistan']:
            return 'Güney Avrupa'
        elif country in ['İsveç', 'Norveç', 'Danimarka', 'Finlandiya']:
            return 'Kuzey Avrupa'
        else:
            return 'Doğu Avrupa'
    
    df['Region'] = df['Country'].apply(assign_region)
    
    return df, companies, molecules_db, therapeutic_areas

# ================================================
# 3. ANALİTİK FONKSİYONLAR
# ================================================
class PharmaAnalytics:
    @staticmethod
    def calculate_market_concentration(df, group_cols=['Country', 'Period'], top_n=3):
        """Pazar konsantrasyonu hesaplama"""
        concentration = {}
        for idx, group in df.groupby(group_cols):
            total_sales = group['USD_MNF'].sum()
            if total_sales > 0:
                top_companies = group.groupby('Corporation')['USD_MNF'].sum().nlargest(top_n).sum()
                hhi = ((group.groupby('Corporation')['USD_MNF'].sum() / total_sales) ** 2).sum() * 10000
                
                concentration[idx] = {
                    'HHI': hhi,
                    'Top3_Share': top_companies / total_sales,
                    'CR4': group.groupby('Corporation')['USD_MNF'].sum().nlargest(4).sum() / total_sales
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
                
                if not price_changes.empty and not volume_changes.empty:
                    recent_price_change = price_changes.iloc[-1]
                    recent_volume_change = volume_changes.iloc[-1]
                    
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
    def identify_white_spaces(df, min_market_size=1000000):
        """Beyaz alan (white space) tespiti"""
        # Basitleştirilmiş versiyon
        all_countries = df['Country'].unique()
        all_molecules = df['Molecule'].unique()
        
        white_space_analysis = []
        
        for country in all_countries:
            for molecule in all_molecules:
                market_data = df[(df['Country'] == country) & (df['Molecule'] == molecule)]
                
                if market_data.empty or market_data['USD_MNF'].sum() < min_market_size:
                    # Benzer ülkelerdeki performans
                    similar_countries = df[
                        (df['Region'] == df[df['Country'] == country]['Region'].iloc[0] if not df[df['Country'] == country].empty else '') & 
                        (df['Molecule'] == molecule)
                    ]
                    
                    if not similar_countries.empty:
                        avg_sales = similar_countries.groupby('Country')['USD_MNF'].sum().mean()
                        avg_growth = similar_countries['Sales_Growth_YoY'].mean()
                        
                        white_space_analysis.append({
                            'Country': country,
                            'Molecule': molecule,
                            'Therapeutic_Area': similar_countries['Therapeutic_Area'].iloc[0] if not similar_countries.empty else 'Bilinmiyor',
                            'Avg_Similar_Market_Size': avg_sales,
                            'Avg_Growth_Rate': avg_growth if not pd.isna(avg_growth) else 0,
                            'Competitor_Count': similar_countries['Corporation'].nunique(),
                            'Potential_Score': avg_sales * (1 + avg_growth) / max(1, similar_countries['Corporation'].nunique())
                        })
        
        return pd.DataFrame(white_space_analysis).sort_values('Potential_Score', ascending=False).head(20)

# ================================================
# 4. GÖRSELLEŞTİRME FONKSİYONLARI
# ================================================
class PharmaVisualizations:
    @staticmethod
    def create_market_evolution_chart(df, metric='USD_MNF', group_by='Country'):
        """Pazar evrim chart'ı"""
        try:
            # Veriyi hazırla
            df['Period'] = pd.Categorical(df['Period'], categories=sorted(df['Period'].unique()), ordered=True)
            trend_data = df.groupby(['Period', group_by])[metric].sum().reset_index()
            
            fig = px.line(trend_data, x='Period', y=metric, color=group_by,
                         title='Pazar Büyüklüğü Trendi',
                         markers=True)
            
            fig.update_layout(height=400)
            return fig
        except Exception as e:
            st.error(f"Chart oluşturma hatası: {e}")
            return None
    
    @staticmethod
    def create_competitive_landscape(df, dimension1='Unit_Avg_Price', dimension2='Sales_Growth_YoY'):
        """Rekabet manzarası haritası"""
        try:
            company_metrics = df.groupby('Corporation').agg({
                dimension1: 'mean',
                dimension2: 'mean',
                'USD_MNF': 'sum',
                'Market_Share': 'mean'
            }).reset_index()
            
            fig = px.scatter(company_metrics,
                            x=dimension1,
                            y=dimension2,
                            size='USD_MNF',
                            color='Market_Share',
                            hover_name='Corporation',
                            size_max=60,
                            color_continuous_scale='RdYlBu',
                            title='Rekabet Manzarası')
            
            # Quadrant çizgileri
            x_median = company_metrics[dimension1].median()
            y_median = company_metrics[dimension2].median()
            
            fig.add_hline(y=y_median, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=x_median, line_dash="dash", line_color="gray", opacity=0.5)
            
            return fig
        except:
            return None
    
    @staticmethod
    def create_therapeutic_area_sunburst(df):
        """Terapötik alan sunburst chart"""
        try:
            ta_data = df.groupby(['Therapeutic_Area', 'Molecule', 'Corporation']).agg({
                'USD_MNF': 'sum',
                'Sales_Growth_YoY': 'mean'
            }).reset_index()
            
            fig = px.sunburst(ta_data,
                             path=['Therapeutic_Area', 'Molecule', 'Corporation'],
                             values='USD_MNF',
                             color='Sales_Growth_YoY',
                             color_continuous_scale='RdYlBu',
                             title='Terapötik Alan Hiyerarşisi')
            
            fig.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=500)
            return fig
        except:
            return None

# ================================================
# 5. ANA UYGULAMA
# ================================================
def main():
    # Veri yükleme
    with st.spinner("Veri yükleniyor..."):
        df, companies, molecules_db, therapeutic_areas = load_comprehensive_data()
    
    # Başlık
    st.markdown('<h1 class="main-title">💊 GLOBAL İLAÇ PAZARI STRATEJİ İSTİHBARAT PLATFORMU</h1>', 
                unsafe_allow_html=True)
    
    # Dashboard açıklaması
    with st.expander("📋 Dashboard Kullanım Kılavuzu", expanded=False):
        st.markdown("""
        Bu dashboard, global ilaç pazarı verilerini analiz etmek için tasarlanmıştır.
        
        **Temel Özellikler:**
        - Gerçek zamanlı veri filtreleme
        - Pazar büyüklüğü ve trend analizi
        - Rekabet analizi
        - Risk değerlendirmesi
        - Büyüme fırsatları tespiti
        """)
    
    # ================================================
    # FİLTRELEME PANELİ
    # ================================================
    st.sidebar.markdown("## 🔍 FİLTRELER")
    
    # Çoklu seçim filtreleri
    with st.sidebar.expander("📍 Coğrafi Filtreler", expanded=True):
        selected_countries = st.multiselect(
            "Ülkeler",
            options=sorted(df['Country'].unique()),
            default=sorted(df['Country'].unique())[:3]
        )
        
        selected_regions = st.multiselect(
            "Bölgeler",
            options=sorted(df['Region'].unique()),
            default=sorted(df['Region'].unique())
        )
    
    with st.sidebar.expander("🏢 Şirket & Molekül", expanded=True):
        selected_companies = st.multiselect(
            "Şirketler",
            options=sorted(df['Corporation'].unique()),
            default=sorted(df['Corporation'].unique())[:3]
        )
        
        selected_molecules = st.multiselect(
            "Moleküller",
            options=sorted(df['Molecule'].unique()),
            default=sorted(df['Molecule'].unique())[:5]
        )
        
        selected_ta = st.multiselect(
            "Terapötik Alanlar",
            options=sorted(df['Therapeutic_Area'].unique()),
            default=sorted(df['Therapeutic_Area'].unique())
        )
    
    with st.sidebar.expander("📈 Pazar Segmentleri", expanded=True):
        selected_sectors = st.multiselect(
            "Kanallar",
            options=sorted(df['Sector'].unique()),
            default=sorted(df['Sector'].unique())
        )
    
    with st.sidebar.expander("⏰ Zaman Periyodu", expanded=True):
        selected_years = st.multiselect(
            "Yıllar",
            options=sorted(df['Year'].unique()),
            default=sorted(df['Year'].unique())[-2:]
        )
        
        selected_quarters = st.multiselect(
            "Çeyrekler",
            options=sorted(df['Quarter'].unique()),
            default=sorted(df['Quarter'].unique())
        )
    
    # Filtreleri uygula
    filter_conditions = []
    
    if selected_countries:
        filter_conditions.append(df['Country'].isin(selected_countries))
    if selected_regions:
        filter_conditions.append(df['Region'].isin(selected_regions))
    if selected_companies:
        filter_conditions.append(df['Corporation'].isin(selected_companies))
    if selected_molecules:
        filter_conditions.append(df['Molecule'].isin(selected_molecules))
    if selected_ta:
        filter_conditions.append(df['Therapeutic_Area'].isin(selected_ta))
    if selected_sectors:
        filter_conditions.append(df['Sector'].isin(selected_sectors))
    if selected_years:
        filter_conditions.append(df['Year'].isin(selected_years))
    if selected_quarters:
        filter_conditions.append(df['Quarter'].isin(selected_quarters))
    
    if filter_conditions:
        filtered_df = df.copy()
        for condition in filter_conditions:
            filtered_df = filtered_df[condition]
    else:
        filtered_df = df.copy()
    
    # Filtrelenmiş veri kontrolü
    if filtered_df.empty:
        st.warning("Seçilen filtrelerle eşleşen veri bulunamadı. Lütfen filtreleri genişletin.")
        return
    
    # ================================================
    # KPI PANELİ
    # ================================================
    st.markdown("## 📈 PİYASA GÖSTERGELERİ")
    
    # Ana KPI'lar
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = filtered_df['USD_MNF'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">TOPLAM PAZAR</div>
            <div class="metric-value">${total_sales/1e6:.1f}M</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_growth = filtered_df['Sales_Growth_YoY'].mean() * 100
        growth_color = "#10B981" if avg_growth > 0 else "#EF4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">YILLIK BÜYÜME</div>
            <div class="metric-value">{avg_growth:.1f}%</div>
            <div style="color: {growth_color}; font-size: 0.9rem;">
                {'↑ Büyüme' if avg_growth > 0 else '↓ Küçülme'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_price = filtered_df['Unit_Avg_Price'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ORT. BİRİM FİYAT</div>
            <div class="metric-value">${avg_price:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        concentration = PharmaAnalytics.calculate_market_concentration(filtered_df)
        hhi = list(concentration.values())[0]['HHI'] if concentration else 0
        hhi_color = "#EF4444" if hhi > 2500 else "#F59E0B" if hhi > 1500 else "#10B981"
        hhi_text = "Yüksek" if hhi > 2500 else "Orta" if hhi > 1500 else "Düşük"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">REKABET YOĞUNLUĞU</div>
            <div class="metric-value">{hhi:.0f} HHI</div>
            <div style="color: {hhi_color}; font-size: 0.9rem;">
                {hhi_text} Konsantrasyon
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # İkincil KPI'lar
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        molecule_count = filtered_df['Molecule'].nunique()
        st.metric("Molekül Çeşitliliği", str(molecule_count))
    
    with col6:
        company_count = filtered_df['Corporation'].nunique()
        st.metric("Aktif Şirket", str(company_count))
    
    with col7:
        country_count = filtered_df['Country'].nunique()
        st.metric("Ülke Kapsamı", str(country_count))
    
    with col8:
        specialty_share = filtered_df[filtered_df['Specialty_Flag'] == 'Specialty']['USD_MNF'].sum() / total_sales * 100
        st.metric("Specialty Payı", f"{specialty_share:.1f}%")
    
    # ================================================
    # İÇGÖRÜLER
    # ================================================
    st.markdown("## 💡 İÇGÖRÜLER")
    
    # Basit içgörüler
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        top_country = filtered_df.groupby('Country')['USD_MNF'].sum().idxmax()
        country_sales = filtered_df.groupby('Country')['USD_MNF'].sum().max()
        st.markdown(f"""
        <div class="insight-card">
            <strong>🏆 Lider Ülke</strong><br>
            {top_country}<br>
            <small>${country_sales/1e6:.1f}M satış</small>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        top_molecule = filtered_df.groupby('Molecule')['USD_MNF'].sum().idxmax()
        molecule_sales = filtered_df.groupby('Molecule')['USD_MNF'].sum().max()
        st.markdown(f"""
        <div class="insight-card success">
            <strong>🧪 En Çok Satılan Molekül</strong><br>
            {top_molecule}<br>
            <small>${molecule_sales/1e6:.1f}M satış</small>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col3:
        top_company = filtered_df.groupby('Corporation')['USD_MNF'].sum().idxmax()
        company_sales = filtered_df.groupby('Corporation')['USD_MNF'].sum().max()
        st.markdown(f"""
        <div class="insight-card">
            <strong>🏢 Lider Şirket</strong><br>
            {top_company}<br>
            <small>${company_sales/1e6:.1f}M satış</small>
        </div>
        """, unsafe_allow_html=True)
    
    # ================================================
    # ANA ANALİZ BÖLÜMLERİ
    # ================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 PAZAR YAPISI", 
        "💰 FİYAT ANALİZİ", 
        "⚔️ REKABET ANALİZİ", 
        "🚀 BÜYÜME FIRSATLARI"
    ])
    
    # TAB 1: PAZAR YAPISI
    with tab1:
        st.markdown('<h3 class="section-title">1. PAZAR YAPISI & DINAMIKLERI</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Pazar evrim chart'ı
            fig = PharmaVisualizations.create_market_evolution_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Terapötik alan analizi
            st.markdown('<h4 class="subsection-title">Terapötik Alan Dağılımı</h4>', unsafe_allow_html=True)
            fig2 = PharmaVisualizations.create_therapeutic_area_sunburst(filtered_df)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Molekül performansı
            st.markdown('<h4 class="subsection-title">Molekül Performansı</h4>', unsafe_allow_html=True)
            molecule_perf = filtered_df.groupby('Molecule').agg({
                'USD_MNF': 'sum',
                'Sales_Growth_YoY': 'mean',
                'Unit_Avg_Price': 'mean'
            }).round(2).sort_values('USD_MNF', ascending=False)
            
            st.dataframe(molecule_perf.head(10), use_container_width=True)
            
            # Ülke bazlı pazar payı
            st.markdown('<h4 class="subsection-title">Ülke Bazlı Pazar Payı</h4>', unsafe_allow_html=True)
            country_share = filtered_df.groupby('Country')['USD_MNF'].sum().sort_values(ascending=False)
            fig3 = px.pie(values=country_share.values, names=country_share.index, 
                         title='Ülke Bazlı Pazar Dağılımı')
            st.plotly_chart(fig3, use_container_width=True)
    
    # TAB 2: FİYAT ANALİZİ
    with tab2:
        st.markdown('<h3 class="section-title">2. FİYAT ANALİZİ</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fiyat dağılımı
            st.markdown('<h4 class="subsection-title">Fiyat Dağılımı</h4>', unsafe_allow_html=True)
            fig = px.histogram(filtered_df, x='Unit_Avg_Price', 
                              title='Birim Fiyat Dağılımı',
                              nbins=30)
            st.plotly_chart(fig, use_container_width=True)
            
            # Molekül bazlı fiyat karşılaştırması
            st.markdown('<h4 class="subsection-title">Molekül Bazlı Fiyat Karşılaştırması</h4>', unsafe_allow_html=True)
            price_by_molecule = filtered_df.groupby('Molecule')['Unit_Avg_Price'].mean().sort_values(ascending=False)
            fig2 = px.bar(price_by_molecule.head(10), 
                         title='En Yüksek Fiyatlı 10 Molekül')
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Fiyat erozyonu analizi
            st.markdown('<h4 class="subsection-title">Fiyat Erozyonu Analizi</h4>', unsafe_allow_html=True)
            erosion_df = PharmaAnalytics.detect_price_erosion(filtered_df)
            
            if not erosion_df.empty:
                st.dataframe(erosion_df[['Country', 'Molecule', 'Corporation', 'Risk_Level', 'Price_Change']], 
                           use_container_width=True)
                
                # Risk dağılımı
                risk_summary = erosion_df['Risk_Level'].value_counts()
                fig3 = px.pie(values=risk_summary.values, names=risk_summary.index,
                             title='Fiyat Erozyonu Risk Dağılımı')
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Fiyat erozyonu tespit edilmedi.")
            
            # Fiyat-hacim ilişkisi
            st.markdown('<h4 class="subsection-title">Fiyat-Hacim İlişkisi</h4>', unsafe_allow_html=True)
            fig4 = px.scatter(filtered_df.sample(min(1000, len(filtered_df))), 
                             x='Unit_Avg_Price', y='Units',
                             hover_data=['Molecule', 'Country'],
                             title='Fiyat-Hacim İlişkisi')
            st.plotly_chart(fig4, use_container_width=True)
    
    # TAB 3: REKABET ANALİZİ
    with tab3:
        st.markdown('<h3 class="section-title">3. REKABET ANALİZİ</h3>', unsafe_allow_html=True)
        
        # Rekabet manzarası
        fig = PharmaVisualizations.create_competitive_landscape(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Şirket performansı
            st.markdown('<h4 class="subsection-title">Şirket Performans Karşılaştırması</h4>', unsafe_allow_html=True)
            company_perf = filtered_df.groupby('Corporation').agg({
                'USD_MNF': 'sum',
                'Market_Share': 'mean',
                'Sales_Growth_YoY': 'mean'
            }).sort_values('USD_MNF', ascending=False)
            
            fig2 = px.bar(company_perf.head(10), x=company_perf.head(10).index, y='USD_MNF',
                         title='Top 10 Şirket - Satış Performansı')
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Pazar konsantrasyonu
            st.markdown('<h4 class="subsection-title">Pazar Konsantrasyonu</h4>', unsafe_allow_html=True)
            concentration = PharmaAnalytics.calculate_market_concentration(filtered_df)
            
            if concentration:
                conc_df = pd.DataFrame.from_dict(concentration, orient='index')
                conc_df = conc_df.reset_index()
                
                if 'index' in conc_df.columns:
                    conc_df[['Country', 'Period']] = pd.DataFrame(conc_df['index'].tolist(), index=conc_df.index)
                    
                    fig3 = px.line(conc_df, x='Period', y='HHI', color='Country',
                                  title='Pazar Konsantrasyonu Trendi (HHI)')
                    st.plotly_chart(fig3, use_container_width=True)
    
    # TAB 4: BÜYÜME FIRSATLARI
    with tab4:
        st.markdown('<h3 class="section-title">4. BÜYÜME FIRSATLARI</h3>', unsafe_allow_html=True)
        
        # Beyaz alan analizi
        white_spaces_df = PharmaAnalytics.identify_white_spaces(filtered_df)
        
        if not white_spaces_df.empty:
            st.markdown('<h4 class="subsection-title">Beyaz Alan Fırsatları</h4>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top fırsatlar tablosu
                st.dataframe(white_spaces_df[['Country', 'Molecule', 'Therapeutic_Area', 'Potential_Score']].head(10),
                           use_container_width=True)
            
            with col2:
                # Fırsat haritası
                fig = px.scatter(white_spaces_df.head(20),
                                x='Avg_Similar_Market_Size',
                                y='Avg_Growth_Rate',
                                size='Potential_Score',
                                color='Therapeutic_Area',
                                hover_name='Molecule',
                                title='Büyüme Fırsatları Haritası',
                                labels={'Avg_Similar_Market_Size': 'Benzer Pazar Büyüklüğü',
                                       'Avg_Growth_Rate': 'Ortalama Büyüme Oranı'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Büyüme trendleri
        st.markdown('<h4 class="subsection-title">Büyüme Trendleri</h4>', unsafe_allow_html=True)
        
        growth_trends = filtered_df.groupby(['Year', 'Therapeutic_Area'])['USD_MNF'].sum().unstack()
        
        if not growth_trends.empty:
            fig = px.line(growth_trends, title='Terapötik Alan Bazlı Büyüme Trendleri')
            st.plotly_chart(fig, use_container_width=True)
    
    # ================================================
    # FOOTER
    # ================================================
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
    <em>Global Pharma Market Intelligence Pro | © 2024</em>
    </div>
    """, unsafe_allow_html=True)

# ================================================
# 6. UYGULAMA BAŞLATMA
# ================================================
if __name__ == "__main__":
    main()
