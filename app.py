# app.py - Profesyonel İlaç Pazarı Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Advanced analytics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import statsmodels.api as sm

# Utilities
from datetime import datetime, timedelta
import json
from io import BytesIO
import time
import gc

# ================================================
# 1. PROFESYONEL KONFİGÜRASYON VE STİL AYARLARI
# ================================================
st.set_page_config(
    page_title="PharmaIntelligence Pro | İlaç Pazarı Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PROFESYONEL DARK THEME CSS STYLES
PROFESSIONAL_CSS = """
<style>
    /* === ROOT VARIABLES === */
    :root {
        --primary-dark: #1a1b26;
        --secondary-dark: #24283b;
        --accent-blue: #7aa2f7;
        --accent-purple: #bb9af7;
        --accent-green: #9ece6a;
        --accent-yellow: #e0af68;
        --accent-red: #f7768e;
        
        --text-primary: #c0caf5;
        --text-secondary: #a9b1d6;
        --text-muted: #565f89;
        
        --bg-primary: #1a1b26;
        --bg-secondary: #24283b;
        --bg-card: #292e42;
        --bg-hover: #414868;
        
        --success: #73daca;
        --warning: #ff9e64;
        --danger: #ff757f;
        --info: #7aa2f7;
        
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
        
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        
        --transition-fast: 200ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-normal: 300ms cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* === GLOBAL STYLES === */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--text-primary);
    }
    
    /* Fix text colors */
    .stMarkdown, .stText, .stDataFrame, 
    .stSelectbox label, .stMultiselect label,
    .stSlider label, .stNumberInput label {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }
    
    /* === TYPOGRAPHY === */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        color: var(--text-primary) !important;
    }
    
    .pharma-title {
        font-size: 2.5rem;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--accent-blue);
    }
    
    .section-title {
        font-size: 1.6rem;
        color: var(--text-primary) !important;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-left: 0.5rem;
        border-left: 4px solid var(--accent-blue);
    }
    
    .subsection-title {
        font-size: 1.3rem;
        color: var(--text-primary) !important;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
    }
    
    /* === METRIC CARDS === */
    .metric-card {
        background: var(--bg-card);
        padding: 1.2rem;
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-md);
        border: 1px solid var(--bg-hover);
        transition: all var(--transition-normal);
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg);
        border-color: var(--accent-blue);
    }
    
    .metric-card.premium {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        color: white !important;
    }
    
    .metric-card.warning {
        background: linear-gradient(135deg, var(--accent-yellow), #ff9e64);
        color: white !important;
    }
    
    .metric-card.danger {
        background: linear-gradient(135deg, var(--accent-red), #ff757f);
        color: white !important;
    }
    
    .metric-card.success {
        background: linear-gradient(135deg, var(--accent-green), #73daca);
        color: white !important;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.3rem 0;
        color: var(--text-primary) !important;
    }
    
    .metric-card.premium .metric-value,
    .metric-card.warning .metric-value,
    .metric-card.danger .metric-value,
    .metric-card.success .metric-value {
        color: white !important;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* === INSIGHT CARDS === */
    .insight-card {
        background: var(--bg-card);
        padding: 1.2rem;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        border-left: 5px solid;
        margin: 0.8rem 0;
        transition: all var(--transition-fast);
    }
    
    .insight-card.info {
        border-left-color: var(--info);
        background: linear-gradient(135deg, rgba(122, 162, 247, 0.1), rgba(122, 162, 247, 0.05));
    }
    
    .insight-card.success {
        border-left-color: var(--success);
        background: linear-gradient(135deg, rgba(115, 218, 202, 0.1), rgba(115, 218, 202, 0.05));
    }
    
    .insight-card.warning {
        border-left-color: var(--warning);
        background: linear-gradient(135deg, rgba(255, 158, 100, 0.1), rgba(255, 158, 100, 0.05));
    }
    
    .insight-card.danger {
        border-left-color: var(--danger);
        background: linear-gradient(135deg, rgba(247, 118, 142, 0.1), rgba(247, 118, 142, 0.05));
    }
    
    .insight-card:hover {
        transform: translateX(3px);
        box-shadow: var(--shadow-md);
    }
    
    .insight-title {
        font-weight: 700;
        color: var(--text-primary) !important;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .insight-content {
        color: var(--text-secondary) !important;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    
    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-card);
        padding: 0.5rem;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1.2rem;
        font-weight: 600;
        color: var(--text-secondary) !important;
        border-radius: var(--radius-sm);
        transition: all var(--transition-fast);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
        color: white !important;
        box-shadow: var(--shadow-md);
    }
    
    /* === SIDEBAR === */
    section[data-testid="stSidebar"] {
        background: var(--primary-dark);
        border-right: 1px solid var(--bg-hover);
    }
    
    .sidebar-title {
        color: var(--text-primary) !important;
        font-size: 1.3rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid var(--accent-blue);
    }
    
    /* === BUTTONS === */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue)) !important;
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    /* === EXPANDERS === */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    
    /* === PROGRESS BAR === */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple)) !important;
    }
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ================================================
# 2. VERİ YÜKLEME VE İŞLEME
# ================================================

class DataProcessor:
    """Veri işleme ve optimizasyon sınıfı"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_large_excel(file, sample_size=None):
        """Büyük Excel dosyasını yükle (100MB'a kadar)"""
        try:
            file_size = file.size / (1024 ** 2)  # MB cinsinden
            
            if file_size > 50:  # 50MB'tan büyükse chunk oku
                chunks = []
                chunk_size = 50000
                
                # İlk 1000 satırı okuyarak yapıyı anla
                preview_df = pd.read_excel(file, nrows=1000)
                
                # Toplam satır sayısını tahmin et
                total_chunks = max(1, int((file_size * 1000000) / (chunk_size * 100)))
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Chunk'ları oku
                for i in range(0, len(preview_df) if sample_size else total_chunks * chunk_size, chunk_size):
                    if sample_size and i >= sample_size:
                        break
                    
                    chunk = pd.read_excel(file, skiprows=i, nrows=chunk_size, header=0)
                    if not chunk.empty:
                        chunks.append(chunk)
                    
                    progress = min((i + 1) / (sample_size if sample_size else total_chunks * chunk_size), 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"📊 Veri yükleniyor: {len(pd.concat(chunks, ignore_index=True)):,} satır")
                
                df = pd.concat(chunks, ignore_index=True)
                progress_bar.empty()
                status_text.empty()
                
            else:
                # Küçük dosyalar için direk okuma
                df = pd.read_excel(file)
            
            # Örneklem alma
            if sample_size and sample_size < len(df):
                df = df.sample(min(sample_size, len(df)), random_state=42)
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            return None
    
    @staticmethod
    def optimize_dataframe(df):
        """DataFrame'i optimize et"""
        # Sütun isimlerini temizle
        df.columns = [str(col).strip().replace('\n', ' ').replace('  ', ' ') for col in df.columns]
        
        # Boşlukları ve gereksiz karakterleri temizle
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
        
        return df
    
    @staticmethod
    def prepare_analytics_data(df):
        """Analiz için veriyi hazırla"""
        # Yeni sütunlar ekle
        if 'MAT Q3 2022 USD MNF' in df.columns and 'MAT Q3 2023 USD MNF' in df.columns:
            df['Sales_Growth_22_23'] = ((df['MAT Q3 2023 USD MNF'] - df['MAT Q3 2022 USD MNF']) / 
                                       df['MAT Q3 2022 USD MNF'].replace(0, np.nan)) * 100
        
        if 'MAT Q3 2023 USD MNF' in df.columns and 'MAT Q3 2024 USD MNF' in df.columns:
            df['Sales_Growth_23_24'] = ((df['MAT Q3 2024 USD MNF'] - df['MAT Q3 2023 USD MNF']) / 
                                       df['MAT Q3 2023 USD MNF'].replace(0, np.nan)) * 100
        
        # Ortalama fiyat hesapla
        price_cols = [col for col in df.columns if 'Avg Price' in col]
        if price_cols:
            df['Avg_Price'] = df[price_cols].mean(axis=1, skipna=True)
        
        return df

# ================================================
# 3. ANALİTİK FONKSİYONLAR
# ================================================

class PharmaAnalytics:
    """Farma analitik fonksiyonları"""
    
    @staticmethod
    def calculate_market_metrics(df):
        """Pazar metriklerini hesapla"""
        metrics = {}
        
        # Toplam satışlar
        sales_cols = [col for col in df.columns if 'USD MNF' in col]
        for col in sales_cols:
            year = col.split('MAT Q3 ')[1].split(' USD')[0] if 'MAT Q3' in col else col
            metrics[f'Total_Sales_{year}'] = df[col].sum()
        
        # Pazar payı (Şirket bazlı)
        if 'Corporation' in df.columns and sales_cols:
            latest_sales_col = sales_cols[-1]  # En güncel yıl
            corp_sales = df.groupby('Corporation')[latest_sales_col].sum()
            total_sales = corp_sales.sum()
            
            if total_sales > 0:
                metrics['HHI_Index'] = ((corp_sales / total_sales * 100) ** 2).sum() / 10000
                metrics['Top3_Share'] = corp_sales.nlargest(3).sum() / total_sales * 100
                metrics['Top5_Share'] = corp_sales.nlargest(5).sum() / total_sales * 100
        
        # Molekül çeşitliliği
        if 'Molecule' in df.columns:
            metrics['Unique_Molecules'] = df['Molecule'].nunique()
            metrics['Molecule_Concentration'] = (
                df.groupby('Molecule')[sales_cols[-1]].sum().nlargest(10).sum() / 
                df[sales_cols[-1]].sum() * 100
            )
        
        # Coğrafi dağılım
        if 'Country' in df.columns:
            metrics['Country_Coverage'] = df['Country'].nunique()
        
        # Terapötik alan (varsa)
        if 'Specialty Product' in df.columns:
            specialty_share = df[df['Specialty Product'] == 'Yes'][sales_cols[-1]].sum() / df[sales_cols[-1]].sum() * 100
            metrics['Specialty_Share'] = specialty_share
        
        return metrics
    
    @staticmethod
    def analyze_growth_trends(df):
        """Büyüme trendlerini analiz et"""
        try:
            growth_data = []
            
            # Yıllık büyüme oranları
            sales_cols = sorted([col for col in df.columns if 'USD MNF' in col and 'MAT Q3' in col])
            
            for i in range(1, len(sales_cols)):
                prev_col = sales_cols[i-1]
                curr_col = sales_cols[i]
                
                if prev_col in df.columns and curr_col in df.columns:
                    year_growth = df.groupby('Country')[curr_col].sum() - df.groupby('Country')[prev_col].sum()
                    percent_growth = (year_growth / df.groupby('Country')[prev_col].sum().replace(0, np.nan)) * 100
                    
                    for country in percent_growth.index:
                        if not pd.isna(percent_growth[country]):
                            growth_data.append({
                                'Country': country,
                                'Period': f"{prev_col.split(' ')[2]}-{curr_col.split(' ')[2]}",
                                'Growth_Rate': percent_growth[country],
                                'Sales_Change': year_growth[country]
                            })
            
            return pd.DataFrame(growth_data)
        except:
            return pd.DataFrame()
    
    @staticmethod
    def perform_market_segmentation(df, n_clusters=4):
        """Pazar segmentasyonu analizi"""
        try:
            # Özellikler oluştur
            features = []
            
            if 'Country' in df.columns and 'Corporation' in df.columns:
                market_data = df.groupby(['Country', 'Corporation']).agg({
                    'MAT Q3 2024 USD MNF': 'sum',
                    'MAT Q3 2024 Unit Avg Price USD MNF': 'mean',
                    'MAT Q3 2024 Units': 'sum'
                }).dropna()
                
                if len(market_data) > n_clusters:
                    # Standardize et
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(market_data)
                    
                    # K-Means clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(features_scaled)
                    
                    market_data['Cluster'] = clusters
                    
                    return {
                        'data': market_data,
                        'clusters': clusters,
                        'inertia': kmeans.inertia_,
                        'cluster_centers': kmeans.cluster_centers_
                    }
            
            return None
        except Exception as e:
            st.warning(f"Segmentasyon hatası: {str(e)}")
            return None
    
    @staticmethod
    def detect_anomalies(df, contamination=0.1):
        """Anomali tespiti"""
        try:
            # Sayısal sütunları seç
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                # İlk 5 sayısal sütunu kullan
                features = df[numeric_cols[:5]].fillna(0)
                
                # Isolation Forest
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                anomalies = iso_forest.fit_predict(features)
                
                # Sonuçları ekle
                result_df = df.copy()
                result_df['Is_Anomaly'] = anomalies == -1
                result_df['Anomaly_Score'] = iso_forest.score_samples(features)
                
                return result_df[result_df['Is_Anomaly']]
            
            return pd.DataFrame()
        except:
            return pd.DataFrame()

# ================================================
# 4. GÖRSELLEŞTİRME FONKSİYONLARI
# ================================================

class VisualizationEngine:
    """Görselleştirme motoru"""
    
    @staticmethod
    def create_sales_overview(df):
        """Satış genel bakış grafikleri"""
        try:
            # Satış sütunlarını bul
            sales_cols = sorted([col for col in df.columns if 'USD MNF' in col and 'MAT Q3' in col])
            
            if len(sales_cols) >= 2:
                # Yıllık toplam satışlar
                yearly_sales = {}
                for col in sales_cols:
                    year = col.split('MAT Q3 ')[1].split(' USD')[0]
                    yearly_sales[year] = df[col].sum()
                
                # Line chart
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=list(yearly_sales.keys()),
                    y=list(yearly_sales.values()),
                    mode='lines+markers',
                    line=dict(color='#7aa2f7', width=3),
                    marker=dict(size=10),
                    name='Toplam Satış'
                ))
                
                fig1.update_layout(
                    title='Yıllık Satış Trendi',
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#c0caf5',
                    xaxis_title='Yıl',
                    yaxis_title='Satış (USD)'
                )
                
                # Bar chart - Son yılın ülke bazlı dağılımı
                if 'Country' in df.columns:
                    country_sales = df.groupby('Country')[sales_cols[-1]].sum().nlargest(15)
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=country_sales.values,
                        y=country_sales.index,
                        orientation='h',
                        marker_color='#bb9af7',
                        name='Ülke Satışları'
                    ))
                    
                    fig2.update_layout(
                        title=f'Top 15 Ülke - {sales_cols[-1].split(" ")[2]}',
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#c0caf5',
                        xaxis_title='Satış (USD)'
                    )
                
                return fig1, fig2 if 'fig2' in locals() else None
            
            return None, None
        except:
            return None, None
    
    @staticmethod
    def create_market_share_chart(df):
        """Pazar payı grafikleri"""
        try:
            sales_cols = [col for col in df.columns if 'USD MNF' in col and 'MAT Q3' in col]
            if not sales_cols or 'Corporation' not in df.columns:
                return None
            
            latest_col = sales_cols[-1]
            market_share = df.groupby('Corporation')[latest_col].sum().nlargest(10)
            total_sales = market_share.sum()
            
            if total_sales > 0:
                market_share_pct = (market_share / total_sales * 100).round(1)
                
                fig = go.Figure()
                
                # Pie chart
                fig.add_trace(go.Pie(
                    labels=market_share_pct.index,
                    values=market_share_pct.values,
                    hole=0.4,
                    marker_colors=px.colors.qualitative.Set3
                ))
                
                fig.update_layout(
                    title=f'Pazar Payı Dağılımı - Top 10 Şirket ({latest_col.split(" ")[2]})',
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#c0caf5',
                    showlegend=True
                )
                
                return fig
            
            return None
        except:
            return None
    
    @staticmethod
    def create_geographic_analysis(df):
        """Coğrafi analiz grafikleri"""
        try:
            if 'Country' in df.columns and 'Region' in df.columns:
                sales_cols = [col for col in df.columns if 'USD MNF' in col and 'MAT Q3' in col]
                if not sales_cols:
                    return None
                
                latest_col = sales_cols[-1]
                
                # Bölgesel dağılım
                regional_sales = df.groupby('Region')[latest_col].sum()
                
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(
                    x=regional_sales.index,
                    y=regional_sales.values,
                    marker_color='#9ece6a',
                    name='Bölgesel Satış'
                ))
                
                fig1.update_layout(
                    title='Bölgesel Satış Dağılımı',
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#c0caf5',
                    xaxis_title='Bölge',
                    yaxis_title='Satış (USD)',
                    xaxis_tickangle=-45
                )
                
                # Harita
                country_sales = df.groupby('Country')[latest_col].sum().reset_index()
                
                fig2 = px.choropleth(
                    country_sales,
                    locations='Country',
                    locationmode='country names',
                    color=latest_col,
                    hover_name='Country',
                    color_continuous_scale='Blues',
                    title='Coğrafi Satış Dağılımı'
                )
                
                fig2.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#c0caf5',
                    geo=dict(
                        bgcolor='rgba(0,0,0,0)',
                        lakecolor='rgba(0,0,0,0)',
                        landcolor='rgba(100,100,100,0.2)'
                    )
                )
                
                return fig1, fig2
            
            return None, None
        except:
            return None, None
    
    @staticmethod
    def create_product_analysis(df):
        """Ürün analiz grafikleri"""
        try:
            if 'Molecule' not in df.columns:
                return None
            
            sales_cols = [col for col in df.columns if 'USD MNF' in col and 'MAT Q3' in col]
            if not sales_cols:
                return None
            
            latest_col = sales_cols[-1]
            
            # Top moleküller
            top_molecules = df.groupby('Molecule')[latest_col].sum().nlargest(15)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_molecules.values,
                y=top_molecules.index,
                orientation='h',
                marker_color='#e0af68',
                name='Molekül Satışları'
            ))
            
            fig.update_layout(
                title='Top 15 Molekül - Satış Performansı',
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#c0caf5',
                xaxis_title='Satış (USD)'
            )
            
            return fig
        except:
            return None

# ================================================
# 5. ANA UYGULAMA
# ================================================

def main():
    # Header
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="pharma-title">💊 PHARMAINTELLIGENCE PRO</h1>
        <p style="font-size: 1.1rem; color: #a9b1d6; max-width: 800px; margin-bottom: 2rem;">
        Enterprise-level pharmaceutical market analytics platform for strategic decision making, 
        competitive intelligence, and market forecasting.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ================================================
    # SIDEBAR - KONTROL PANELİ
    # ================================================
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">🎛️ KONTROL PANELİ</h2>', unsafe_allow_html=True)
        
        # Session State
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'analytics' not in st.session_state:
            st.session_state.analytics = None
        
        # File Upload
        with st.expander("📁 VERİ YÜKLEME", expanded=True):
            uploaded_file = st.file_uploader(
                "Excel/CSV Dosyası Yükleyin",
                type=['xlsx', 'xls', 'csv'],
                help="100MB'a kadar dosya desteklenir"
            )
            
            if uploaded_file:
                col1, col2 = st.columns(2)
                with col1:
                    use_sample = st.checkbox("Örneklem Kullan", value=True)
                with col2:
                    sample_size = st.number_input("Örneklem Büyüklüğü", 
                                                min_value=1000,
                                                max_value=500000,
                                                value=50000,
                                                step=10000) if use_sample else None
                
                if st.button("🚀 Yükle & Analiz Et", type="primary", use_container_width=True):
                    with st.spinner("Veri işleniyor..."):
                        processor = DataProcessor()
                        
                        if use_sample and sample_size:
                            df = processor.load_large_excel(uploaded_file, sample_size=sample_size)
                        else:
                            df = processor.load_large_excel(uploaded_file)
                        
                        if df is not None:
                            df = processor.optimize_dataframe(df)
                            df = processor.prepare_analytics_data(df)
                            
                            st.session_state.df = df
                            st.session_state.analytics = PharmaAnalytics()
                            st.rerun()
        
        # Filtreler
        if st.session_state.df is not None:
            with st.expander("🔍 FİLTRELER", expanded=True):
                df = st.session_state.df
                
                # Ülke filtresi
                if 'Country' in df.columns:
                    countries = sorted(df['Country'].dropna().unique())
                    selected_countries = st.multiselect(
                        "Ülkeler",
                        options=countries,
                        default=countries[:5] if len(countries) > 5 else countries
                    )
                
                # Şirket filtresi
                if 'Corporation' in df.columns:
                    companies = sorted(df['Corporation'].dropna().unique())
                    selected_companies = st.multiselect(
                        "Şirketler",
                        options=companies,
                        default=companies[:5] if len(companies) > 5 else companies
                    )
                
                # Molekül filtresi
                if 'Molecule' in df.columns:
                    molecules = sorted(df['Molecule'].dropna().unique())
                    selected_molecules = st.multiselect(
                        "Moleküller",
                        options=molecules,
                        default=molecules[:10] if len(molecules) > 10 else molecules
                    )
        
        # Analiz Ayarları
        with st.expander("⚙️ ANALİZ AYARLARI", expanded=False):
            analysis_level = st.select_slider(
                "Analiz Seviyesi",
                options=['Temel', 'Standart', 'Gelişmiş'],
                value='Standart'
            )
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #565f89;">
        <strong>PharmaIntelligence Pro</strong><br>
        v3.0 | © 2024
        </div>
        """, unsafe_allow_html=True)
    
    # ================================================
    # ANA İÇERİK
    # ================================================
    
    if st.session_state.df is None:
        # Hoşgeldiniz ekranı
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem; background: #292e42; 
                     border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
                <div style="font-size: 4rem; margin-bottom: 1rem;">💼</div>
                <h2 style="color: #c0caf5;">PharmaIntelligence Pro'ya Hoşgeldiniz</h2>
                <p style="color: #a9b1d6; margin-bottom: 2rem;">
                İlaç pazarı verilerinizi yükleyin ve güçlü analitik özelliklerin kilidini açın.
                </p>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                    <div style="text-align: left; padding: 1rem; background: #414868; border-radius: 10px;">
                        <div style="font-size: 1.5rem; color: #7aa2f7;">📈</div>
                        <div style="font-weight: 600; color: #c0caf5;">Pazar Analizi</div>
                        <div style="font-size: 0.9rem; color: #a9b1d6;">Derin pazar içgörüleri</div>
                    </div>
                    <div style="text-align: left; padding: 1rem; background: #414868; border-radius: 10px;">
                        <div style="font-size: 1.5rem; color: #7aa2f7;">💰</div>
                        <div style="font-weight: 600; color: #c0caf5;">Fiyat Zekası</div>
                        <div style="font-size: 0.9rem; color: #a9b1d6;">Rekabetçi fiyatlandırma</div>
                    </div>
                    <div style="text-align: left; padding: 1rem; background: #414868; border-radius: 10px;">
                        <div style="font-size: 1.5rem; color: #7aa2f7;">🚀</div>
                        <div style="font-weight: 600; color: #c0caf5;">Büyüme Tahmini</div>
                        <div style="font-size: 0.9rem; color: #a9b1d6;">Öngörülebilir analitik</div>
                    </div>
                    <div style="text-align: left; padding: 1rem; background: #414868; border-radius: 10px;">
                        <div style="font-size: 1.5rem; color: #7aa2f7;">🏆</div>
                        <div style="font-weight: 600; color: #c0caf5;">Rekabet Analizi</div>
                        <div style="font-size: 0.9rem; color: #a9b1d6;">Rakiplerinizi analiz edin</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return
    
    # ================================================
    # VERİ ANALİZİ BÖLÜMÜ
    # ================================================
    
    df = st.session_state.df
    analytics = st.session_state.analytics
    
    # Quick Stats
    total_sales_2024 = df['MAT Q3 2024 USD MNF'].sum() if 'MAT Q3 2024 USD MNF' in df.columns else 0
    total_molecules = df['Molecule'].nunique() if 'Molecule' in df.columns else 0
    total_countries = df['Country'].nunique() if 'Country' in df.columns else 0
    total_companies = df['Corporation'].nunique() if 'Corporation' in df.columns else 0
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #7aa2f7, #bb9af7); 
                color: white; padding: 1.5rem; border-radius: 15px; 
                margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.9rem; opacity: 0.9;">TOPLAM PAZAR DEĞERİ (2024)</div>
                <div style="font-size: 2.2rem; font-weight: 800;">${total_sales_2024/1e6:.1f}M</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.9;">MOLEKÜL</div>
                <div style="font-size: 1.8rem; font-weight: 700;">{total_molecules}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.9;">ÜLKE</div>
                <div style="font-size: 1.8rem; font-weight: 700;">{total_countries}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.9;">ŞİRKET</div>
                <div style="font-size: 1.8rem; font-weight: 700;">{total_companies}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Ana Tablar
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 GENEL BAKIŞ",
        "📈 PAZAR ANALİZİ",
        "💰 FİYAT ANALİZİ",
        "🏆 REKABET ANALİZİ",
        "⚙️ GELİŞMİŞ ANALİTİK"
    ])
    
    # TAB 1: GENEL BAKIŞ
    with tab1:
        st.markdown('<h2 class="section-title">Genel Bakış</h2>', unsafe_allow_html=True)
        
        # Metrik Kartları
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = df['MAT Q3 2024 Unit Avg Price USD MNF'].mean() if 'MAT Q3 2024 Unit Avg Price USD MNF' in df.columns else 0
            st.markdown(f"""
            <div class="metric-card premium">
                <div class="metric-label">ORTALAMA FİYAT (2024)</div>
                <div class="metric-value">${avg_price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sales_growth = df['Sales_Growth_23_24'].mean() if 'Sales_Growth_23_24' in df.columns else 0
            growth_color = "success" if sales_growth > 0 else "danger"
            st.markdown(f"""
            <div class="metric-card {growth_color}">
                <div class="metric-label">BÜYÜME ORANI (23-24)</div>
                <div class="metric-value">{sales_growth:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            market_metrics = analytics.calculate_market_metrics(df)
            hhi = market_metrics.get('HHI_Index', 0)
            hhi_status = "danger" if hhi > 2500 else "warning" if hhi > 1500 else "success"
            st.markdown(f"""
            <div class="metric-card {hhi_status}">
                <div class="metric-label">REKABET YOĞUNLUĞU (HHI)</div>
                <div class="metric-value">{hhi:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            top3_share = market_metrics.get('Top3_Share', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">TOP 3 PAYI</div>
                <div class="metric-value">{top3_share:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Ana Grafikler
        st.markdown('<h3 class="subsection-title">Satış Trendleri</h3>', unsafe_allow_html=True)
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            fig1, fig2 = VisualizationEngine.create_sales_overview(df)
            if fig1:
                st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': True})
        
        with chart_col2:
            if fig2:
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': True})
        
        # İçgörüler
        st.markdown('<h3 class="subsection-title">Önemli İçgörüler</h3>', unsafe_allow_html=True)
        
        insight_cols = st.columns(2)
        
        with insight_cols[0]:
            # En çok satan ülke
            if 'Country' in df.columns and 'MAT Q3 2024 USD MNF' in df.columns:
                top_country = df.groupby('Country')['MAT Q3 2024 USD MNF'].sum().idxmax()
                country_sales = df.groupby('Country')['MAT Q3 2024 USD MNF'].sum().max()
                
                st.markdown(f"""
                <div class="insight-card success">
                    <div class="insight-title">🏆 Lider Ülke</div>
                    <div class="insight-content">
                        <strong>{top_country}</strong> en yüksek satışa sahip ülke.
                        <br>Toplam satış: <strong>${country_sales/1e6:.1f}M</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # En çok satan molekül
            if 'Molecule' in df.columns:
                top_molecule = df.groupby('Molecule')['MAT Q3 2024 USD MNF'].sum().idxmax()
                molecule_sales = df.groupby('Molecule')['MAT Q3 2024 USD MNF'].sum().max()
                
                st.markdown(f"""
                <div class="insight-card info">
                    <div class="insight-title">🧪 En Çok Satan Molekül</div>
                    <div class="insight-content">
                        <strong>{top_molecule}</strong> en yüksek satışa sahip molekül.
                        <br>Toplam satış: <strong>${molecule_sales/1e6:.1f}M</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with insight_cols[1]:
            # En büyük şirket
            if 'Corporation' in df.columns:
                top_company = df.groupby('Corporation')['MAT Q3 2024 USD MNF'].sum().idxmax()
                company_sales = df.groupby('Corporation')['MAT Q3 2024 USD MNF'].sum().max()
                market_share = (company_sales / df['MAT Q3 2024 USD MNF'].sum()) * 100
                
                st.markdown(f"""
                <div class="insight-card warning">
                    <div class="insight-title">🏢 Lider Şirket</div>
                    <div class="insight-content">
                        <strong>{top_company}</strong> pazar lideri.
                        <br>Pazar payı: <strong>{market_share:.1f}%</strong>
                        <br>Satış: <strong>${company_sales/1e6:.1f}M</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Specialty vs Non-specialty
            if 'Specialty Product' in df.columns:
                specialty_sales = df[df['Specialty Product'] == 'Yes']['MAT Q3 2024 USD MNF'].sum()
                total_sales = df['MAT Q3 2024 USD MNF'].sum()
                specialty_share = (specialty_sales / total_sales) * 100 if total_sales > 0 else 0
                
                st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-title">💊 Specialty Payı</div>
                    <div class="insight-content">
                        Specialty ürünler toplam pazarın <strong>{specialty_share:.1f}%</strong>'ini oluşturuyor.
                        <br>Toplam: <strong>${specialty_sales/1e6:.1f}M</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # TAB 2: PAZAR ANALİZİ
    with tab2:
        st.markdown('<h2 class="section-title">Pazar Analizi</h2>', unsafe_allow_html=True)
        
        # Coğrafi Analiz
        st.markdown('<h3 class="subsection-title">Coğrafi Dağılım</h3>', unsafe_allow_html=True)
        
        geo_col1, geo_col2 = st.columns(2)
        
        with geo_col1:
            fig1, fig2 = VisualizationEngine.create_geographic_analysis(df)
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
        
        with geo_col2:
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        
        # Büyüme Analizi
        st.markdown('<h3 class="subsection-title">Büyüme Trendleri</h3>', unsafe_allow_html=True)
        
        growth_data = analytics.analyze_growth_trends(df)
        if not growth_data.empty:
            growth_col1, growth_col2 = st.columns(2)
            
            with growth_col1:
                # En hızlı büyüyen ülkeler
                fastest_growing = growth_data.nlargest(10, 'Growth_Rate')
                fig = px.bar(
                    fastest_growing,
                    x='Growth_Rate',
                    y='Country',
                    orientation='h',
                    title='En Hızlı Büyüyen 10 Ülke',
                    color='Growth_Rate',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#c0caf5'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with growth_col2:
                # En çok satış artışı olan ülkeler
                highest_increase = growth_data.nlargest(10, 'Sales_Change')
                fig = px.bar(
                    highest_increase,
                    x='Sales_Change',
                    y='Country',
                    orientation='h',
                    title='En Çok Satış Artışı Olan 10 Ülke',
                    color='Sales_Change',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#c0caf5'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Pazar Segmentasyonu
        st.markdown('<h3 class="subsection-title">Pazar Segmentasyonu</h3>', unsafe_allow_html=True)
        
        if st.button("🔍 Segmentasyon Analizi Yap", type="primary"):
            with st.spinner("Pazar segmentasyonu analiz ediliyor..."):
                segmentation_results = analytics.perform_market_segmentation(df)
                
                if segmentation_results:
                    seg_col1, seg_col2 = st.columns(2)
                    
                    with seg_col1:
                        st.markdown("**Segment Profilleri**")
                        st.dataframe(
                            segmentation_results['data'].groupby('Cluster').mean(),
                            use_container_width=True
                        )
                    
                    with seg_col2:
                        st.metric("Segment Sayısı", len(segmentation_results['data']['Cluster'].unique()))
                        st.metric("Inertia", f"{segmentation_results['inertia']:,.0f}")
                    
                    # Segmentasyon görselleştirme
                    fig = px.scatter(
                        segmentation_results['data'].reset_index(),
                        x='MAT Q3 2024 USD MNF',
                        y='MAT Q3 2024 Unit Avg Price USD MNF',
                        color='Cluster',
                        size='MAT Q3 2024 Units',
                        hover_data=['Country', 'Corporation'],
                        title='Pazar Segmentasyonu',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#c0caf5'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: FİYAT ANALİZİ
    with tab3:
        st.markdown('<h2 class="section-title">Fiyat Analizi</h2>', unsafe_allow_html=True)
        
        price_cols = [col for col in df.columns if 'Avg Price' in col]
        
        if price_cols:
            # Fiyat Dağılımı
            st.markdown('<h3 class="subsection-title">Fiyat Dağılımı</h3>', unsafe_allow_html=True)
            
            price_col1, price_col2 = st.columns(2)
            
            with price_col1:
                latest_price_col = price_cols[-1]
                fig = px.histogram(
                    df[latest_price_col].dropna(),
                    nbins=50,
                    title=f'Fiyat Dağılımı ({latest_price_col})',
                    color_discrete_sequence=['#7aa2f7']
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0)',
                    font_color='#c0caf5',
                    xaxis_title='Fiyat (USD)',
                    yaxis_title='Frekans'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with price_col2:
                # Ülke bazlı ortalama fiyatlar
                if 'Country' in df.columns:
                    country_prices = df.groupby('Country')[latest_price_col].mean().nlargest(15)
                    
                    fig = px.bar(
                        country_prices,
                        orientation='h',
                        title='Ülke Bazlı Ortalama Fiyatlar (Top 15)',
                        color=country_prices.values,
                        color_continuous_scale='RdYlBu_r'
                    )
                    fig.update_layout(
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#c0caf5',
                        xaxis_title='Ortalama Fiyat (USD)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Fiyat Trendleri
            st.markdown('<h3 class="subsection-title">Fiyat Trendleri</h3>', unsafe_allow_html=True)
            
            if len(price_cols) >= 2:
                price_trend_data = []
                for col in price_cols:
                    year = col.split('MAT Q3 ')[1].split(' Unit')[0] if 'MAT Q3' in col else col
                    price_trend_data.append({
                        'Year': year,
                        'Avg_Price': df[col].mean()
                    })
                
                price_trend_df = pd.DataFrame(price_trend_data)
                
                fig = px.line(
                    price_trend_df,
                    x='Year',
                    y='Avg_Price',
                    markers=True,
                    title='Yıllık Ortalama Fiyat Trendi',
                    line_shape='spline'
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#c0caf5',
                    xaxis_title='Yıl',
                    yaxis_title='Ortalama Fiyat (USD)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Fiyat-Hacim İlişkisi
        st.markdown('<h3 class="subsection-title">Fiyat-Hacim İlişkisi</h3>', unsafe_allow_html=True)
        
        if 'MAT Q3 2024 Unit Avg Price USD MNF' in df.columns and 'MAT Q3 2024 Units' in df.columns:
            sample_df = df.sample(min(1000, len(df)))
            
            fig = px.scatter(
                sample_df,
                x='MAT Q3 2024 Unit Avg Price USD MNF',
                y='MAT Q3 2024 Units',
                size='MAT Q3 2024 USD MNF',
                color='Country' if 'Country' in df.columns else None,
                hover_name='Molecule' if 'Molecule' in df.columns else None,
                title='Fiyat-Hacim İlişkisi',
                log_x=True,
                log_y=True
            )
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#c0caf5'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: REKABET ANALİZİ
    with tab4:
        st.markdown('<h2 class="section-title">Rekabet Analizi</h2>', unsafe_allow_html=True)
        
        # Pazar Payı Analizi
        st.markdown('<h3 class="subsection-title">Pazar Payı Dağılımı</h3>', unsafe_allow_html=True)
        
        market_share_fig = VisualizationEngine.create_market_share_chart(df)
        if market_share_fig:
            st.plotly_chart(market_share_fig, use_container_width=True)
        
        # Şirket Performansı
        st.markdown('<h3 class="subsection-title">Şirket Performans Karşılaştırması</h3>', unsafe_allow_html=True)
        
        if 'Corporation' in df.columns and 'MAT Q3 2024 USD MNF' in df.columns:
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                # Top şirketler
                top_companies = df.groupby('Corporation')['MAT Q3 2024 USD MNF'].sum().nlargest(15)
                
                fig = px.bar(
                    top_companies,
                    orientation='h',
                    title='Top 15 Şirket - Satış Performansı',
                    color=top_companies.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#c0caf5',
                    xaxis_title='Satış (USD)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with comp_col2:
                # Rekabet yoğunluğu metrikleri
                market_metrics = analytics.calculate_market_metrics(df)
                
                metrics_data = [
                    ("Toplam Şirket", market_metrics.get('unique_companies', 0)),
                    ("HHI İndeksi", f"{market_metrics.get('HHI_Index', 0):,.0f}"),
                    ("Top 3 Payı", f"{market_metrics.get('Top3_Share', 0):.1f}%"),
                    ("Top 5 Payı", f"{market_metrics.get('Top5_Share', 0):.1f}%"),
                    ("Lider Şirket Payı", f"{(top_companies.iloc[0] / top_companies.sum() * 100):.1f}%")
                ]
                
                for metric_name, metric_value in metrics_data:
                    st.metric(metric_name, metric_value)
        
        # Ürün Performansı
        st.markdown('<h3 class="subsection-title">Ürün Performans Analizi</h3>', unsafe_allow_html=True)
        
        product_fig = VisualizationEngine.create_product_analysis(df)
        if product_fig:
            st.plotly_chart(product_fig, use_container_width=True)
    
    # TAB 5: GELİŞMİŞ ANALİTİK
    with tab5:
        st.markdown('<h2 class="section-title">Gelişmiş Analitik</h2>', unsafe_allow_html=True)
        
        # Anomali Tespiti
        st.markdown('<h3 class="subsection-title">Anomali Tespiti</h3>', unsafe_allow_html=True)
        
        if st.button("🔍 Anomalileri Tespit Et", type="primary"):
            with st.spinner("Anomali tespiti yapılıyor..."):
                anomalies = analytics.detect_anomalies(df)
                
                if not anomalies.empty:
                    st.markdown(f"**{len(anomalies)} anomali tespit edildi**")
                    
                    anomaly_cols = st.columns(2)
                    
                    with anomaly_cols[0]:
                        st.dataframe(
                            anomalies.head(10),
                            use_container_width=True
                        )
                    
                    with anomaly_cols[1]:
                        # Anomali dağılımı
                        if 'Anomaly_Score' in anomalies.columns:
                            fig = px.histogram(
                                anomalies,
                                x='Anomaly_Score',
                                nbins=30,
                                title='Anomali Skor Dağılımı',
                                color_discrete_sequence=['#f7768e']
                            )
                            fig.update_layout(
                                height=400,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='#c0caf5'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Anomali tespit edilmedi.")
        
        # Veri Kalitesi Analizi
        st.markdown('<h3 class="subsection-title">Veri Kalitesi Analizi</h3>', unsafe_allow_html=True)
        
        quality_cols = st.columns(4)
        
        with quality_cols[0]:
            total_rows = len(df)
            st.metric("Toplam Satır", f"{total_rows:,}")
        
        with quality_cols[1]:
            missing_values = df.isnull().sum().sum()
            missing_percentage = (missing_values / (total_rows * len(df.columns))) * 100
            st.metric("Eksik Veri", f"{missing_percentage:.1f}%")
        
        with quality_cols[2]:
            duplicate_rows = df.duplicated().sum()
            duplicate_percentage = (duplicate_rows / total_rows) * 100
            st.metric("Kopya Satır", f"{duplicate_percentage:.1f}%")
        
        with quality_cols[3]:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Sayısal Sütun", numeric_cols)
        
        # Korelasyon Analizi
        st.markdown('<h3 class="subsection-title">Korelasyon Analizi</h3>', unsafe_allow_html=True)
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu',
                title='Korelasyon Matrisi'
            )
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#c0caf5'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Veri Önizleme
        st.markdown('<h3 class="subsection-title">Veri Önizleme</h3>', unsafe_allow_html=True)
        
        preview_col1, preview_col2 = st.columns([1, 3])
        
        with preview_col1:
            rows_to_show = st.slider("Gösterilecek satır", 10, 1000, 100)
        
        with preview_col2:
            st.dataframe(
                df.head(rows_to_show),
                use_container_width=True,
                height=400
            )
    
    # ================================================
    # RAPORLAMA VE İNDİRME
    # ================================================
    
    st.markdown("---")
    st.markdown('<h2 class="section-title">Raporlama</h2>', unsafe_allow_html=True)
    
    report_cols = st.columns(3)
    
    with report_cols[0]:
        if st.button("📊 Excel Raporu Oluştur", use_container_width=True):
            with st.spinner("Excel raporu oluşturuluyor..."):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Raw_Data', index=False)
                    
                    # Özet metrikler
                    market_metrics = analytics.calculate_market_metrics(df)
                    metrics_df = pd.DataFrame(list(market_metrics.items()), columns=['Metric', 'Value'])
                    metrics_df.to_excel(writer, sheet_name='Summary_Metrics', index=False)
                    
                    # Pazar payı
                    if 'Corporation' in df.columns:
                        market_share = df.groupby('Corporation')['MAT Q3 2024 USD MNF'].sum().nlargest(50)
                        market_share_df = market_share.reset_index()
                        market_share_df.columns = ['Corporation', 'Market_Share']
                        market_share_df['Percentage'] = (market_share_df['Market_Share'] / market_share_df['Market_Share'].sum()) * 100
                        market_share_df.to_excel(writer, sheet_name='Market_Share', index=False)
                    
                    writer.save()
                
                st.download_button(
                    label="⬇️ Excel İndir",
                    data=output.getvalue(),
                    file_name=f"pharma_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    with report_cols[1]:
        if st.button("📈 JSON Raporu Oluştur", use_container_width=True):
            with st.spinner("JSON raporu oluşturuluyor..."):
                market_metrics = analytics.calculate_market_metrics(df)
                
                report_data = {
                    'generated_date': datetime.now().isoformat(),
                    'data_summary': {
                        'total_rows': len(df),
                        'total_columns': len(df.columns),
                        'data_columns': list(df.columns)
                    },
                    'market_metrics': market_metrics,
                    'top_performers': {
                        'top_countries': df.groupby('Country')['MAT Q3 2024 USD MNF'].sum().nlargest(5).to_dict(),
                        'top_companies': df.groupby('Corporation')['MAT Q3 2024 USD MNF'].sum().nlargest(5).to_dict(),
                        'top_molecules': df.groupby('Molecule')['MAT Q3 2024 USD MNF'].sum().nlargest(5).to_dict()
                    }
                }
                
                st.download_button(
                    label="⬇️ JSON İndir",
                    data=json.dumps(report_data, indent=2, ensure_ascii=False),
                    file_name=f"pharma_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    with report_cols[2]:
        if st.button("🔄 Analizi Sıfırla", use_container_width=True):
            st.session_state.df = None
            st.session_state.analytics = None
            st.rerun()
    
    # ================================================
    # FOOTER
    # ================================================
    
    st.markdown("---")
    
    footer_cols = st.columns(3)
    
    with footer_cols[0]:
        st.markdown("""
        **📞 Destek**
        - support@pharmaintelligence.com
        - +90 212 123 4567
        - 7/24 Enterprise Destek
        """)
    
    with footer_cols[1]:
        st.markdown("""
        **🔒 Güvenlik**
        - GDPR Uyumlu
        - ISO 27001 Sertifikalı
        - Uçtan Uca Şifreleme
        """)
    
    with footer_cols[2]:
        st.markdown("""
        **🔄 Güncellemeler**
        - Son Güncelleme: {}
        - Sonraki Bakım: {}
        - Versiyon: 3.0
        """.format(
            datetime.now().strftime("%d/%m/%Y"),
            (datetime.now() + timedelta(days=30)).strftime("%d/%m/%Y")
        ))
    
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 1.5rem; 
                background: linear-gradient(135deg, #7aa2f7, #bb9af7); 
                color: white; border-radius: 15px;">
        <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
            PharmaIntelligence Pro Enterprise Platform
        </div>
        <div style="font-size: 0.9rem; opacity: 0.9;">
            Gelişmiş Analitik • Gerçek Zamanlı İçgörüler • Öngörülebilir Zeka
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================================
# 6. UYGULAMA BAŞLATMA
# ================================================

if __name__ == "__main__":
    gc.collect()
    main()
