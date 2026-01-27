# app.py - Pharma Analytics Dashboard - SONUÇ ODAKLI
"""
PHARMA MARKET ANALYTICS DASHBOARD
İlaç Pazarı Analiz Platformu - Sadece Analiz, Keşif Değil
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# KONFİGÜRASYON
# ============================================================================

st.set_page_config(
    page_title="Pharma Market Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #DC2626 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F2937;
        border-left: 5px solid #DC2626;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #DC2626;
        height: 100%;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1F2937;
        line-height: 1;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    .kpi-trend {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    .trend-up { color: #10B981; }
    .trend-down { color: #EF4444; }
    .trend-neutral { color: #6B7280; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ANALİZ FONKSİYONLARI - SONUÇ ODAKLI
# ============================================================================

def calculate_market_share(df):
    """Pazar payı analizi"""
    if df.empty:
        return pd.DataFrame()
    
    # SATIŞ DEĞERİNE GÖRE PAZAR PAYI
    if 'MAT Q3 2024 USD MNF' in df.columns:
        market_total = df['MAT Q3 2024 USD MNF'].sum()
        df['MARKET_SHARE_%'] = (df['MAT Q3 2024 USD MNF'] / market_total * 100).round(2)
    
    # STANDART ÜNİTE BAZINDA PAZAR PAYI
    if 'MAT Q3 2024 STANDARD UNITS' in df.columns:
        unit_total = df['MAT Q3 2024 STANDARD UNITS'].sum()
        df['UNIT_SHARE_%'] = (df['MAT Q3 2024 STANDARD UNITS'] / unit_total * 100).round(2)
    
    return df

def calculate_growth_rates(df):
    """Büyüme oranları analizi"""
    if df.empty:
        return df
    
    growth_data = []
    
    for idx, row in df.iterrows():
        try:
            # 2023-2024 büyüme oranı
            if ('MAT Q3 2023 USD MNF' in df.columns and 
                'MAT Q3 2024 USD MNF' in df.columns and
                pd.notna(row['MAT Q3 2023 USD MNF']) and 
                row['MAT Q3 2023 USD MNF'] != 0):
                
                sales_growth = ((row['MAT Q3 2024 USD MNF'] - row['MAT Q3 2023 USD MNF']) / 
                               row['MAT Q3 2023 USD MNF'] * 100)
            else:
                sales_growth = np.nan
            
            # Volume büyüme
            if ('MAT Q3 2023 STANDARD UNITS' in df.columns and 
                'MAT Q3 2024 STANDARD UNITS' in df.columns and
                pd.notna(row['MAT Q3 2023 STANDARD UNITS']) and 
                row['MAT Q3 2023 STANDARD UNITS'] != 0):
                
                volume_growth = ((row['MAT Q3 2024 STANDARD UNITS'] - row['MAT Q3 2023 STANDARD UNITS']) / 
                                row['MAT Q3 2023 STANDARD UNITS'] * 100)
            else:
                volume_growth = np.nan
            
            # Fiyat değişimi
            if ('MAT Q3 2023 SU AVG PRICE USD MNF' in df.columns and 
                'MAT Q3 2024 SU AVG PRICE USD MNF' in df.columns and
                pd.notna(row['MAT Q3 2023 SU AVG PRICE USD MNF']) and 
                row['MAT Q3 2023 SU AVG PRICE USD MNF'] != 0):
                
                price_change = ((row['MAT Q3 2024 SU AVG PRICE USD MNF'] - row['MAT Q3 2023 SU AVG PRICE USD MNF']) / 
                               row['MAT Q3 2023 SU AVG PRICE USD MNF'] * 100)
            else:
                price_change = np.nan
            
            growth_data.append({
                'PRODUCT': row.get('INTERNATIONAL PRODUCT', ''),
                'MOLECULE': row.get('MOLECULE', ''),
                'MANUFACTURER': row.get('MANUFACTURER', ''),
                'SALES_GROWTH_%': round(sales_growth, 1) if not pd.isna(sales_growth) else np.nan,
                'VOLUME_GROWTH_%': round(volume_growth, 1) if not pd.isna(volume_growth) else np.nan,
                'PRICE_CHANGE_%': round(price_change, 1) if not pd.isna(price_change) else np.nan,
                'SALES_2024': row.get('MAT Q3 2024 USD MNF', 0),
                'SALES_2023': row.get('MAT Q3 2023 USD MNF', 0),
                'VOLUME_2024': row.get('MAT Q3 2024 STANDARD UNITS', 0)
            })
            
        except:
            continue
    
    return pd.DataFrame(growth_data)

def analyze_molecule_performance(df):
    """Molekül bazlı performans analizi"""
    if df.empty or 'MOLECULE' not in df.columns:
        return pd.DataFrame()
    
    molecule_stats = []
    
    for molecule in df['MOLECULE'].unique():
        molecule_df = df[df['MOLECULE'] == molecule]
        
        # Toplam satış
        total_sales_2024 = molecule_df['MAT Q3 2024 USD MNF'].sum() if 'MAT Q3 2024 USD MNF' in df.columns else 0
        total_sales_2023 = molecule_df['MAT Q3 2023 USD MNF'].sum() if 'MAT Q3 2023 USD MNF' in df.columns else 0
        
        # Büyüme
        if total_sales_2023 > 0:
            growth_rate = ((total_sales_2024 - total_sales_2023) / total_sales_2023 * 100)
        else:
            growth_rate = 0 if total_sales_2024 > 0 else np.nan
        
        # Üretici sayısı
        manufacturers = molecule_df['MANUFACTURER'].nunique()
        
        # Ürün sayısı
        products = molecule_df['INTERNATIONAL PRODUCT'].nunique()
        
        # Ortalama fiyat
        avg_price = molecule_df['MAT Q3 2024 SU AVG PRICE USD MNF'].mean() if 'MAT Q3 2024 SU AVG PRICE USD MNF' in df.columns else 0
        
        molecule_stats.append({
            'MOLECULE': molecule,
            'TOTAL_SALES_2024': total_sales_2024,
            'TOTAL_SALES_2023': total_sales_2023,
            'GROWTH_%': round(growth_rate, 1) if not pd.isna(growth_rate) else 0,
            'MANUFACTURERS': manufacturers,
            'PRODUCTS': products,
            'AVG_PRICE': round(avg_price, 2),
            'MARKET_SHARE_%': 0  # Sonradan hesaplanacak
        })
    
    result_df = pd.DataFrame(molecule_stats)
    
    # Pazar payı hesapla
    if 'TOTAL_SALES_2024' in result_df.columns:
        total_market = result_df['TOTAL_SALES_2024'].sum()
        result_df['MARKET_SHARE_%'] = (result_df['TOTAL_SALES_2024'] / total_market * 100).round(2)
    
    return result_df.sort_values('TOTAL_SALES_2024', ascending=False)

def analyze_manufacturer_performance(df):
    """Üretici bazlı performans analizi"""
    if df.empty or 'MANUFACTURER' not in df.columns:
        return pd.DataFrame()
    
    manufacturer_stats = []
    
    for manufacturer in df['MANUFACTURER'].unique():
        man_df = df[df['MANUFACTURER'] == manufacturer]
        
        # Satış verileri
        total_sales_2024 = man_df['MAT Q3 2024 USD MNF'].sum() if 'MAT Q3 2024 USD MNF' in df.columns else 0
        total_sales_2023 = man_df['MAT Q3 2023 USD MNF'].sum() if 'MAT Q3 2023 USD MNF' in df.columns else 0
        
        # Büyüme
        if total_sales_2023 > 0:
            growth_rate = ((total_sales_2024 - total_sales_2023) / total_sales_2023 * 100)
        else:
            growth_rate = 0 if total_sales_2024 > 0 else np.nan
        
        # Portföy analizi
        molecules = man_df['MOLECULE'].nunique()
        products = man_df['INTERNATIONAL PRODUCT'].nunique()
        
        # Ortalama fiyat
        avg_price = man_df['MAT Q3 2024 SU AVG PRICE USD MNF'].mean() if 'MAT Q3 2024 SU AVG PRICE USD MNF' in df.columns else 0
        
        # Specialty vs Non-specialty
        specialty_sales = 0
        if 'SPECIALTY PRODUCT' in df.columns:
            specialty_sales = man_df[man_df['SPECIALTY PRODUCT'].str.contains('SPECIALTY', na=False)]['MAT Q3 2024 USD MNF'].sum()
        
        manufacturer_stats.append({
            'MANUFACTURER': manufacturer,
            'TOTAL_SALES_2024': total_sales_2024,
            'TOTAL_SALES_2023': total_sales_2023,
            'GROWTH_%': round(growth_rate, 1) if not pd.isna(growth_rate) else 0,
            'MOLECULES': molecules,
            'PRODUCTS': products,
            'AVG_PRICE': round(avg_price, 2),
            'SPECIALTY_SALES': specialty_sales,
            'SPECIALTY_SHARE_%': (specialty_sales / total_sales_2024 * 100).round(2) if total_sales_2024 > 0 else 0
        })
    
    result_df = pd.DataFrame(manufacturer_stats)
    
    # Pazar payı hesapla
    if 'TOTAL_SALES_2024' in result_df.columns:
        total_market = result_df['TOTAL_SALES_2024'].sum()
        result_df['MARKET_SHARE_%'] = (result_df['TOTAL_SALES_2024'] / total_market * 100).round(2)
    
    return result_df.sort_values('TOTAL_SALES_2024', ascending=False)

def analyze_price_segments(df):
    """Fiyat segmentasyonu analizi"""
    if df.empty or 'MAT Q3 2024 SU AVG PRICE USD MNF' not in df.columns:
        return pd.DataFrame()
    
    # Fiyat aralıklarına göre segmentasyon
    price_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
    price_labels = ['<10$', '10-50$', '50-100$', '100-500$', '500-1000$', '>1000$']
    
    df['PRICE_SEGMENT'] = pd.cut(df['MAT Q3 2024 SU AVG PRICE USD MNF'], 
                                 bins=price_bins, labels=price_labels, right=False)
    
    segment_stats = []
    for segment in price_labels:
        segment_df = df[df['PRICE_SEGMENT'] == segment]
        
        if len(segment_df) > 0:
            total_sales = segment_df['MAT Q3 2024 USD MNF'].sum()
            total_volume = segment_df['MAT Q3 2024 STANDARD UNITS'].sum() if 'MAT Q3 2024 STANDARD UNITS' in df.columns else 0
            product_count = len(segment_df)
            avg_price = segment_df['MAT Q3 2024 SU AVG PRICE USD MNF'].mean()
            
            segment_stats.append({
                'PRICE_SEGMENT': segment,
                'TOTAL_SALES': total_sales,
                'TOTAL_VOLUME': total_volume,
                'PRODUCT_COUNT': product_count,
                'AVG_PRICE': round(avg_price, 2),
                'SALES_PER_PRODUCT': round(total_sales / product_count, 2) if product_count > 0 else 0
            })
    
    return pd.DataFrame(segment_stats)

def analyze_specialty_vs_generic(df):
    """Specialty vs Non-specialty analizi"""
    if df.empty or 'SPECIALTY PRODUCT' not in df.columns:
        return pd.DataFrame()
    
    df['PRODUCT_TYPE'] = df['SPECIALTY PRODUCT'].apply(
        lambda x: 'SPECIALTY' if isinstance(x, str) and 'SPECIALTY' in x.upper() else 'NON-SPECIALTY'
    )
    
    type_stats = []
    
    for p_type in ['SPECIALTY', 'NON-SPECIALTY']:
        type_df = df[df['PRODUCT_TYPE'] == p_type]
        
        total_sales_2024 = type_df['MAT Q3 2024 USD MNF'].sum()
        total_sales_2023 = type_df['MAT Q3 2023 USD MNF'].sum() if 'MAT Q3 2023 USD MNF' in df.columns else 0
        
        if total_sales_2023 > 0:
            growth = ((total_sales_2024 - total_sales_2023) / total_sales_2023 * 100)
        else:
            growth = 0
        
        product_count = len(type_df)
        avg_price = type_df['MAT Q3 2024 SU AVG PRICE USD MNF'].mean() if 'MAT Q3 2024 SU AVG PRICE USD MNF' in df.columns else 0
        
        type_stats.append({
            'PRODUCT_TYPE': p_type,
            'TOTAL_SALES_2024': total_sales_2024,
            'MARKET_SHARE_%': (total_sales_2024 / df['MAT Q3 2024 USD MNF'].sum() * 100).round(2) if df['MAT Q3 2024 USD MNF'].sum() > 0 else 0,
            'GROWTH_%': round(growth, 1),
            'PRODUCT_COUNT': product_count,
            'AVG_PRICE': round(avg_price, 2),
            'SALES_PER_PRODUCT': round(total_sales_2024 / product_count, 2) if product_count > 0 else 0
        })
    
    return pd.DataFrame(type_stats)

def identify_top_performers(df, n=10):
    """En iyi performans gösteren ürünleri belirle"""
    if df.empty:
        return pd.DataFrame()
    
    # Satışa göre sırala
    if 'MAT Q3 2024 USD MNF' in df.columns:
        top_sales = df.nlargest(n, 'MAT Q3 2024 USD MNF')[['INTERNATIONAL PRODUCT', 'MOLECULE', 
                                                          'MANUFACTURER', 'MAT Q3 2024 USD MNF',
                                                          'MAT Q3 2023 USD MNF']].copy()
        
        # Büyüme oranı ekle
        top_sales['GROWTH_%'] = ((top_sales['MAT Q3 2024 USD MNF'] - top_sales['MAT Q3 2023 USD MNF']) / 
                                 top_sales['MAT Q3 2023 USD MNF'] * 100).round(1)
        
        return top_sales
    
    return pd.DataFrame()

def identify_emerging_products(df, min_sales=100000, min_growth=20):
    """Yükselen ürünleri belirle (yüksek büyüme + belirli satış hacmi)"""
    if df.empty:
        return pd.DataFrame()
    
    required_cols = ['INTERNATIONAL PRODUCT', 'MOLECULE', 'MANUFACTURER',
                     'MAT Q3 2024 USD MNF', 'MAT Q3 2023 USD MNF']
    
    if all(col in df.columns for col in required_cols):
        # Satışı min_sales üzerinde ve büyümesi min_growth üzerinde olanlar
        emerging = df[
            (df['MAT Q3 2024 USD MNF'] >= min_sales) &
            (df['MAT Q3 2023 USD MNF'] > 0)
        ].copy()
        
        if not emerging.empty:
            emerging['GROWTH_%'] = ((emerging['MAT Q3 2024 USD MNF'] - emerging['MAT Q3 2023 USD MNF']) / 
                                    emerging['MAT Q3 2023 USD MNF'] * 100)
            
            emerging = emerging[emerging['GROWTH_%'] >= min_growth]
            
            return emerging[required_cols + ['GROWTH_%']].sort_values('GROWTH_%', ascending=False)
    
    return pd.DataFrame()

# ============================================================================
# GÖRSELLEŞTİRME FONKSİYONLARI - ANALİZ ODAKLI
# ============================================================================

def plot_market_share_treemap(df, title="Pazar Payı Dağılımı"):
    """Pazar payı treemap grafiği"""
    if df.empty or 'MARKET_SHARE_%' not in df.columns:
        return None
    
    # En yüksek pazar payına sahip 20 ürün
    top_products = df.nlargest(20, 'MARKET_SHARE_%')
    
    fig = px.treemap(
        top_products,
        path=[px.Constant("Pazar"), 'MANUFACTURER', 'MOLECULE', 'INTERNATIONAL PRODUCT'],
        values='MARKET_SHARE_%',
        color='MAT Q3 2024 USD MNF',
        color_continuous_scale='RdBu',
        title=title,
        hover_data=['MAT Q3 2024 USD MNF', 'GROWTH_%'] if 'GROWTH_%' in df.columns else ['MAT Q3 2024 USD MNF']
    )
    
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig

def plot_growth_matrix(df):
    """Büyüme matrisi (BCG Matrix benzeri)"""
    if df.empty or 'MARKET_SHARE_%' not in df.columns or 'GROWTH_%' not in df.columns:
        return None
    
    fig = px.scatter(
        df,
        x='MARKET_SHARE_%',
        y='GROWTH_%',
        size='MAT Q3 2024 USD MNF',
        color='MANUFACTURER',
        hover_name='INTERNATIONAL PRODUCT',
        title='Büyüme-Pazar Payı Matrisi',
        labels={
            'MARKET_SHARE_%': 'Pazar Payı (%)',
            'GROWTH_%': 'Büyüme Oranı (%)'
        },
        size_max=60
    )
    
    # Quadrant çizgileri
    if 'MARKET_SHARE_%' in df.columns and 'GROWTH_%' in df.columns:
        market_median = df['MARKET_SHARE_%'].median()
        growth_median = df['GROWTH_%'].median()
        
        fig.add_hline(y=growth_median, line_dash="dash", line_color="gray")
        fig.add_vline(x=market_median, line_dash="dash", line_color="gray")
        
        # Quadrant etiketleri
        fig.add_annotation(x=market_median/2, y=growth_median*1.5, text="Yıldız", showarrow=False)
        fig.add_annotation(x=market_median*1.5, y=growth_median*1.5, text="Lider", showarrow=False)
        fig.add_annotation(x=market_median/2, y=growth_median/2, text="Soru İşareti", showarrow=False)
        fig.add_annotation(x=market_median*1.5, y=growth_median/2, text="Nakit İneği", showarrow=False)
    
    fig.update_layout(height=600)
    return fig

def plot_sales_composition(df, group_by='MANUFACTURER'):
    """Satış kompozisyonu grafiği"""
    if df.empty or group_by not in df.columns:
        return None
    
    if group_by == 'MANUFACTURER':
        grouped = df.groupby('MANUFACTURER')['MAT Q3 2024 USD MNF'].sum().reset_index()
        title = 'Üreticilere Göre Satış Dağılımı'
    elif group_by == 'MOLECULE':
        grouped = df.groupby('MOLECULE')['MAT Q3 2024 USD MNF'].sum().reset_index()
        title = 'Moleküllere Göre Satış Dağılımı'
    else:
        return None
    
    # En yüksek 10'u göster
    top_grouped = grouped.nlargest(10, 'MAT Q3 2024 USD MNF')
    
    fig = px.bar(
        top_grouped,
        x=group_by,
        y='MAT Q3 2024 USD MNF',
        title=title,
        color='MAT Q3 2024 USD MNF',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title=group_by,
        yaxis_title='Satış (USD)',
        xaxis_tickangle=-45
    )
    
    return fig

def plot_price_vs_volume(df):
    """Fiyat-Volume analizi"""
    if df.empty or 'MAT Q3 2024 SU AVG PRICE USD MNF' not in df.columns or 'MAT Q3 2024 STANDARD UNITS' not in df.columns:
        return None
    
    fig = px.scatter(
        df,
        x='MAT Q3 2024 SU AVG PRICE USD MNF',
        y='MAT Q3 2024 STANDARD UNITS',
        size='MAT Q3 2024 USD MNF',
        color='MANUFACTURER',
        hover_name='INTERNATIONAL PRODUCT',
        title='Fiyat-Volume Analizi',
        log_x=True,
        log_y=True,
        labels={
            'MAT Q3 2024 SU AVG PRICE USD MNF': 'Ortalama Fiyat (USD)',
            'MAT Q3 2024 STANDARD UNITS': 'Satış Volume'
        }
    )
    
    fig.update_layout(height=500)
    return fig

def plot_time_series_growth(df):
    """Zaman serisi büyüme grafiği"""
    if df.empty:
        return None
    
    # Yıllara göre toplam satış
    years_data = {}
    
    for year in [2022, 2023, 2024]:
        col_name = f'MAT Q3 {year} USD MNF'
        if col_name in df.columns:
            years_data[year] = df[col_name].sum()
    
    if len(years_data) < 2:
        return None
    
    years_df = pd.DataFrame(list(years_data.items()), columns=['Yıl', 'Toplam Satış'])
    
    fig = px.line(
        years_df,
        x='Yıl',
        y='Toplam Satış',
        title='Yıllara Göre Toplam Satış Trendi',
        markers=True
    )
    
    # Büyüme yüzdelerini ekle
    for i in range(1, len(years_df)):
        prev = years_df.iloc[i-1]['Toplam Satış']
        curr = years_df.iloc[i]['Toplam Satış']
        growth = ((curr - prev) / prev * 100) if prev > 0 else 0
        
        fig.add_annotation(
            x=years_df.iloc[i]['Yıl'],
            y=curr,
            text=f"%{growth:.1f}",
            showarrow=True,
            arrowhead=1
        )
    
    fig.update_layout(
        xaxis_title="Yıl",
        yaxis_title="Toplam Satış (USD)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f"
    )
    
    return fig

# ============================================================================
# DASHBOARD ARAYÜZÜ
# ============================================================================

def render_kpi_cards(df):
    """Ana KPI kartlarını render et"""
    if df.empty:
        st.warning("Veri yüklenmedi. Lütfen Excel dosyası yükleyin.")
        return
    
    st.markdown('<div class="section-header">📈 ANA PERFORMANS GÖSTERGELERİ</div>', unsafe_allow_html=True)
    
    # KPI hesaplamaları
    total_sales_2024 = df['MAT Q3 2024 USD MNF'].sum() if 'MAT Q3 2024 USD MNF' in df.columns else 0
    total_sales_2023 = df['MAT Q3 2023 USD MNF'].sum() if 'MAT Q3 2023 USD MNF' in df.columns else 0
    
    if total_sales_2023 > 0:
        sales_growth = ((total_sales_2024 - total_sales_2023) / total_sales_2023 * 100)
    else:
        sales_growth = 0
    
    total_products = len(df)
    total_molecules = df['MOLECULE'].nunique() if 'MOLECULE' in df.columns else 0
    total_manufacturers = df['MANUFACTURER'].nunique() if 'MANUFACTURER' in df.columns else 0
    
    avg_price = df['MAT Q3 2024 SU AVG PRICE USD MNF'].mean() if 'MAT Q3 2024 SU AVG PRICE USD MNF' in df.columns else 0
    
    # 4 ana KPI
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">${total_sales_2024:,.0f}</div>
            <div class="kpi-label">TOPLAM PAZAR (2024)</div>
            <div class="kpi-trend {'trend-up' if sales_growth > 0 else 'trend-down'}">
                {'↗️' if sales_growth > 0 else '↘️'} %{sales_growth:.1f} vs 2023
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_products}</div>
            <div class="kpi-label">TOPLAM ÜRÜN</div>
            <div class="kpi-trend trend-neutral">
                {total_molecules} Molekül
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_manufacturers}</div>
            <div class="kpi-label">ÜRETİCİ FİRMA</div>
            <div class="kpi-trend trend-neutral">
                Pazarda Aktif
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">${avg_price:.2f}</div>
            <div class="kpi-label">ORTALAMA FİYAT</div>
            <div class="kpi-trend trend-neutral">
                Standart Ünite Başına
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_market_analysis(df):
    """Pazar analizi bölümü"""
    st.markdown('<div class="section-header">📊 PAZAR ANALİZİ</div>', unsafe_allow_html=True)
    
    # Pazar payı analizi
    df_with_share = calculate_market_share(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pazar payı treemap
        if not df_with_share.empty:
            fig = plot_market_share_treemap(df_with_share)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Molekül performansı
        molecule_perf = analyze_molecule_performance(df)
        if not molecule_perf.empty:
            st.markdown("**🏆 TOP 5 MOLEKÜL**")
            top_molecules = molecule_perf.head(5)
            for _, row in top_molecules.iterrows():
                cols = st.columns([3, 2, 2])
                with cols[0]:
                    st.text(row['MOLECULE'])
                with cols[1]:
                    st.text(f"%{row['MARKET_SHARE_%']}")
                with cols[2]:
                    growth_class = "trend-up" if row['GROWTH_%'] > 0 else "trend-down"
                    st.markdown(f'<span class="{growth_class}">%{row["GROWTH_%"]}</span>', unsafe_allow_html=True)
    
    with col2:
        # Üretici performansı
        manufacturer_perf = analyze_manufacturer_performance(df)
        if not manufacturer_perf.empty:
            st.markdown("**🏭 TOP 5 ÜRETİCİ**")
            top_manufacturers = manufacturer_perf.head(5)
            
            for _, row in top_manufacturers.iterrows():
                cols = st.columns([3, 2, 2])
                with cols[0]:
                    st.text(row['MANUFACTURER'][:20])
                with cols[1]:
                    st.text(f"%{row['MARKET_SHARE_%']}")
                with cols[2]:
                    growth_class = "trend-up" if row['GROWTH_%'] > 0 else "trend-down"
                    st.markdown(f'<span class="{growth_class}">%{row["GROWTH_%"]}</span>', unsafe_allow_html=True)
        
        # Specialty vs Generic
        type_analysis = analyze_specialty_vs_generic(df)
        if not type_analysis.empty:
            st.markdown("**💊 ÜRÜN TİPİ DAĞILIMI**")
            for _, row in type_analysis.iterrows():
                cols = st.columns([3, 3, 2])
                with cols[0]:
                    st.text(row['PRODUCT_TYPE'])
                with cols[1]:
                    st.text(f"%{row['MARKET_SHARE_%']}")
                with cols[2]:
                    growth_class = "trend-up" if row['GROWTH_%'] > 0 else "trend-down"
                    st.markdown(f'<span class="{growth_class}">%{row["GROWTH_%"]}</span>', unsafe_allow_html=True)

def render_growth_analysis(df):
    """Büyüme analizi bölümü"""
    st.markdown('<div class="section-header">🚀 BÜYÜME ANALİZİ</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Zaman serisi büyüme
        fig = plot_time_series_growth(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # En hızlı büyüyen ürünler
        growth_df = calculate_growth_rates(df)
        if not growth_df.empty:
            fastest_growing = growth_df.nlargest(5, 'SALES_GROWTH_%')
            st.markdown("**⚡ EN HIZLI BÜYÜYEN ÜRÜNLER**")
            
            for _, row in fastest_growing.iterrows():
                if pd.notna(row['SALES_GROWTH_%']):
                    cols = st.columns([4, 3])
                    with cols[0]:
                        st.text(row['PRODUCT'][:25])
                    with cols[1]:
                        st.markdown(f'<span class="trend-up">↗ %{row["SALES_GROWTH_%"]:.0f}</span>', unsafe_allow_html=True)
    
    with col2:
        # Büyüme matrisi
        growth_df = calculate_growth_rates(df)
        if not growth_df.empty and 'GROWTH_%' in df.columns:
            fig = plot_growth_matrix(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Yükselen ürünler
        emerging = identify_emerging_products(df, min_sales=100000, min_growth=50)
        if not emerging.empty:
            st.markdown("**🌟 YÜKSELEN ÜRÜNLER**")
            
            for _, row in emerging.head(3).iterrows():
                cols = st.columns([4, 3])
                with cols[0]:
                    st.text(row['INTERNATIONAL PRODUCT'][:25])
                with cols[1]:
                    st.markdown(f'<span class="trend-up">🚀 %{row["GROWTH_%"]:.0f}</span>', unsafe_allow_html=True)

def render_price_analysis(df):
    """Fiyat analizi bölümü"""
    st.markdown('<div class="section-header">💰 FİYAT ANALİZİ</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fiyat-Volume analizi
        fig = plot_price_vs_volume(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Fiyat segmentasyonu
        price_segments = analyze_price_segments(df)
        if not price_segments.empty:
            st.markdown("**📊 FİYAT SEGMENTLERİ**")
            for _, row in price_segments.iterrows():
                cols = st.columns([2, 3, 2])
                with cols[0]:
                    st.text(row['PRICE_SEGMENT'])
                with cols[1]:
                    st.text(f"${row['TOTAL_SALES']:,.0f}")
                with cols[2]:
                    st.text(f"{row['PRODUCT_COUNT']} ürün")
    
    with col2:
        # Satış kompozisyonu
        fig = plot_sales_composition(df, group_by='MANUFACTURER')
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # En pahalı ürünler
        if 'MAT Q3 2024 SU AVG PRICE USD MNF' in df.columns:
            expensive_products = df.nlargest(5, 'MAT Q3 2024 SU AVG PRICE USD MNF')
            st.markdown("**💎 EN PAHALI ÜRÜNLER**")
            
            for _, row in expensive_products.iterrows():
                cols = st.columns([4, 3])
                with cols[0]:
                    st.text(row['INTERNATIONAL PRODUCT'][:25])
                with cols[1]:
                    st.text(f"${row['MAT Q3 2024 SU AVG PRICE USD MNF']:,.2f}")

def render_competitive_analysis(df):
    """Rekabet analizi bölümü"""
    st.markdown('<div class="section-header">🥊 REKABET ANALİZİ</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top ürünler
        top_products = identify_top_performers(df, n=10)
        if not top_products.empty:
            st.markdown("**🏆 TOP 10 ÜRÜN**")
            
            for idx, row in top_products.iterrows():
                cols = st.columns([4, 3, 2])
                with cols[0]:
                    st.text(row['INTERNATIONAL PRODUCT'][:20])
                with cols[1]:
                    st.text(f"${row['MAT Q3 2024 USD MNF']:,.0f}")
                with cols[2]:
                    growth_class = "trend-up" if row['GROWTH_%'] > 0 else "trend-down"
                    st.markdown(f'<span class="{growth_class}">%{row["GROWTH_%"]}</span>', unsafe_allow_html=True)
    
    with col2:
        # Molekül bazlı rekabet
        molecule_perf = analyze_molecule_performance(df)
        if not molecule_perf.empty:
            # En rekabetçi moleküller (çok üreticili)
            competitive_molecules = molecule_perf.nlargest(5, 'MANUFACTURERS')
            st.markdown("**⚔️ REKABETÇİ MOLEKÜLLER**")
            
            for _, row in competitive_molecules.iterrows():
                cols = st.columns([4, 2, 2])
                with cols[0]:
                    st.text(row['MOLECULE'][:20])
                with cols[1]:
                    st.text(f"{row['MANUFACTURERS']} firma")
                with cols[2]:
                    st.text(f"%{row['MARKET_SHARE_%']}")

def render_strategic_recommendations(df):
    """Stratejik öneriler bölümü"""
    st.markdown('<div class="section-header">🎯 STRATEJİK ÖNERİLER</div>', unsafe_allow_html=True)
    
    recommendations = []
    
    # 1. Pazar payı analizi
    df_with_share = calculate_market_share(df)
    if not df_with_share.empty:
        top_5_share = df_with_share['MARKET_SHARE_%'].nlargest(5).sum()
        if top_5_share > 50:
            recommendations.append({
                'type': 'warning',
                'title': 'Pazar Konsantrasyonu Yüksek',
                'message': f'Top 5 ürün pazarın %{top_5_share:.0f}\'ini kontrol ediyor. Yeni ürün girişleri zor olabilir.'
            })
    
    # 2. Büyüme analizi
    growth_df = calculate_growth_rates(df)
    if not growth_df.empty:
        avg_growth = growth_df['SALES_GROWTH_%'].mean()
        if avg_growth > 20:
            recommendations.append({
                'type': 'success',
                'title': 'Pazar Dinamik ve Büyüyor',
                'message': f'Ortalama büyüme oranı %{avg_growth:.0f}. Pazar genişleme fırsatları mevcut.'
            })
    
    # 3. Fiyat analizi
    if 'MAT Q3 2024 SU AVG PRICE USD MNF' in df.columns:
        price_std = df['MAT Q3 2024 SU AVG PRICE USD MNF'].std()
        if price_std > df['MAT Q3 2024 SU AVG PRICE USD MNF'].mean() * 0.5:
            recommendations.append({
                'type': 'info',
                'title': 'Fiyat Segmentasyonu Fırsatı',
                'message': 'Fiyat varyasyonu yüksek. Farklı segmentlere yönelik ürünler geliştirilebilir.'
            })
    
    # 4. Üretici konsantrasyonu
    manufacturer_perf = analyze_manufacturer_performance(df)
    if not manufacturer_perf.empty:
        top_3_share = manufacturer_perf['MARKET_SHARE_%'].nlargest(3).sum()
        if top_3_share > 60:
            recommendations.append({
                'type': 'warning',
                'title': 'Üretici Konsantrasyonu',
                'message': f'Top 3 üretici pazarın %{top_3_share:.0f}\'ini kontrol ediyor. Tedarikçi çeşitlendirmesi önerilir.'
            })
    
    # Önerileri göster
    cols = st.columns(2)
    for idx, rec in enumerate(recommendations[:4]):  # Max 4 öneri
        with cols[idx % 2]:
            if rec['type'] == 'success':
                st.success(f"**{rec['title']}**\n\n{rec['message']}")
            elif rec['type'] == 'warning':
                st.warning(f"**{rec['title']}**\n\n{rec['message']}")
            else:
                st.info(f"**{rec['title']}**\n\n{rec['message']}")

# ============================================================================
# ANA UYGULAMA
# ============================================================================

def main():
    # Başlık
    st.markdown('<div class="main-header">💊 PHARMA MARKET ANALYTICS DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown("**İlaç Pazarı Analiz Platformu - Anında Analiz, Keşif Değil**")
    
    # Sidebar - Dosya yükleme
    with st.sidebar:
        st.markdown("### 📁 VERİ YÜKLEME")
        uploaded_file = st.file_uploader("Excel dosyası yükleyin", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                
                # Kolon isimlerini standardize et
                df.columns = [str(col).strip().upper() for col in df.columns]
                
                # Session state'e kaydet
                st.session_state['df'] = df
                st.success(f"✓ {uploaded_file.name} yüklendi!")
                
                # Hızlı bilgiler
                st.markdown("---")
                st.markdown("### 📊 VERİ ÖZETİ")
                st.write(f"**Satır:** {len(df):,}")
                st.write(f"**Kolon:** {len(df.columns)}")
                st.write(f"**Ürün:** {df['INTERNATIONAL PRODUCT'].nunique() if 'INTERNATIONAL PRODUCT' in df.columns else 'N/A'}")
                
            except Exception as e:
                st.error(f"Dosya yükleme hatası: {str(e)}")
        
        # Demo butonu
        if st.button("🎮 DEMO VERİSİ İLE DENE", type="primary", use_container_width=True):
            # Örnek veri oluştur
            demo_data = pd.DataFrame({
                'INTERNATIONAL PRODUCT': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
                'MOLECULE': ['Molecule X', 'Molecule Y', 'Molecule X', 'Molecule Z', 'Molecule Y'],
                'MANUFACTURER': ['Company A', 'Company B', 'Company A', 'Company C', 'Company B'],
                'SPECIALTY PRODUCT': ['SPECIALTY INFERRED', 'NON SPECIALTY', 'SPECIALTY INFERRED', 'NON SPECIALTY', 'SPECIALTY INFERRED'],
                'MAT Q3 2024 USD MNF': [1000000, 750000, 500000, 300000, 250000],
                'MAT Q3 2023 USD MNF': [900000, 700000, 450000, 280000, 200000],
                'MAT Q3 2024 STANDARD UNITS': [10000, 15000, 5000, 3000, 2000],
                'MAT Q3 2023 STANDARD UNITS': [9500, 14000, 4800, 2900, 1900],
                'MAT Q3 2024 SU AVG PRICE USD MNF': [100, 50, 100, 100, 125],
                'MAT Q3 2023 SU AVG PRICE USD MNF': [95, 50, 94, 97, 105]
            })
            
            st.session_state['df'] = demo_data
            st.success("✅ Demo verisi yüklendi!")
            st.rerun()
    
    # Ana içerik
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        
        # KPI'lar
        render_kpi_cards(df)
        
        # Analiz bölümleri
        render_market_analysis(df)
        render_growth_analysis(df)
        render_price_analysis(df)
        render_competitive_analysis(df)
        render_strategic_recommendations(df)
        
        # Detaylı tablo (isteğe bağlı)
        with st.expander("📋 DETAYLI VERİ TABLOSU", expanded=False):
            st.dataframe(df, use_container_width=True)
            
    else:
        # Hoşgeldin ekranı
        st.markdown("""
        <div style="text-align: center; padding: 4rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 border-radius: 15px; color: white; margin: 2rem 0;">
            <h2 style="color: white; margin-bottom: 1.5rem;">🚀 HEMEN ANALİZE BAŞLAYIN</h2>
            <p style="font-size: 1.2rem; margin-bottom: 2rem;">
                Sol taraftan Excel dosyanızı yükleyin veya demo verisi ile platformu deneyin
            </p>
            <p style="font-size: 1rem; opacity: 0.9;">
                Desteklenen veri formatı: İlaç pazarı verileri (ürün, molekül, üretici, satış, fiyat)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Örnek analiz çıktısı
        st.markdown("### 📈 ÖRNEK ANALİZ ÇIKTILARI")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Pazar Büyüklüğü", "$125M", "%15.2")
        with col2:
            st.metric("Ortalama Fiyat", "$85.50", "St. Ünite")
        with col3:
            st.metric("Ürün Çeşitliliği", "1,250", "Molekül")

if __name__ == "__main__":
    main()
