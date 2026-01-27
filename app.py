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
    .data-card {
        background: #F9FAFB;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ANALİZ FONKSİYONLARI - SONUÇ ODAKLI
# ============================================================================

def normalize_column_names(df):
    """Kolon isimlerini normalize et ve standartlaştır"""
    if df.empty:
        return df
    
    column_mapping = {}
    for col in df.columns:
        col_str = str(col).strip().upper()
        
        # Anahtar kelimelere göre eşleştirme
        if 'INTERNATIONAL PRODUCT' in col_str or 'PRODUCT' in col_str:
            column_mapping[col] = 'PRODUCT'
        elif 'MOLECULE' in col_str:
            column_mapping[col] = 'MOLECULE'
        elif 'MANUFACTURER' in col_str:
            column_mapping[col] = 'MANUFACTURER'
        elif 'SPECIALTY' in col_str:
            column_mapping[col] = 'SPECIALTY_PRODUCT'
        elif '2024' in col_str and ('USD' in col_str or 'MNF' in col_str):
            if '2024' in col_str and 'USD' in col_str:
                column_mapping[col] = 'SALES_2024'
            elif '2024' in col_str and 'UNITS' in col_str:
                column_mapping[col] = 'UNITS_2024'
            elif '2024' in col_str and 'PRICE' in col_str:
                column_mapping[col] = 'PRICE_2024'
        elif '2023' in col_str and ('USD' in col_str or 'MNF' in col_str):
            if '2023' in col_str and 'USD' in col_str:
                column_mapping[col] = 'SALES_2023'
            elif '2023' in col_str and 'UNITS' in col_str:
                column_mapping[col] = 'UNITS_2023'
            elif '2023' in col_str and 'PRICE' in col_str:
                column_mapping[col] = 'PRICE_2023'
        elif '2022' in col_str and ('USD' in col_str or 'MNF' in col_str):
            if '2022' in col_str and 'USD' in col_str:
                column_mapping[col] = 'SALES_2022'
    
    # Kolon isimlerini güncelle
    df = df.rename(columns=column_mapping)
    
    # Eksik kolonları kontrol et
    required_cols = ['PRODUCT', 'MOLECULE', 'MANUFACTURER', 'SALES_2024']
    for col in required_cols:
        if col not in df.columns:
            # Benzer kolon bulmaya çalış
            for df_col in df.columns:
                if col.split('_')[0].lower() in df_col.lower():
                    df = df.rename(columns={df_col: col})
                    break
    
    return df

def calculate_market_share(df):
    """Pazar payı analizi"""
    if df.empty or 'SALES_2024' not in df.columns:
        return df
    
    df = df.copy()
    
    try:
        # Toplam satış
        total_sales = df['SALES_2024'].sum()
        if total_sales > 0:
            df['MARKET_SHARE_%'] = (df['SALES_2024'] / total_sales * 100).round(2)
        else:
            df['MARKET_SHARE_%'] = 0
    except:
        df['MARKET_SHARE_%'] = 0
    
    return df

def calculate_growth_rates(df):
    """Büyüme oranları analizi"""
    if df.empty:
        return pd.DataFrame()
    
    growth_data = []
    
    for idx, row in df.iterrows():
        try:
            # 2023-2024 büyüme oranı
            sales_2024 = row.get('SALES_2024', 0)
            sales_2023 = row.get('SALES_2023', 0)
            
            if pd.notna(sales_2023) and sales_2023 != 0:
                sales_growth = ((sales_2024 - sales_2023) / sales_2023 * 100)
            else:
                sales_growth = np.nan
            
            # Volume büyüme
            units_2024 = row.get('UNITS_2024', 0)
            units_2023 = row.get('UNITS_2023', 0)
            
            if pd.notna(units_2023) and units_2023 != 0:
                volume_growth = ((units_2024 - units_2023) / units_2023 * 100)
            else:
                volume_growth = np.nan
            
            # Fiyat değişimi
            price_2024 = row.get('PRICE_2024', 0)
            price_2023 = row.get('PRICE_2023', 0)
            
            if pd.notna(price_2023) and price_2023 != 0:
                price_change = ((price_2024 - price_2023) / price_2023 * 100)
            else:
                price_change = np.nan
            
            growth_data.append({
                'PRODUCT': row.get('PRODUCT', ''),
                'MOLECULE': row.get('MOLECULE', ''),
                'MANUFACTURER': row.get('MANUFACTURER', ''),
                'SALES_GROWTH_%': round(sales_growth, 1) if not pd.isna(sales_growth) else np.nan,
                'VOLUME_GROWTH_%': round(volume_growth, 1) if not pd.isna(volume_growth) else np.nan,
                'PRICE_CHANGE_%': round(price_change, 1) if not pd.isna(price_change) else np.nan,
                'SALES_2024': sales_2024,
                'SALES_2023': sales_2023,
                'VOLUME_2024': units_2024
            })
            
        except:
            continue
    
    return pd.DataFrame(growth_data)

def analyze_molecule_performance(df):
    """Molekül bazlı performans analizi"""
    if df.empty or 'MOLECULE' not in df.columns:
        return pd.DataFrame()
    
    molecule_stats = []
    molecules = df['MOLECULE'].dropna().unique()
    
    for molecule in molecules[:50]:  # İlk 50 molekülü analiz et
        try:
            molecule_df = df[df['MOLECULE'] == molecule]
            
            # Toplam satış
            total_sales_2024 = molecule_df['SALES_2024'].sum() if 'SALES_2024' in df.columns else 0
            total_sales_2023 = molecule_df['SALES_2023'].sum() if 'SALES_2023' in df.columns else 0
            
            # Büyüme
            if total_sales_2023 > 0:
                growth_rate = ((total_sales_2024 - total_sales_2023) / total_sales_2023 * 100)
            else:
                growth_rate = 0 if total_sales_2024 > 0 else np.nan
            
            # Üretici sayısı
            manufacturers = molecule_df['MANUFACTURER'].nunique() if 'MANUFACTURER' in molecule_df.columns else 0
            
            # Ürün sayısı
            products = molecule_df['PRODUCT'].nunique() if 'PRODUCT' in molecule_df.columns else 0
            
            # Ortalama fiyat
            avg_price = molecule_df['PRICE_2024'].mean() if 'PRICE_2024' in molecule_df.columns else 0
            
            molecule_stats.append({
                'MOLECULE': str(molecule)[:50],  # Uzun isimleri kısalt
                'TOTAL_SALES_2024': total_sales_2024,
                'TOTAL_SALES_2023': total_sales_2023,
                'GROWTH_%': round(growth_rate, 1) if not pd.isna(growth_rate) else 0,
                'MANUFACTURERS': manufacturers,
                'PRODUCTS': products,
                'AVG_PRICE': round(avg_price, 2) if not pd.isna(avg_price) else 0,
                'MARKET_SHARE_%': 0  # Sonradan hesaplanacak
            })
        except:
            continue
    
    if not molecule_stats:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(molecule_stats)
    
    # Pazar payı hesapla
    if 'TOTAL_SALES_2024' in result_df.columns:
        total_market = result_df['TOTAL_SALES_2024'].sum()
        if total_market > 0:
            result_df['MARKET_SHARE_%'] = (result_df['TOTAL_SALES_2024'] / total_market * 100).round(2)
        else:
            result_df['MARKET_SHARE_%'] = 0
    
    return result_df.sort_values('TOTAL_SALES_2024', ascending=False).head(20)  # Top 20'yi göster

def analyze_manufacturer_performance(df):
    """Üretici bazlı performans analizi"""
    if df.empty or 'MANUFACTURER' not in df.columns:
        return pd.DataFrame()
    
    manufacturer_stats = []
    manufacturers = df['MANUFACTURER'].dropna().unique()
    
    for manufacturer in manufacturers[:50]:  # İlk 50 üreticiyi analiz et
        try:
            man_df = df[df['MANUFACTURER'] == manufacturer]
            
            # Satış verileri
            total_sales_2024 = man_df['SALES_2024'].sum() if 'SALES_2024' in man_df.columns else 0
            total_sales_2023 = man_df['SALES_2023'].sum() if 'SALES_2023' in man_df.columns else 0
            
            # Büyüme
            if total_sales_2023 > 0:
                growth_rate = ((total_sales_2024 - total_sales_2023) / total_sales_2023 * 100)
            else:
                growth_rate = 0 if total_sales_2024 > 0 else np.nan
            
            # Portföy analizi
            molecules = man_df['MOLECULE'].nunique() if 'MOLECULE' in man_df.columns else 0
            products = man_df['PRODUCT'].nunique() if 'PRODUCT' in man_df.columns else 0
            
            # Ortalama fiyat
            avg_price = man_df['PRICE_2024'].mean() if 'PRICE_2024' in man_df.columns else 0
            
            # Specialty vs Non-specialty
            specialty_sales = 0
            if 'SPECIALTY_PRODUCT' in man_df.columns:
                try:
                    specialty_mask = man_df['SPECIALTY_PRODUCT'].astype(str).str.contains('SPECIALTY', case=False, na=False)
                    if specialty_mask.any():
                        specialty_sales = man_df.loc[specialty_mask, 'SALES_2024'].sum()
                except:
                    specialty_sales = 0
            
            manufacturer_stats.append({
                'MANUFACTURER': str(manufacturer)[:50],  # Uzun isimleri kısalt
                'TOTAL_SALES_2024': total_sales_2024,
                'TOTAL_SALES_2023': total_sales_2023,
                'GROWTH_%': round(growth_rate, 1) if not pd.isna(growth_rate) else 0,
                'MOLECULES': molecules,
                'PRODUCTS': products,
                'AVG_PRICE': round(avg_price, 2) if not pd.isna(avg_price) else 0,
                'SPECIALTY_SALES': specialty_sales,
                'SPECIALTY_SHARE_%': (specialty_sales / total_sales_2024 * 100).round(2) if total_sales_2024 > 0 else 0
            })
        except:
            continue
    
    if not manufacturer_stats:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(manufacturer_stats)
    
    # Pazar payı hesapla
    if 'TOTAL_SALES_2024' in result_df.columns:
        total_market = result_df['TOTAL_SALES_2024'].sum()
        if total_market > 0:
            result_df['MARKET_SHARE_%'] = (result_df['TOTAL_SALES_2024'] / total_market * 100).round(2)
        else:
            result_df['MARKET_SHARE_%'] = 0
    
    return result_df.sort_values('TOTAL_SALES_2024', ascending=False).head(20)  # Top 20'yi göster

def analyze_price_segments(df):
    """Fiyat segmentasyonu analizi"""
    if df.empty or 'PRICE_2024' not in df.columns:
        return pd.DataFrame()
    
    try:
        # Fiyat aralıklarına göre segmentasyon
        price_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
        price_labels = ['<10$', '10-50$', '50-100$', '100-500$', '500-1000$', '>1000$']
        
        df_copy = df.copy()
        df_copy['PRICE_SEGMENT'] = pd.cut(
            df_copy['PRICE_2024'], 
            bins=price_bins, 
            labels=price_labels, 
            right=False
        )
        
        segment_stats = []
        for segment in price_labels:
            segment_df = df_copy[df_copy['PRICE_SEGMENT'] == segment]
            
            if len(segment_df) > 0:
                total_sales = segment_df['SALES_2024'].sum()
                total_volume = segment_df['UNITS_2024'].sum() if 'UNITS_2024' in segment_df.columns else 0
                product_count = len(segment_df)
                avg_price = segment_df['PRICE_2024'].mean()
                
                segment_stats.append({
                    'PRICE_SEGMENT': segment,
                    'TOTAL_SALES': total_sales,
                    'TOTAL_VOLUME': total_volume,
                    'PRODUCT_COUNT': product_count,
                    'AVG_PRICE': round(avg_price, 2),
                    'SALES_PER_PRODUCT': round(total_sales / product_count, 2) if product_count > 0 else 0
                })
        
        return pd.DataFrame(segment_stats)
    except:
        return pd.DataFrame()

def analyze_specialty_vs_generic(df):
    """Specialty vs Non-specialty analizi"""
    if df.empty or 'SPECIALTY_PRODUCT' not in df.columns:
        return pd.DataFrame()
    
    try:
        df_copy = df.copy()
        
        # Product type'ı belirle
        def get_product_type(x):
            if pd.isna(x):
                return 'UNKNOWN'
            x_str = str(x).upper()
            if 'SPECIALTY' in x_str:
                return 'SPECIALTY'
            elif 'NON' in x_str or 'GENERIC' in x_str:
                return 'NON-SPECIALTY'
            else:
                return 'UNKNOWN'
        
        df_copy['PRODUCT_TYPE'] = df_copy['SPECIALTY_PRODUCT'].apply(get_product_type)
        
        type_stats = []
        
        for p_type in ['SPECIALTY', 'NON-SPECIALTY', 'UNKNOWN']:
            type_df = df_copy[df_copy['PRODUCT_TYPE'] == p_type]
            
            if len(type_df) > 0:
                total_sales_2024 = type_df['SALES_2024'].sum()
                total_sales_2023 = type_df['SALES_2023'].sum() if 'SALES_2023' in type_df.columns else 0
                
                if total_sales_2023 > 0:
                    growth = ((total_sales_2024 - total_sales_2023) / total_sales_2023 * 100)
                else:
                    growth = 0
                
                product_count = len(type_df)
                avg_price = type_df['PRICE_2024'].mean() if 'PRICE_2024' in type_df.columns else 0
                
                total_market = df_copy['SALES_2024'].sum()
                market_share = (total_sales_2024 / total_market * 100).round(2) if total_market > 0 else 0
                
                type_stats.append({
                    'PRODUCT_TYPE': p_type,
                    'TOTAL_SALES_2024': total_sales_2024,
                    'MARKET_SHARE_%': market_share,
                    'GROWTH_%': round(growth, 1),
                    'PRODUCT_COUNT': product_count,
                    'AVG_PRICE': round(avg_price, 2),
                    'SALES_PER_PRODUCT': round(total_sales_2024 / product_count, 2) if product_count > 0 else 0
                })
        
        return pd.DataFrame(type_stats)
    except:
        return pd.DataFrame()

def identify_top_performers(df, n=10):
    """En iyi performans gösteren ürünleri belirle"""
    if df.empty or 'SALES_2024' not in df.columns:
        return pd.DataFrame()
    
    try:
        # Satışa göre sırala
        top_sales = df.nlargest(n, 'SALES_2024')[['PRODUCT', 'MOLECULE', 
                                                  'MANUFACTURER', 'SALES_2024',
                                                  'SALES_2023']].copy()
        
        # Büyüme oranı ekle
        top_sales['GROWTH_%'] = 0
        for idx, row in top_sales.iterrows():
            if row['SALES_2023'] > 0:
                growth = ((row['SALES_2024'] - row['SALES_2023']) / row['SALES_2023'] * 100)
                top_sales.at[idx, 'GROWTH_%'] = round(growth, 1)
        
        return top_sales
    except:
        return pd.DataFrame()

def identify_emerging_products(df, min_sales=100000, min_growth=20):
    """Yükselen ürünleri belirle"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        required_cols = ['PRODUCT', 'MOLECULE', 'MANUFACTURER', 'SALES_2024', 'SALES_2023']
        
        if all(col in df.columns for col in required_cols):
            # Satışı min_sales üzerinde ve büyümesi min_growth üzerinde olanlar
            emerging = df[
                (df['SALES_2024'] >= min_sales) &
                (df['SALES_2023'] > 0)
            ].copy()
            
            if not emerging.empty:
                emerging['GROWTH_%'] = ((emerging['SALES_2024'] - emerging['SALES_2023']) / 
                                        emerging['SALES_2023'] * 100)
                
                emerging = emerging[emerging['GROWTH_%'] >= min_growth]
                
                return emerging[required_cols + ['GROWTH_%']].sort_values('GROWTH_%', ascending=False)
    except:
        pass
    
    return pd.DataFrame()

# ============================================================================
# GÖRSELLEŞTİRME FONKSİYONLARI - ANALİZ ODAKLI
# ============================================================================

def plot_market_share_treemap(df, title="Pazar Payı Dağılımı"):
    """Pazar payı treemap grafiği"""
    if df.empty or 'MARKET_SHARE_%' not in df.columns:
        return None
    
    try:
        # En yüksek pazar payına sahip 20 ürün
        top_products = df.nlargest(20, 'MARKET_SHARE_%').copy()
        
        # Eksik değerleri doldur
        for col in ['MANUFACTURER', 'MOLECULE', 'PRODUCT']:
            if col in top_products.columns:
                top_products[col] = top_products[col].fillna('Bilinmeyen')
        
        fig = px.treemap(
            top_products,
            path=[px.Constant("Pazar"), 'MANUFACTURER', 'MOLECULE', 'PRODUCT'],
            values='MARKET_SHARE_%',
            color='SALES_2024',
            color_continuous_scale='RdBu',
            title=title,
            hover_data=['SALES_2024']
        )
        
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        return fig
    except:
        return None

def plot_sales_composition(df, group_by='MANUFACTURER'):
    """Satış kompozisyonu grafiği"""
    if df.empty or group_by not in df.columns:
        return None
    
    try:
        if group_by == 'MANUFACTURER':
            grouped = df.groupby('MANUFACTURER')['SALES_2024'].sum().reset_index()
            title = 'Üreticilere Göre Satış Dağılımı'
        elif group_by == 'MOLECULE':
            grouped = df.groupby('MOLECULE')['SALES_2024'].sum().reset_index()
            title = 'Moleküllere Göre Satış Dağılımı'
        else:
            return None
        
        # En yüksek 10'u göster
        top_grouped = grouped.nlargest(10, 'SALES_2024')
        
        fig = px.bar(
            top_grouped,
            x=group_by,
            y='SALES_2024',
            title=title,
            color='SALES_2024',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title=group_by,
            yaxis_title='Satış (USD)',
            xaxis_tickangle=-45
        )
        
        return fig
    except:
        return None

def plot_price_vs_volume(df):
    """Fiyat-Volume analizi"""
    if df.empty or 'PRICE_2024' not in df.columns or 'UNITS_2024' not in df.columns:
        return None
    
    try:
        fig = px.scatter(
            df,
            x='PRICE_2024',
            y='UNITS_2024',
            size='SALES_2024',
            color='MANUFACTURER' if 'MANUFACTURER' in df.columns else None,
            hover_name='PRODUCT' if 'PRODUCT' in df.columns else None,
            title='Fiyat-Volume Analizi',
            log_x=True,
            log_y=True,
            labels={
                'PRICE_2024': 'Ortalama Fiyat (USD)',
                'UNITS_2024': 'Satış Volume'
            }
        )
        
        fig.update_layout(height=500)
        return fig
    except:
        return None

def plot_time_series_growth(df):
    """Zaman serisi büyüme grafiği"""
    if df.empty:
        return None
    
    try:
        # Yıllara göre toplam satış
        years_data = {}
        
        for year in [2022, 2023, 2024]:
            col_name = f'SALES_{year}'
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
    except:
        return None

# ============================================================================
# DASHBOARD ARAYÜZÜ
# ============================================================================

def render_kpi_cards(df):
    """Ana KPI kartlarını render et"""
    if df.empty:
        st.warning("📁 Veri yüklenmedi. Lütfen Excel dosyası yükleyin.")
        return
    
    st.markdown('<div class="section-header">📈 ANA PERFORMANS GÖSTERGELERİ</div>', unsafe_allow_html=True)
    
    try:
        # KPI hesaplamaları
        total_sales_2024 = df['SALES_2024'].sum() if 'SALES_2024' in df.columns else 0
        total_sales_2023 = df['SALES_2023'].sum() if 'SALES_2023' in df.columns else 0
        
        if total_sales_2023 > 0:
            sales_growth = ((total_sales_2024 - total_sales_2023) / total_sales_2023 * 100)
        else:
            sales_growth = 0
        
        total_products = len(df)
        total_molecules = df['MOLECULE'].nunique() if 'MOLECULE' in df.columns else 0
        total_manufacturers = df['MANUFACTURER'].nunique() if 'MANUFACTURER' in df.columns else 0
        
        avg_price = df['PRICE_2024'].mean() if 'PRICE_2024' in df.columns else 0
        
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
            
    except Exception as e:
        st.error(f"KPI hesaplama hatası: {str(e)}")

def render_market_analysis(df):
    """Pazar analizi bölümü"""
    st.markdown('<div class="section-header">📊 PAZAR ANALİZİ</div>', unsafe_allow_html=True)
    
    try:
        # Pazar payı analizi
        df_with_share = calculate_market_share(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 TOP 5 MOLEKÜL**")
            
            # Molekül performansı
            molecule_perf = analyze_molecule_performance(df)
            if not molecule_perf.empty:
                top_molecules = molecule_perf.head(5)
                
                for _, row in top_molecules.iterrows():
                    cols = st.columns([3, 2, 2])
                    with cols[0]:
                        st.text(row['MOLECULE'][:20])
                    with cols[1]:
                        st.text(f"%{row['MARKET_SHARE_%']:.1f}")
                    with cols[2]:
                        growth_class = "trend-up" if row['GROWTH_%'] > 0 else "trend-down"
                        st.markdown(f'<span class="{growth_class}">%{row["GROWTH_%"]:.1f}</span>', unsafe_allow_html=True)
            else:
                st.info("Molekül analizi yapılamadı")
            
            # Pazar payı treemap
            if not df_with_share.empty:
                fig = plot_market_share_treemap(df_with_share)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**🏭 TOP 5 ÜRETİCİ**")
            
            # Üretici performansı
            manufacturer_perf = analyze_manufacturer_performance(df)
            if not manufacturer_perf.empty:
                top_manufacturers = manufacturer_perf.head(5)
                
                for _, row in top_manufacturers.iterrows():
                    cols = st.columns([3, 2, 2])
                    with cols[0]:
                        st.text(row['MANUFACTURER'][:20])
                    with cols[1]:
                        st.text(f"%{row['MARKET_SHARE_%']:.1f}")
                    with cols[2]:
                        growth_class = "trend-up" if row['GROWTH_%'] > 0 else "trend-down"
                        st.markdown(f'<span class="{growth_class}">%{row["GROWTH_%"]:.1f}</span>', unsafe_allow_html=True)
            else:
                st.info("Üretici analizi yapılamadı")
            
            # Specialty vs Generic
            type_analysis = analyze_specialty_vs_generic(df)
            if not type_analysis.empty:
                st.markdown("**💊 ÜRÜN TİPİ DAĞILIMI**")
                for _, row in type_analysis.iterrows():
                    if row['PRODUCT_TYPE'] != 'UNKNOWN':
                        cols = st.columns([3, 3, 2])
                        with cols[0]:
                            st.text(row['PRODUCT_TYPE'])
                        with cols[1]:
                            st.text(f"%{row['MARKET_SHARE_%']:.1f}")
                        with cols[2]:
                            growth_class = "trend-up" if row['GROWTH_%'] > 0 else "trend-down"
                            st.markdown(f'<span class="{growth_class}">%{row["GROWTH_%"]:.1f}</span>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Pazar analizi hatası: {str(e)}")

def render_growth_analysis(df):
    """Büyüme analizi bölümü"""
    st.markdown('<div class="section-header">🚀 BÜYÜME ANALİZİ</div>', unsafe_allow_html=True)
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Zaman serisi büyüme
            fig = plot_time_series_growth(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Zaman serisi verisi bulunamadı")
            
            # En hızlı büyüyen ürünler
            growth_df = calculate_growth_rates(df)
            if not growth_df.empty and 'SALES_GROWTH_%' in growth_df.columns:
                fastest_growing = growth_df.dropna(subset=['SALES_GROWTH_%']).nlargest(5, 'SALES_GROWTH_%')
                if not fastest_growing.empty:
                    st.markdown("**⚡ EN HIZLI BÜYÜYEN ÜRÜNLER**")
                    
                    for _, row in fastest_growing.iterrows():
                        if pd.notna(row['SALES_GROWTH_%']):
                            cols = st.columns([4, 3])
                            with cols[0]:
                                product_name = str(row.get('PRODUCT', ''))[:25]
                                st.text(product_name)
                            with cols[1]:
                                st.markdown(f'<span class="trend-up">↗ %{row["SALES_GROWTH_%"]:.0f}</span>', unsafe_allow_html=True)
        
        with col2:
            # Satış kompozisyonu
            fig = plot_sales_composition(df, group_by='MANUFACTURER')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Satış kompozisyon grafiği oluşturulamadı")
            
            # Yükselen ürünler
            emerging = identify_emerging_products(df, min_sales=100000, min_growth=50)
            if not emerging.empty:
                st.markdown("**🌟 YÜKSELEN ÜRÜNLER**")
                
                for _, row in emerging.head(3).iterrows():
                    cols = st.columns([4, 3])
                    with cols[0]:
                        product_name = str(row.get('PRODUCT', ''))[:25]
                        st.text(product_name)
                    with cols[1]:
                        st.markdown(f'<span class="trend-up">🚀 %{row["GROWTH_%"]:.0f}</span>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Büyüme analizi hatası: {str(e)}")

def render_price_analysis(df):
    """Fiyat analizi bölümü"""
    st.markdown('<div class="section-header">💰 FİYAT ANALİZİ</div>', unsafe_allow_html=True)
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Fiyat-Volume analizi
            fig = plot_price_vs_volume(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Fiyat-Volume analizi için veri bulunamadı")
            
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
                        st.text(f"{int(row['PRODUCT_COUNT'])} ürün")
        
        with col2:
            # Molekül bazlı satış
            fig = plot_sales_composition(df, group_by='MOLECULE')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # En pahalı ürünler
            if 'PRICE_2024' in df.columns:
                expensive_products = df.dropna(subset=['PRICE_2024']).nlargest(5, 'PRICE_2024')
                if not expensive_products.empty:
                    st.markdown("**💎 EN PAHALI ÜRÜNLER**")
                    
                    for _, row in expensive_products.iterrows():
                        cols = st.columns([4, 3])
                        with cols[0]:
                            product_name = str(row.get('PRODUCT', ''))[:25]
                            st.text(product_name)
                        with cols[1]:
                            st.text(f"${row['PRICE_2024']:,.2f}")
    
    except Exception as e:
        st.error(f"Fiyat analizi hatası: {str(e)}")

def render_competitive_analysis(df):
    """Rekabet analizi bölümü"""
    st.markdown('<div class="section-header">🥊 REKABET ANALİZİ</div>', unsafe_allow_html=True)
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top ürünler
            top_products = identify_top_performers(df, n=10)
            if not top_products.empty:
                st.markdown("**🏆 TOP 10 ÜRÜN**")
                
                for idx, row in top_products.iterrows():
                    cols = st.columns([4, 3, 2])
                    with cols[0]:
                        product_name = str(row.get('PRODUCT', ''))[:20]
                        st.text(product_name)
                    with cols[1]:
                        st.text(f"${row['SALES_2024']:,.0f}")
                    with cols[2]:
                        growth_class = "trend-up" if row['GROWTH_%'] > 0 else "trend-down"
                        st.markdown(f'<span class="{growth_class}">%{row["GROWTH_%"]:.1f}</span>', unsafe_allow_html=True)
            else:
                st.info("Top ürün analizi yapılamadı")
        
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
                        st.text(f"{int(row['MANUFACTURERS'])} firma")
                    with cols[2]:
                        st.text(f"%{row['MARKET_SHARE_%']:.1f}")
    
    except Exception as e:
        st.error(f"Rekabet analizi hatası: {str(e)}")

def render_strategic_recommendations(df):
    """Stratejik öneriler bölümü"""
    st.markdown('<div class="section-header">🎯 STRATEJİK ÖNERİLER</div>', unsafe_allow_html=True)
    
    try:
        recommendations = []
        
        # 1. Pazar büyüklüğü analizi
        total_sales_2024 = df['SALES_2024'].sum() if 'SALES_2024' in df.columns else 0
        
        if total_sales_2024 > 100000000:  # 100M USD üzeri
            recommendations.append({
                'type': 'success',
                'title': 'Büyük Pazar Fırsatı',
                'message': f'Pazar büyüklüğü ${total_sales_2024:,.0f}. Yatırım için uygun büyüklükte bir pazar.'
            })
        
        # 2. Pazar konsantrasyonu
        df_with_share = calculate_market_share(df)
        if not df_with_share.empty and 'MARKET_SHARE_%' in df_with_share.columns:
            top_5_share = df_with_share['MARKET_SHARE_%'].nlargest(5).sum()
            if top_5_share > 50:
                recommendations.append({
                    'type': 'warning',
                    'title': 'Yüksek Pazar Konsantrasyonu',
                    'message': f'Top 5 ürün pazarın %{top_5_share:.0f}\'ini kontrol ediyor.'
                })
        
        # 3. Ortalama fiyat analizi
        if 'PRICE_2024' in df.columns:
            avg_price = df['PRICE_2024'].mean()
            price_std = df['PRICE_2024'].std()
            
            if price_std > avg_price * 0.5:
                recommendations.append({
                    'type': 'info',
                    'title': 'Fiyat Segmentasyonu',
                    'message': f'Fiyat varyasyonu yüksek (Ort: ${avg_price:.2f}). Farklı segmentlere hitap edebilirsiniz.'
                })
        
        # 4. Büyüme potansiyeli
        if 'SALES_2023' in df.columns and 'SALES_2024' in df.columns:
            total_growth = ((df['SALES_2024'].sum() - df['SALES_2023'].sum()) / 
                           df['SALES_2023'].sum() * 100) if df['SALES_2023'].sum() > 0 else 0
            
            if total_growth > 15:
                recommendations.append({
                    'type': 'success',
                    'title': 'Yüksek Büyüme Potansiyeli',
                    'message': f'Pazar %{total_growth:.1f} büyüyor. Genişleme için uygun zaman.'
                })
        
        # Önerileri göster
        if recommendations:
            cols = st.columns(2)
            for idx, rec in enumerate(recommendations[:4]):  # Max 4 öneri
                with cols[idx % 2]:
                    if rec['type'] == 'success':
                        st.success(f"**{rec['title']}**\n\n{rec['message']}")
                    elif rec['type'] == 'warning':
                        st.warning(f"**{rec['title']}**\n\n{rec['message']}")
                    else:
                        st.info(f"**{rec['title']}**\n\n{rec['message']}")
        else:
            st.info("Stratejik öneri üretmek için yeterli veri bulunamadı.")
            
    except Exception as e:
        st.error(f"Stratejik öneriler hatası: {str(e)}")

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
                
                # Kolon isimlerini normalize et
                df = normalize_column_names(df)
                
                # Eksik değerleri temizle
                df = df.fillna(0)
                
                # Numerik kolonları dönüştür
                numeric_cols = ['SALES_2024', 'SALES_2023', 'SALES_2022', 
                               'UNITS_2024', 'UNITS_2023', 'UNITS_2022',
                               'PRICE_2024', 'PRICE_2023', 'PRICE_2022']
                
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Session state'e kaydet
                st.session_state['df'] = df
                st.session_state['uploaded_file_name'] = uploaded_file.name
                
                st.success(f"✅ {uploaded_file.name} başarıyla yüklendi!")
                
                # Hızlı bilgiler
                st.markdown("---")
                st.markdown("### 📊 VERİ ÖZETİ")
                st.write(f"**Satır:** {len(df):,}")
                st.write(f"**Kolon:** {len(df.columns)}")
                
                if 'PRODUCT' in df.columns:
                    st.write(f"**Ürün Sayısı:** {df['PRODUCT'].nunique():,}")
                if 'MOLECULE' in df.columns:
                    st.write(f"**Molekül Sayısı:** {df['MOLECULE'].nunique():,}")
                if 'MANUFACTURER' in df.columns:
                    st.write(f"**Üretici Sayısı:** {df['MANUFACTURER'].nunique():,}")
                
            except Exception as e:
                st.error(f"❌ Dosya yükleme hatası: {str(e)}")
                st.info("Lütfen farklı bir dosya yükleyin veya formatı kontrol edin.")
        
        # Demo butonu
        st.markdown("---")
        if st.button("🎮 DEMO VERİSİ İLE DENE", type="primary", use_container_width=True):
            try:
                # Örnek veri oluştur
                demo_data = pd.DataFrame({
                    'PRODUCT': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
                    'MOLECULE': ['Molecule X', 'Molecule Y', 'Molecule X', 'Molecule Z', 'Molecule Y'],
                    'MANUFACTURER': ['Company A', 'Company B', 'Company A', 'Company C', 'Company B'],
                    'SPECIALTY_PRODUCT': ['SPECIALTY', 'NON-SPECIALTY', 'SPECIALTY', 'NON-SPECIALTY', 'SPECIALTY'],
                    'SALES_2024': [1000000, 750000, 500000, 300000, 250000],
                    'SALES_2023': [900000, 700000, 450000, 280000, 200000],
                    'UNITS_2024': [10000, 15000, 5000, 3000, 2000],
                    'UNITS_2023': [9500, 14000, 4800, 2900, 1900],
                    'PRICE_2024': [100, 50, 100, 100, 125],
                    'PRICE_2023': [95, 50, 94, 97, 105]
                })
                
                st.session_state['df'] = demo_data
                st.session_state['uploaded_file_name'] = "demo_data.xlsx"
                
                st.success("✅ Demo verisi yüklendi!")
                st.rerun()
            except Exception as e:
                st.error(f"Demo veri hatası: {str(e)}")
    
    # Ana içerik
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        
        # Dosya bilgisi
        if 'uploaded_file_name' in st.session_state:
            st.info(f"**Yüklenen Dosya:** {st.session_state['uploaded_file_name']}")
        
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
            st.dataframe(df, use_container_width=True, height=400)
            
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
        
        # Özellikler
        st.markdown("### ✨ ÖZELLİKLER")
        
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown("""
            <div class="data-card">
                <h4>📈 Anlık KPI'lar</h4>
                <p>Veri yüklenir yüklenmez ana performans göstergeleri</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown("""
            <div class="data-card">
                <h4>📊 Pazar Analizi</h4>
                <p>Molekül ve üretici bazında pazar payı analizi</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown("""
            <div class="data-card">
                <h4>🚀 Büyüme Analizi</h4>
                <p>Yıllık büyüme oranları ve trend analizi</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
