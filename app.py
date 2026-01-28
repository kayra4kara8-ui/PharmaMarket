# app.py - Büyük Veri Seti için Global İlaç Pazarı Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time
import gc

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
        font-size: 2.5rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
        margin: 0.3rem 0;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1F2937;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    
    .stDataFrame {
        font-size: 0.9rem;
    }
    
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ================================================
# 2. VERİ YÜKLEME VE OPTİMİZASYON FONKSİYONLARI
# ================================================

@st.cache_data(ttl=3600, show_spinner="Veriler optimize ediliyor...")
def optimize_dataframe(df):
    """Büyük DataFrame'leri optimize et"""
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Sayısal sütunları optimize et
        if col_type in ['int64', 'int32']:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
                
        elif col_type in ['float64']:
            df[col] = df[col].astype(np.float32)
    
    # Kategorik sütunları optimize et
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Eşsiz değer sayısı azsa
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    st.sidebar.success(f"Bellek kullanımı: {end_mem:.2f} MB ({(start_mem - end_mem)/start_mem*100:.1f}% azaltıldı)")
    
    return df

@st.cache_data(ttl=3600, max_entries=3)
def load_excel_data(uploaded_file, sample_size=None):
    """Excel verisini yükle ve optimize et"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Dosya boyutunu kontrol et
        file_size = uploaded_file.size / (1024 ** 2)  # MB cinsinden
        
        # Büyük dosyalar için chunk okuma
        if file_size > 50:  # 50MB'tan büyükse
            status_text.text("Büyük dosya yükleniyor... Bu biraz zaman alabilir.")
            
            # İlk 1000 satırı okuyarak yapıyı anla
            sample_df = pd.read_excel(uploaded_file, nrows=1000)
            
            # Eğer örneklem istendiyse
            if sample_size and sample_size < len(sample_df):
                return sample_df.sample(min(sample_size, len(sample_df)))
            
            # Tüm veriyi yükle (optimize edilmiş)
            chunks = []
            chunk_size = 50000
            
            with pd.ExcelFile(uploaded_file) as xls:
                sheet_name = xls.sheet_names[0]
                total_rows = sum(1 for _ in pd.read_excel(xls, sheet_name, chunksize=10000))
                
                for i, chunk in enumerate(pd.read_excel(xls, sheet_name, chunksize=chunk_size)):
                    chunks.append(chunk)
                    progress = (i * chunk_size) / total_rows
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Veri yükleniyor: {i * chunk_size:,} / {total_rows:,} satır")
            
            df = pd.concat(chunks, ignore_index=True)
            progress_bar.progress(1.0)
            
        else:
            # Küçük dosyalar için direk okuma
            df = pd.read_excel(uploaded_file)
            progress_bar.progress(1.0)
        
        status_text.text("Veri optimize ediliyor...")
        df = optimize_dataframe(df)
        
        status_text.text("Hesaplanmış metrikler ekleniyor...")
        
        # Temel hesaplanmış metrikleri ekle (veri varsa)
        if 'USD_MNF' in df.columns and 'Units' in df.columns:
            df['Price_Per_Unit'] = df['USD_MNF'] / df['Units'].replace(0, np.nan)
        
        if 'Year' in df.columns and 'Quarter' in df.columns:
            df['Period'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)
        
        # Gerekli sütunları kontrol et ve ekle
        required_columns = {
            'Country': 'Country' if 'Country' in df.columns else 
                      'COUNTRY' if 'COUNTRY' in df.columns else 'Ülke',
            'Corporation': 'Corporation' if 'Corporation' in df.columns else 
                          'CORPORATION' if 'CORPORATION' in df.columns else 'Şirket',
            'Molecule': 'Molecule' if 'Molecule' in df.columns else 
                       'MOLECULE' if 'MOLECULE' in df.columns else 'Molekül',
            'USD_MNF': 'USD_MNF' if 'USD_MNF' in df.columns else 
                      'SALES' if 'SALES' in df.columns else 'Satış',
            'Units': 'Units' if 'Units' in df.columns else 
                    'UNITS' if 'UNITS' in df.columns else 'Birim'
        }
        
        # Sütun isimlerini standartlaştır
        column_mapping = {}
        for std_name, possible_names in required_columns.items():
            if isinstance(possible_names, str):
                if possible_names in df.columns:
                    column_mapping[possible_names] = std_name
            else:
                for name in possible_names:
                    if name in df.columns:
                        column_mapping[name] = std_name
                        break
        
        df = df.rename(columns=column_mapping)
        
        # Eksik sütunları ekle
        for col in ['Country', 'Corporation', 'Molecule', 'USD_MNF', 'Units']:
            if col not in df.columns:
                df[col] = np.nan
        
        status_text.text("")
        progress_bar.empty()
        
        return df
        
    except Exception as e:
        st.error(f"Veri yükleme hatası: {str(e)}")
        return None

def detect_data_columns(df):
    """Veri setindeki sütunları otomatik tespit et"""
    column_info = {}
    
    for col in df.columns:
        col_info = {
            'type': str(df[col].dtype),
            'unique_count': df[col].nunique(),
            'missing_count': df[col].isnull().sum(),
            'sample_values': df[col].dropna().unique()[:5].tolist()
        }
        column_info[col] = col_info
    
    return column_info

# ================================================
# 3. ANALİTİK FONKSİYONLAR (OPTİMİZE EDİLMİŞ)
# ================================================

class OptimizedPharmaAnalytics:
    """Büyük veri setleri için optimize edilmiş analitik fonksiyonlar"""
    
    @staticmethod
    def calculate_summary_statistics(df, sample_size=100000):
        """Büyük veri setleri için özet istatistikler"""
        if len(df) > sample_size:
            sample_df = df.sample(sample_size, random_state=42)
        else:
            sample_df = df
        
        summary = {
            'total_rows': len(df),
            'total_sales': df['USD_MNF'].sum() if 'USD_MNF' in df.columns else 0,
            'avg_price': df['Price_Per_Unit'].mean() if 'Price_Per_Unit' in df.columns else 0,
            'unique_countries': df['Country'].nunique() if 'Country' in df.columns else 0,
            'unique_molecules': df['Molecule'].nunique() if 'Molecule' in df.columns else 0,
            'unique_companies': df['Corporation'].nunique() if 'Corporation' in df.columns else 0,
            'data_period': f"{df['Year'].min() if 'Year' in df.columns else 'N/A'} - {df['Year'].max() if 'Year' in df.columns else 'N/A'}"
        }
        
        return summary
    
    @staticmethod
    def calculate_market_share(df, group_by='Corporation'):
        """Pazar payı hesaplama (optimize edilmiş)"""
        try:
            total_sales = df['USD_MNF'].sum()
            
            if group_by in df.columns:
                market_share = df.groupby(group_by)['USD_MNF'].sum() / total_sales * 100
                market_share = market_share.sort_values(ascending=False).head(20)
                return market_share
            return pd.Series()
        except:
            return pd.Series()
    
    @staticmethod
    def calculate_growth_rates(df, date_col='Period', value_col='USD_MNF'):
        """Büyüme oranları hesaplama"""
        try:
            if date_col not in df.columns:
                return pd.Series()
            
            # Dönem bazlı toplam satışlar
            period_sales = df.groupby(date_col)[value_col].sum().sort_index()
            
            # Çeyreklik büyüme
            if len(period_sales) > 1:
                qoq_growth = period_sales.pct_change() * 100
                return qoq_growth
            return pd.Series()
        except:
            return pd.Series()
    
    @staticmethod
    def identify_top_performers(df, metric='USD_MNF', top_n=10):
        """En iyi performans gösterenleri tespit et"""
        try:
            if metric not in df.columns:
                return pd.DataFrame()
            
            # Molekül bazlı
            top_molecules = df.groupby('Molecule')[metric].sum().nlargest(top_n)
            
            # Şirket bazlı
            top_companies = df.groupby('Corporation')[metric].sum().nlargest(top_n)
            
            # Ülke bazlı
            top_countries = df.groupby('Country')[metric].sum().nlargest(top_n)
            
            return {
                'molecules': top_molecules,
                'companies': top_companies,
                'countries': top_countries
            }
        except:
            return {}
    
    @staticmethod
    def detect_price_changes(df, window=4):
        """Fiyat değişimlerini tespit et"""
        try:
            if 'Price_Per_Unit' not in df.columns:
                return pd.DataFrame()
            
            # Grup bazında fiyat değişimleri
            price_changes = df.groupby(['Molecule', 'Country'])['Price_Per_Unit'].agg(['mean', 'std', 'count'])
            price_changes['cv'] = price_changes['std'] / price_changes['mean'] * 100  # Değişim katsayısı
            
            # Yüksek volatiliteli ürünler
            high_volatility = price_changes[price_changes['cv'] > 30].sort_values('cv', ascending=False)
            
            return high_volatility.head(20)
        except:
            return pd.DataFrame()

# ================================================
# 4. GÖRSELLEŞTİRME FONKSİYONLARI (OPTİMİZE)
# ================================================

class OptimizedVisualizations:
    """Büyük veri setleri için optimize edilmiş görselleştirmeler"""
    
    @staticmethod
    def create_sales_trend_chart(df, time_col='Period', sample_size=50000):
        """Satış trend grafiği (optimize edilmiş)"""
        try:
            if time_col not in df.columns:
                return None
            
            # Veriyi örnekle
            if len(df) > sample_size:
                plot_df = df.sample(sample_size, random_state=42)
            else:
                plot_df = df
            
            # Zaman bazlı toplam satışlar
            if 'USD_MNF' in plot_df.columns:
                time_sales = plot_df.groupby(time_col)['USD_MNF'].sum().reset_index()
                
                fig = px.line(time_sales, x=time_col, y='USD_MNF',
                             title='Satış Trendleri',
                             labels={'USD_MNF': 'Satış (USD)', time_col: 'Dönem'})
                
                fig.update_layout(height=400)
                return fig
            return None
        except Exception as e:
            st.warning(f"Grafik oluşturulamadı: {str(e)}")
            return None
    
    @staticmethod
    def create_market_share_chart(market_share_data, title="Pazar Payı Dağılımı"):
        """Pazar payı grafiği"""
        try:
            if len(market_share_data) == 0:
                return None
            
            market_share_df = market_share_data.reset_index()
            market_share_df.columns = ['Category', 'Share']
            
            fig = px.bar(market_share_df.head(10), x='Category', y='Share',
                        title=title,
                        labels={'Share': 'Pazar Payı (%)', 'Category': ''})
            
            fig.update_layout(height=400, xaxis_tickangle=-45)
            return fig
        except:
            return None
    
    @staticmethod
    def create_geographic_distribution(df, location_col='Country', value_col='USD_MNF'):
        """Coğrafi dağılım haritası"""
        try:
            if location_col not in df.columns or value_col not in df.columns:
                return None
            
            # Ülke bazlı toplam satışlar
            geo_data = df.groupby(location_col)[value_col].sum().reset_index()
            
            fig = px.choropleth(geo_data,
                               locations=location_col,
                               locationmode='country names',
                               color=value_col,
                               hover_name=location_col,
                               title='Coğrafi Satış Dağılımı',
                               color_continuous_scale='Blues')
            
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
            return fig
        except:
            return None
    
    @staticmethod
    def create_top_performers_chart(top_data, category, title_suffix):
        """En iyi performans gösterenler grafiği"""
        try:
            if len(top_data) == 0:
                return None
            
            top_df = top_data.reset_index()
            top_df.columns = [category, 'Value']
            
            fig = px.bar(top_df.head(10), x=category, y='Value',
                        title=f'Top 10 {title_suffix}',
                        labels={'Value': 'Satış (USD)'})
            
            fig.update_layout(height=400, xaxis_tickangle=-45)
            return fig
        except:
            return None

# ================================================
# 5. ANA UYGULAMA
# ================================================

def main():
    st.markdown('<h1 class="main-title">💊 GLOBAL İLAÇ PAZARI ANALİZ PLATFORMU</h1>', 
                unsafe_allow_html=True)
    
    # ================================================
    # VERİ YÜKLEME BÖLÜMÜ
    # ================================================
    st.sidebar.markdown("## 📁 VERİ YÜKLEME")
    
    uploaded_file = st.sidebar.file_uploader(
        "Excel dosyası yükleyin",
        type=['xlsx', 'xls', 'csv'],
        help="Max 500MB. .xlsx, .xls veya .csv formatında olmalıdır."
    )
    
    if uploaded_file is None:
        st.info("👈 Lütfen sol taraftan Excel dosyası yükleyin")
        st.stop()
    
    # Yükleme seçenekleri
    with st.sidebar.expander("Yükleme Ayarları", expanded=True):
        use_sample = st.checkbox("Örneklem kullan (hız için)", value=True)
        sample_size = st.number_input("Örneklem büyüklüğü", 
                                    min_value=1000, 
                                    max_value=1000000,
                                    value=50000,
                                    step=1000) if use_sample else None
    
    # Veriyi yükle
    with st.spinner("Veri yükleniyor..."):
        if use_sample and sample_size:
            df = load_excel_data(uploaded_file, sample_size=sample_size)
        else:
            df = load_excel_data(uploaded_file)
    
    if df is None or df.empty:
        st.error("Veri yüklenemedi veya boş dosya!")
        st.stop()
    
    # Veri bilgilerini göster
    st.sidebar.markdown(f"**Veri Bilgisi:**")
    st.sidebar.markdown(f"- Toplam Satır: {len(df):,}")
    st.sidebar.markdown(f"- Sütun Sayısı: {len(df.columns)}")
    
    # Sütunları göster
    with st.sidebar.expander("Sütunları Gör", expanded=False):
        for col in df.columns:
            st.write(f"**{col}** ({df[col].dtype})")
    
    # ================================================
    # FİLTRELEME PANELİ
    # ================================================
    st.sidebar.markdown("## 🔍 FİLTRELER")
    
    # Dinamik filtreler oluştur
    filter_options = {}
    
    with st.sidebar.expander("Temel Filtreler", expanded=True):
        # Kategorik sütunlar için filtreler
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols[:5]:  # İlk 5 kategorik sütun için filtre
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 50:  # Çok fazla değer yoksa
                selected = st.multiselect(
                    f"{col}",
                    options=sorted(unique_vals),
                    default=[]
                )
                if selected:
                    filter_options[col] = selected
    
    with st.sidebar.expander("Sayısal Filtreler", expanded=False):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols[:3]:  # İlk 3 sayısal sütun için filtre
            if df[col].notna().any():
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                
                if not pd.isna(min_val) and not pd.isna(max_val):
                    selected_range = st.slider(
                        f"{col} aralığı",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )
                    filter_options[col] = selected_range
    
    # Filtreleri uygula
    filtered_df = df.copy()
    
    for col, value in filter_options.items():
        if col in filtered_df.columns:
            if isinstance(value, tuple):  # Aralık filtresi
                min_val, max_val = value
                filtered_df = filtered_df[(filtered_df[col] >= min_val) & 
                                         (filtered_df[col] <= max_val)]
            else:  # Liste filtresi
                filtered_df = filtered_df[filtered_df[col].isin(value)]
    
    st.sidebar.markdown(f"**Filtrelenmiş:** {len(filtered_df):,} satır")
    
    # ================================================
    # KPI PANELİ
    # ================================================
    st.markdown("## 📊 ÖZET GÖSTERGELER")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = filtered_df['USD_MNF'].sum() if 'USD_MNF' in filtered_df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">TOPLAM SATIŞ</div>
            <div class="metric-value">${total_sales/1e6:.1f}M</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_price = filtered_df['Price_Per_Unit'].mean() if 'Price_Per_Unit' in filtered_df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ORT. FİYAT</div>
            <div class="metric-value">${avg_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        molecule_count = filtered_df['Molecule'].nunique() if 'Molecule' in filtered_df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MOLEKÜL</div>
            <div class="metric-value">{molecule_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        company_count = filtered_df['Corporation'].nunique() if 'Corporation' in filtered_df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ŞİRKET</div>
            <div class="metric-value">{company_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ================================================
    # ANALİZ TABLARI
    # ================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 GENEL BAKIŞ", 
        "🏆 TOP PERFORMERS", 
        "📊 DETAYLI ANALİZ", 
        "📥 RAPORLAR"
    ])
    
    # TAB 1: GENEL BAKIŞ
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Satış trend grafiği
            fig1 = OptimizedVisualizations.create_sales_trend_chart(filtered_df)
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
            
            # Coğrafi dağılım
            if 'Country' in filtered_df.columns:
                fig2 = OptimizedVisualizations.create_geographic_distribution(filtered_df)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Pazar payı
            market_share = OptimizedPharmaAnalytics.calculate_market_share(filtered_df, 'Corporation')
            if len(market_share) > 0:
                fig3 = OptimizedVisualizations.create_market_share_chart(market_share, "Şirket Pazar Payları")
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)
            
            # Molekül pazar payı
            molecule_share = OptimizedPharmaAnalytics.calculate_market_share(filtered_df, 'Molecule')
            if len(molecule_share) > 0:
                fig4 = OptimizedVisualizations.create_market_share_chart(molecule_share.head(10), "Molekül Pazar Payları")
                if fig4:
                    st.plotly_chart(fig4, use_container_width=True)
    
    # TAB 2: TOP PERFORMERS
    with tab2:
        # En iyi performans gösterenleri hesapla
        top_performers = OptimizedPharmaAnalytics.identify_top_performers(filtered_df)
        
        if top_performers:
            col1, col2 = st.columns(2)
            
            with col1:
                # En çok satan moleküller
                if 'molecules' in top_performers:
                    fig1 = OptimizedVisualizations.create_top_performers_chart(
                        top_performers['molecules'], 
                        'Molecule', 
                        'Moleküller'
                    )
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)
                
                # En çok satan ülkeler
                if 'countries' in top_performers:
                    fig3 = OptimizedVisualizations.create_top_performers_chart(
                        top_performers['countries'], 
                        'Country', 
                        'Ülkeler'
                    )
                    if fig3:
                        st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                # En çok satan şirketler
                if 'companies' in top_performers:
                    fig2 = OptimizedVisualizations.create_top_performers_chart(
                        top_performers['companies'], 
                        'Corporation', 
                        'Şirketler'
                    )
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True)
                
                # Fiyat volatilitesi
                price_changes = OptimizedPharmaAnalytics.detect_price_changes(filtered_df)
                if not price_changes.empty:
                    st.markdown("**📊 Yüksek Fiyat Volatilitesi Olan Ürünler**")
                    st.dataframe(price_changes, use_container_width=True)
    
    # TAB 3: DETAYLI ANALİZ
    with tab3:
        # İnteraktif pivot tablo
        st.markdown("### 🔍 Detaylı Analiz")
        
        analysis_col1, analysis_col2 = st.columns([1, 3])
        
        with analysis_col1:
            # Analiz türü seçimi
            analysis_type = st.selectbox(
                "Analiz Türü",
                ["Satış Trendi", "Fiyat Analizi", "Pazar Payı", "Büyüme Analizi"]
            )
            
            # Gruplama seçimi
            group_options = [col for col in ['Country', 'Corporation', 'Molecule', 'Year', 'Quarter'] 
                           if col in filtered_df.columns]
            
            if group_options:
                group_by = st.multiselect(
                    "Gruplama",
                    options=group_options,
                    default=group_options[:2] if len(group_options) >= 2 else group_options[:1]
                )
        
        with analysis_col2:
            if group_by and 'USD_MNF' in filtered_df.columns:
                # Gruplandırılmış veri
                try:
                    grouped_data = filtered_df.groupby(group_by)['USD_MNF'].agg(['sum', 'mean', 'count']).reset_index()
                    st.dataframe(
                        grouped_data.sort_values('sum', ascending=False).head(50),
                        use_container_width=True,
                        height=400
                    )
                except Exception as e:
                    st.warning(f"Gruplama hatası: {str(e)}")
        
        # Veri önizleme
        st.markdown("### 👁️ Veri Önizleme")
        
        preview_cols = st.columns(3)
        with preview_cols[0]:
            rows_to_show = st.slider("Gösterilecek satır", 10, 1000, 100)
        
        st.dataframe(filtered_df.head(rows_to_show), use_container_width=True)
    
    # TAB 4: RAPORLAR
    with tab4:
        st.markdown("### 📥 Rapor Oluşturma")
        
        report_col1, report_col2 = st.columns(2)
        
        with report_col1:
            report_type = st.selectbox(
                "Rapor Tipi",
                ["Özet Rapor", "Detaylı Analiz", "Pazar Payı Raporu", "Performans Raporu"]
            )
            
            include_charts = st.checkbox("Grafikleri ekle", value=True)
            include_data = st.checkbox("Ham veriyi ekle", value=False)
        
        with report_col2:
            # Rapor önizleme
            summary_stats = OptimizedPharmaAnalytics.calculate_summary_statistics(filtered_df)
            
            st.markdown("**Rapor Özeti:**")
            for key, value in summary_stats.items():
                st.write(f"- {key.replace('_', ' ').title()}: {value}")
        
        # İndirme butonları
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV olarak indir
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 CSV İndir",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel olarak indir
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Data')
                writer.save()
            
            st.download_button(
                label="📊 Excel İndir",
                data=output.getvalue(),
                file_name="filtered_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            # Özet rapor indir
            report_data = {
                'generated_date': datetime.now().isoformat(),
                'filters_applied': filter_options,
                'summary_statistics': summary_stats,
                'data_sample': filtered_df.head(100).to_dict('records') if include_data else []
            }
            
            st.download_button(
                label="📄 JSON Rapor İndir",
                data=json.dumps(report_data, indent=2, ensure_ascii=False),
                file_name="pharma_report.json",
                mime="application/json"
            )
    
    # ================================================
    # FOOTER
    # ================================================
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
        <strong>Global Pharma Analytics Platform</strong> | 
        Veri: {len(df):,} satır | 
        Son güncelleme: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        </div>
        """,
        unsafe_allow_html=True
    )

# ================================================
# 6. UYGULAMA BAŞLATMA
# ================================================
if __name__ == "__main__":
    # Garbage collection
    gc.collect()
    
    # Uygulamayı çalıştır
    main()
