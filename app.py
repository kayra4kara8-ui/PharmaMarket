# app.py - Gelişmiş İlaç Pazarı Dashboard (ML Modelleri Dahil)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML ve İstatistik Kütüphaneleri
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from scipy.optimize import curve_fit

# Coğrafi analiz için
import pycountry
import geopandas as gpd

# Yardımcı araçlar
from datetime import datetime, timedelta
import json
from io import BytesIO
import time
import gc
import traceback
from typing import Dict, List, Optional, Tuple, Any
import math

# ================================================
# 1. PROFESYONEL KONFİGÜRASYON VE STİL AYARLARI
# ================================================
st.set_page_config(
    page_title="PharmaIntelligence Pro | Enterprise Pharma Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://pharmaintelligence.com/enterprise-support',
        'Report a bug': "https://pharmaintelligence.com/enterprise-bug-report",
        'About': """
        ### PharmaIntelligence Enterprise v6.0
        • International Product Analytics
        • Predictive Modeling with ML
        • Real-time Market Intelligence
        • Advanced Segmentation
        • Automated Reporting
        • Machine Learning Integration
        © 2024 PharmaIntelligence Inc. All Rights Reserved
        """
    }
)

# Kapsamlı CSS Stilleri
PROFESSIONAL_CSS = """
<style>
    /* Ana tema değişkenleri */
    :root {
        --primary-dark: #0c1a32;
        --secondary-dark: #14274e;
        --accent-blue: #2d7dd2;
        --accent-blue-light: #4a9fe3;
        --accent-blue-dark: #1a5fa0;
        --accent-cyan: #2acaea;
        --accent-teal: #30c9c9;
        --success: #2dd2a3;
        --warning: #f2c94c;
        --danger: #eb5757;
        --info: #2d7dd2;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        --bg-primary: #0c1a32;
        --bg-secondary: #14274e;
        --bg-card: #1e3a5f;
        --bg-hover: #2d4a7a;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--primary-dark), var(--secondary-dark));
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Metrik kartları */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
    }
    
    /* Custom sınıflar */
    .section-title {
        font-size: 1.8rem;
        color: var(--text-primary);
        font-weight: 800;
        margin: 2.5rem 0 1.5rem 0;
        padding-left: 1rem;
        border-left: 5px solid var(--accent-blue);
        background: linear-gradient(90deg, rgba(45, 125, 210, 0.1), transparent);
        padding: 1rem;
        border-radius: 12px;
    }
    
    .subsection-title {
        font-size: 1.4rem;
        color: var(--text-primary);
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--bg-hover);
    }
    
    .ml-card {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--bg-hover);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .ml-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        border-color: var(--accent-blue);
    }
    
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .alert-success {
        background: linear-gradient(135deg, rgba(45, 210, 163, 0.2), rgba(45, 210, 163, 0.1));
        border-left: 5px solid var(--success);
        color: var(--text-primary);
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(242, 201, 76, 0.2), rgba(242, 201, 76, 0.1));
        border-left: 5px solid var(--warning);
        color: var(--text-primary);
    }
    
    .alert-danger {
        background: linear-gradient(135deg, rgba(235, 87, 87, 0.2), rgba(235, 87, 87, 0.1));
        border-left: 5px solid var(--danger);
        color: var(--text-primary);
    }
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ================================================
# 2. VERİ YÖNETİM SINIFI
# ================================================

class DataManager:
    """Veri yükleme ve ön işleme sınıfı"""
    
    def __init__(self):
        self.df = None
        self.df_long = None
        
    def load_data(self, uploaded_file):
        """Excel/CSV dosyasını yükle"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            st.success(f"✅ Veri yüklendi: {len(df)} satır, {len(df.columns)} sütun")
            return df
        except Exception as e:
            st.error(f"❌ Dosya yükleme hatası: {str(e)}")
            return None
    
    def preprocess_data(self, df):
        """Veriyi ön işleme - Wide-to-Long dönüşümü"""
        try:
            # Sütun isimlerini temizle
            df.columns = [str(col).strip() for col in df.columns]
            
            # Wide formatı tespit et (MAT, USD, Unit gibi sütunlar)
            metric_columns = []
            for col in df.columns:
                if any(keyword in str(col) for keyword in ['MAT', 'USD', 'Unit', 'Price', 'Growth', 'Q']):
                    metric_columns.append(col)
            
            if not metric_columns:
                st.warning("📊 Zaman serisi sütunları bulunamadı. Veri zaten long format olabilir.")
                return df, None
            
            # ID sütunlarını belirle
            id_cols = [col for col in df.columns if col not in metric_columns]
            
            st.info(f"🔍 {len(metric_columns)} metrik sütunu bulundu. Wide-to-Long dönüşümü yapılıyor...")
            
            # Melt işlemi
            df_long = pd.melt(
                df,
                id_vars=id_cols,
                value_vars=metric_columns,
                var_name='Metric_Period',
                value_name='Value'
            )
            
            # Metric_Period sütununu ayrıştır
            df_long[['Metric', 'Year', 'Quarter']] = df_long['Metric_Period'].str.extract(
                r'(\w+)\s+(Q\d+)?\s*(\d{4})'
            )
            
            # NaN değerleri temizle
            df_long = df_long.dropna(subset=['Value'])
            
            # Ülke normalizasyonu
            if 'Country' in df_long.columns:
                df_long = self._normalize_country_names(df_long)
            
            # Feature engineering
            df_long = self._create_features(df_long)
            
            st.success(f"✅ Ön işleme tamamlandı. Long format: {len(df_long)} satır")
            
            return df, df_long
            
        except Exception as e:
            st.error(f"❌ Ön işleme hatası: {str(e)}")
            return df, None
    
    def _normalize_country_names(self, df):
        """Ülke isimlerini standartlaştır"""
        country_mapping = {
            'USA': 'United States',
            'US': 'United States',
            'U.S.A': 'United States',
            'United States of America': 'United States',
            'UK': 'United Kingdom',
            'U.K': 'United Kingdom',
            'UAE': 'United Arab Emirates',
            'S. Korea': 'South Korea',
            'South Korea': 'Korea, Republic of',
            'Russia': 'Russian Federation',
            'Iran': 'Iran, Islamic Republic of',
            'Vietnam': 'Viet Nam',
            'Syria': 'Syrian Arab Republic',
            'Laos': 'Lao People\'s Democratic Republic',
            'Bolivia': 'Bolivia, Plurinational State of',
            'Venezuela': 'Venezuela, Bolivarian Republic of',
            'Tanzania': 'Tanzania, United Republic of',
            'Moldova': 'Moldova, Republic of',
            'Macedonia': 'North Macedonia',
            'Turkey': 'Türkiye'
        }
        
        if 'Country' in df.columns:
            df['Country_Normalized'] = df['Country'].replace(country_mapping)
            
            # Pycountry ile ISO kodlarını al
            iso_codes = []
            for country in df['Country_Normalized']:
                try:
                    country_obj = pycountry.countries.search_fuzzy(str(country))[0]
                    iso_codes.append(country_obj.alpha_3)
                except:
                    iso_codes.append(None)
            
            df['ISO_Code'] = iso_codes
        
        return df
    
    def _create_features(self, df):
        """Yeni özellikler oluştur"""
        # Yıl ve çeyrek bazında gruplama
        if 'Year' in df.columns and 'Value' in df.columns:
            year_col = df['Year'].astype(str)
            
            # YoY büyümesi hesapla
            df_sorted = df.sort_values(['Year', 'Quarter'])
            df['YoY_Growth'] = df.groupby('Metric')['Value'].pct_change() * 100
            
            # Hareketli ortalamalar
            df['MA_3'] = df.groupby('Metric')['Value'].rolling(window=3, min_periods=1).mean().values
            df['MA_6'] = df.groupby('Metric')['Value'].rolling(window=6, min_periods=1).mean().values
            
            # Volatilite
            df['Volatility'] = df.groupby('Metric')['Value'].rolling(window=4, min_periods=1).std().values
            
            # Fiyat varyansı (eğer fiyat metrikleri varsa)
            if 'Price' in df['Metric'].values:
                price_df = df[df['Metric'].str.contains('Price', case=False, na=False)]
                if not price_df.empty:
                    price_variance = price_df.groupby('Country_Normalized')['Value'].var()
                    df['Price_Variance'] = df['Country_Normalized'].map(price_variance)
        
        return df
    
    def get_wide_format(self, df_long):
        """Long formatı wide formata çevir"""
        try:
            # Pivot tablo oluştur
            df_wide = df_long.pivot_table(
                index=['Country_Normalized', 'Corporation', 'Molecule'],
                columns=['Metric', 'Year'],
                values='Value',
                aggfunc='mean'
            )
            
            # Sütun isimlerini düzelt
            df_wide.columns = [f"{col[0]}_{col[1]}" for col in df_wide.columns]
            df_wide = df_wide.reset_index()
            
            return df_wide
        except:
            return None

# ================================================
# 3. GÖRSELLEŞTİRME SINIFI
# ================================================

class Visualizer:
    """Görselleştirme sınıfı"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Bold
    
    def create_metric_cards(self, df, df_long):
        """Metrik kartları oluştur"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = df_long[df_long['Metric'].str.contains('USD', case=False, na=False)]['Value'].sum()
            st.metric("💰 Toplam Satış", f"${total_sales/1e9:.2f}B")
        
        with col2:
            if 'YoY_Growth' in df_long.columns:
                avg_growth = df_long['YoY_Growth'].mean()
                st.metric("📈 Ort. Büyüme", f"%{avg_growth:.1f}")
        
        with col3:
            unique_countries = df_long['Country_Normalized'].nunique()
            st.metric("🌍 Ülke Sayısı", f"{unique_countries}")
        
        with col4:
            unique_molecules = df_long['Molecule'].nunique()
            st.metric("💊 Molekül Çeşitliliği", f"{unique_molecules}")
    
    def create_geographic_map(self, df_long, metric_type='USD'):
        """Coğrafi harita oluştur"""
        try:
            # Metrik tipine göre filtrele
            metric_df = df_long[df_long['Metric'].str.contains(metric_type, case=False, na=False)]
            
            if metric_df.empty:
                st.warning(f"❌ '{metric_type}' metrik tipinde veri bulunamadı.")
                return None
            
            # Ülke bazında toplam değer
            country_data = metric_df.groupby('Country_Normalized')['Value'].sum().reset_index()
            
            # Dünya haritası oluştur
            fig = px.choropleth(
                country_data,
                locations='Country_Normalized',
                locationmode='country names',
                color='Value',
                hover_name='Country_Normalized',
                hover_data={'Value': ':.2f'},
                color_continuous_scale='Viridis',
                title=f'{metric_type} Bazında Coğrafi Dağılım',
                projection='natural earth'
            )
            
            fig.update_layout(
                height=600,
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='natural earth',
                    bgcolor='rgba(0,0,0,0)',
                    landcolor='lightgray'
                ),
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Harita oluşturma hatası: {str(e)}")
            return None
    
    def create_competition_analysis(self, df_long):
        """Rekabet analizi grafikleri"""
        try:
            # Şirket bazında satış analizi
            corp_data = df_long[
                df_long['Metric'].str.contains('USD', case=False, na=False)
            ].groupby('Corporation')['Value'].sum().reset_index()
            
            corp_data = corp_data.sort_values('Value', ascending=False)
            
            # Pareto analizi (80/20 kuralı)
            corp_data['Cumulative_Sum'] = corp_data['Value'].cumsum()
            corp_data['Cumulative_Percentage'] = corp_data['Cumulative_Sum'] / corp_data['Value'].sum() * 100
            corp_data['Pareto'] = np.where(corp_data['Cumulative_Percentage'] <= 80, 'Top %80', 'Bottom %20')
            
            # Grafik oluştur
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Şirket Bazında Satışlar', 'Pareto Analizi (80/20 Kuralı)'),
                vertical_spacing=0.15
            )
            
            # Bar chart
            fig.add_trace(
                go.Bar(
                    x=corp_data['Corporation'][:15],
                    y=corp_data['Value'][:15],
                    name='Satış',
                    marker_color=self.color_palette[0]
                ),
                row=1, col=1
            )
            
            # Pareto çizgisi
            fig.add_trace(
                go.Scatter(
                    x=corp_data['Corporation'][:15],
                    y=corp_data['Cumulative_Percentage'][:15],
                    name='Kümülatif %',
                    yaxis='y2',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=800,
                showlegend=True,
                xaxis=dict(tickangle=45),
                xaxis2=dict(tickangle=45),
                yaxis2=dict(
                    title='Kümülatif Yüzde (%)',
                    range=[0, 100]
                )
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Rekabet analizi hatası: {str(e)}")
            return None
    
    def create_molecule_analysis(self, df_long):
        """Molekül analizi grafikleri"""
        try:
            # Molekül bazında analiz
            mol_data = df_long[
                df_long['Metric'].str.contains('USD', case=False, na=False)
            ].groupby('Molecule')['Value'].sum().reset_index()
            
            mol_data = mol_data.sort_values('Value', ascending=False).head(20)
            
            # Fiyat-hacim analizi
            if 'Price' in df_long['Metric'].values and 'Unit' in df_long['Metric'].values:
                price_data = df_long[df_long['Metric'].str.contains('Price', case=False, na=False)]
                unit_data = df_long[df_long['Metric'].str.contains('Unit', case=False, na=False)]
                
                # Molekül bazında ortalama fiyat ve birim
                avg_price = price_data.groupby('Molecule')['Value'].mean()
                total_units = unit_data.groupby('Molecule')['Value'].sum()
                
                price_volume_df = pd.DataFrame({
                    'Avg_Price': avg_price,
                    'Total_Units': total_units
                }).dropna()
                
                # Scatter plot
                fig = px.scatter(
                    price_volume_df.head(30),
                    x='Avg_Price',
                    y='Total_Units',
                    size='Total_Units',
                    color='Avg_Price',
                    hover_name=price_volume_df.index,
                    title='Fiyat-Hacim İlişkisi',
                    trendline='ols',
                    trendline_color_override='red'
                )
                
                return fig
            
            return None
            
        except Exception as e:
            st.error(f"Molekül analizi hatası: {str(e)}")
            return None
    
    def create_time_series_analysis(self, df_long):
        """Zaman serisi analizi"""
        try:
            # Zaman serisi verisi hazırla
            ts_data = df_long[
                df_long['Metric'].str.contains('USD', case=False, na=False)
            ].copy()
            
            if 'Year' in ts_data.columns and 'Quarter' in ts_data.columns:
                ts_data['Period'] = ts_data['Year'] + ' ' + ts_data['Quarter']
                ts_data['Date'] = pd.to_datetime(
                    ts_data['Year'] + ' ' + ts_data['Quarter'].str.replace('Q', ''),
                    format='%Y %q'
                )
            
            # Zaman serisi grafiği
            fig = px.line(
                ts_data,
                x='Date',
                y='Value',
                color='Country_Normalized',
                title='Ülke Bazında Zaman Serisi Analizi',
                markers=True
            )
            
            fig.update_layout(
                height=500,
                xaxis_title='Dönem',
                yaxis_title='Satış Değeri (USD)',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Zaman serisi hatası: {str(e)}")
            return None

# ================================================
# 4. MAKİNE ÖĞRENMESİ SINIFI
# ================================================

class MLModelManager:
    """ML model yönetimi sınıfı"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
    
    def prepare_forecasting_data(self, df_long, country=None, molecule=None):
        """Tahminleme için veri hazırla"""
        try:
            # Filtreleme
            forecast_data = df_long[
                df_long['Metric'].str.contains('USD', case=False, na=False)
            ].copy()
            
            if country:
                forecast_data = forecast_data[forecast_data['Country_Normalized'] == country]
            if molecule:
                forecast_data = forecast_data[forecast_data['Molecule'] == molecule]
            
            # Zaman indeksi oluştur
            if 'Year' in forecast_data.columns and 'Quarter' in forecast_data.columns:
                forecast_data = forecast_data.sort_values(['Year', 'Quarter'])
                forecast_data['Time_Index'] = range(len(forecast_data))
                
                X = forecast_data[['Time_Index']].values
                y = forecast_data['Value'].values
                
                return X, y, forecast_data
            
            return None, None, None
            
        except Exception as e:
            st.error(f"Tahmin verisi hazırlama hatası: {str(e)}")
            return None, None, None
    
    def forecast_sales(self, X, y, periods=4):
        """Satış tahmini yap"""
        try:
            # Veriyi train/test olarak ayır
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Random Forest Regressor modeli
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Tahminler
            y_pred = model.predict(X_test)
            
            # Gelecek tahmini
            last_index = X[-1][0]
            future_indices = np.array([[last_index + i] for i in range(1, periods + 1)])
            future_predictions = model.predict(future_indices)
            
            # Model performansı
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results = {
                'model': model,
                'y_test': y_test,
                'y_pred': y_pred,
                'future_predictions': future_predictions,
                'mae': mae,
                'r2': r2,
                'future_indices': future_indices.flatten()
            }
            
            return results
            
        except Exception as e:
            st.error(f"Tahminleme hatası: {str(e)}")
            return None
    
    def perform_clustering(self, df_wide, n_clusters=4):
        """Kümeleme analizi yap"""
        try:
            # Sayısal sütunları seç
            numeric_cols = df_wide.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Kümeleme için yeterli sayısal sütun yok.")
                return None
            
            # Veriyi hazırla
            clustering_data = df_wide[numeric_cols].fillna(0)
            
            # Ölçeklendirme
            scaled_data = self.scaler.fit_transform(clustering_data)
            
            # Elbow method ile optimal küme sayısını bul
            wcss = []
            max_clusters = min(10, len(clustering_data))
            
            for i in range(1, max_clusters):
                kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
                kmeans.fit(scaled_data)
                wcss.append(kmeans.inertia_)
            
            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            
            # PCA ile boyut indirgeme (3D görselleştirme için)
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(scaled_data)
            
            results = {
                'clusters': clusters,
                'centers': kmeans.cluster_centers_,
                'labels': kmeans.labels_,
                'pca_result': pca_result,
                'wcss': wcss,
                'explained_variance': pca.explained_variance_ratio_
            }
            
            return results
            
        except Exception as e:
            st.error(f"Kümeleme hatası: {str(e)}")
            return None
    
    def detect_anomalies(self, df_long, contamination=0.1):
        """Anomali tespiti yap"""
        try:
            # Anomali tespiti için veri hazırla
            anomaly_data = df_long[
                df_long['Metric'].str.contains('USD', case=False, na=False)
            ]['Value'].values.reshape(-1, 1)
            
            # Isolation Forest modeli
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42
            )
            
            anomalies = iso_forest.fit_predict(anomaly_data)
            
            # Anomali skorları
            anomaly_scores = iso_forest.decision_function(anomaly_data)
            
            results = {
                'anomalies': anomalies,
                'scores': anomaly_scores,
                'model': iso_forest,
                'normal_count': np.sum(anomalies == 1),
                'anomaly_count': np.sum(anomalies == -1)
            }
            
            return results
            
        except Exception as e:
            st.error(f"Anomali tespiti hatası: {str(e)}")
            return None

# ================================================
# 5. ARAYÜZ YÖNETİCİSİ
# ================================================

class UIManager:
    """Kullanıcı arayüzü yöneticisi"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.visualizer = Visualizer()
        self.ml_manager = MLModelManager()
        
    def render_sidebar(self):
        """Sidebar'ı oluştur"""
        with st.sidebar:
            st.title("🎛️ Kontrol Paneli")
            
            # Veri yükleme
            st.subheader("📁 Veri Yükleme")
            uploaded_file = st.file_uploader(
                "Excel/CSV dosyası yükleyin",
                type=['xlsx', 'xls', 'csv'],
                help="MAT Q3 2022 USD... formatında sütunlar içermeli"
            )
            
            # Demo verisi butonu
            if st.button("🔄 Demo Verisi Yükle", use_container_width=True):
                self.load_demo_data()
            
            st.divider()
            
            # Simülasyon aracı
            st.subheader("🎯 What-If Simülasyonu")
            price_increase = st.slider(
                "Fiyat Artış Oranı (%)",
                min_value=0,
                max_value=100,
                value=10,
                step=5
            )
            
            if st.button("📊 Simülasyon Çalıştır", use_container_width=True):
                self.run_simulation(price_increase)
            
            st.divider()
            
            # ML ayarları
            st.subheader("🤖 ML Ayarları")
            self.forecast_periods = st.slider(
                "Tahmin Periyodu",
                min_value=1,
                max_value=12,
                value=4,
                step=1
            )
            
            self.n_clusters = st.slider(
                "Küme Sayısı",
                min_value=2,
                max_value=8,
                value=4,
                step=1
            )
            
            st.divider()
            
            # Hakkında
            st.caption("""
            **PharmaIntelligence Pro v6.0**
            
            Kurumsal düzeyde ilaç pazarı analizi platformu.
            ML modelleri ile tahminleme ve segmentasyon.
            """)
            
            return uploaded_file
    
    def load_demo_data(self):
        """Demo verisi oluştur"""
        try:
            # Demo verisi oluştur
            countries = ['United States', 'Germany', 'Japan', 'China', 'United Kingdom', 
                        'France', 'Italy', 'Spain', 'Canada', 'Australia']
            corporations = ['Pfizer', 'Novartis', 'Roche', 'Merck', 'Johnson & Johnson',
                          'GSK', 'Sanofi', 'AbbVie', 'AstraZeneca', 'Bayer']
            molecules = ['Atorvastatin', 'Adalimumab', 'Apixaban', 'Pembrolizumab',
                        'Trastuzumab', 'Bevacizumab', 'Rituximab', 'Insulin Glargine',
                        'Sitagliptin', 'Etanercept']
            
            years = ['2022', '2023', '2024']
            quarters = ['Q1', 'Q2', 'Q3', 'Q4']
            
            data = []
            for country in countries:
                for corp in corporations[:3]:  # Her ülke için 3 şirket
                    for mol in molecules[:2]:  # Her şirket için 2 molekül
                        for year in years:
                            for quarter in quarters:
                                # Rastgele satış değeri
                                base_sales = np.random.uniform(1000000, 10000000)
                                trend = (years.index(year) + 1) * 0.2  # Her yıl %20 trend
                                sales = base_sales * (1 + trend)
                                
                                # Rastgele büyüme ekle
                                sales *= np.random.uniform(0.8, 1.2)
                                
                                data.append({
                                    'Country': country,
                                    'Corporation': corp,
                                    'Molecule': mol,
                                    f'MAT {quarter} {year} USD': sales,
                                    f'MAT {quarter} {year} Units': np.random.randint(1000, 100000),
                                    f'MAT {quarter} {year} Price': np.random.uniform(10, 1000)
                                })
            
            df = pd.DataFrame(data)
            
            # Session state'e kaydet
            st.session_state.raw_data = df
            st.session_state.data_loaded = True
            
            st.success("✅ Demo verisi yüklendi!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Demo verisi oluşturma hatası: {str(e)}")
    
    def run_simulation(self, price_increase):
        """Fiyat artış simülasyonu çalıştır"""
        try:
            if 'df_long' not in st.session_state:
                st.warning("❌ Önce veri yükleyin!")
                return
            
            df_long = st.session_state.df_long
            
            # Fiyat metriklerini bul
            price_metrics = df_long[df_long['Metric'].str.contains('Price', case=False, na=False)].copy()
            
            if price_metrics.empty:
                st.warning("❌ Fiyat metrikleri bulunamadı!")
                return
            
            # Fiyatları artır
            original_total = price_metrics['Value'].sum()
            price_metrics['New_Value'] = price_metrics['Value'] * (1 + price_increase / 100)
            new_total = price_metrics['New_Value'].sum()
            
            # Satış metriklerini bul
            sales_metrics = df_long[df_long['Metric'].str.contains('USD', case=False, na=False)].copy()
            
            if not sales_metrics.empty:
                # Basit elastikiyet hesabı (fiyat %10 artınca satış %5 azalır varsayımı)
                elasticity = -0.5  # Orta elastikiyet
                sales_change = elasticity * (price_increase / 100)
                
                original_sales = sales_metrics['Value'].sum()
                sales_metrics['New_Value'] = sales_metrics['Value'] * (1 + sales_change)
                new_sales = sales_metrics['New_Value'].sum()
                
                # Sonuçları göster
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Fiyat Artışı",
                        f"%{price_increase}",
                        f"%{(new_total - original_total) / original_total * 100:.1f}"
                    )
                
                with col2:
                    st.metric(
                        "Satış Değişimi",
                        f"${new_sales/1e9:.2f}B",
                        f"%{(new_sales - original_sales) / original_sales * 100:.1f}"
                    )
                
                with col3:
                    revenue_change = (new_total * new_sales) - (original_total * original_sales)
                    st.metric(
                        "Gelir Değişimi",
                        f"${revenue_change/1e9:.2f}B",
                        f"%{(revenue_change / (original_total * original_sales)) * 100:.1f}"
                    )
                
                # Grafik
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Orijinal',
                    x=['Fiyat', 'Satış'],
                    y=[original_total/1e6, original_sales/1e9],
                    marker_color='blue'
                ))
                
                fig.add_trace(go.Bar(
                    name='Yeni',
                    x=['Fiyat', 'Satış'],
                    y=[new_total/1e6, new_sales/1e9],
                    marker_color='red'
                ))
                
                fig.update_layout(
                    title='Simülasyon Sonuçları',
                    yaxis_title='Değer',
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Simülasyon hatası: {str(e)}")
    
    def render_tab1_overview(self, df, df_long):
        """Tab 1: Genel Bakış"""
        st.markdown('<h2 class="section-title">🏠 Genel Bakış</h2>', unsafe_allow_html=True)
        
        # Metrik kartları
        self.visualizer.create_metric_cards(df, df_long)
        
        # Hızlı istatistikler
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="subsection-title">📈 Hızlı İstatistikler</h3>', unsafe_allow_html=True)
            
            stats_data = {
                'Metrik': ['Toplam Kayıt', 'Benzersiz Ülke', 'Benzersiz Şirket', 'Benzersiz Molekül', 
                          'Ortalama Satış', 'Maksimum Satış', 'Minimum Satış'],
                'Değer': [
                    len(df_long),
                    df_long['Country_Normalized'].nunique(),
                    df_long['Corporation'].nunique(),
                    df_long['Molecule'].nunique(),
                    f"${df_long['Value'].mean():,.0f}",
                    f"${df_long['Value'].max():,.0f}",
                    f"${df_long['Value'].min():,.0f}"
                ]
            }
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown('<h3 class="subsection-title">📊 Veri Önizleme</h3>', unsafe_allow_html=True)
            
            # Veri formatı seçimi
            data_format = st.radio(
                "Veri Formatı:",
                ["Ham Veri (Wide)", "İşlenmiş Veri (Long)"],
                horizontal=True
            )
            
            if data_format == "Ham Veri (Wide)":
                st.dataframe(df.head(100), use_container_width=True, height=400)
            else:
                st.dataframe(df_long.head(100), use_container_width=True, height=400)
        
        # Trend grafikleri
        st.markdown('<h3 class="subsection-title">📈 Trend Analizi</h3>', unsafe_allow_html=True)
        
        # YoY büyüme grafiği
        if 'YoY_Growth' in df_long.columns:
            growth_data = df_long[['Year', 'YoY_Growth']].dropna()
            
            if not growth_data.empty:
                fig = px.box(
                    growth_data,
                    x='Year',
                    y='YoY_Growth',
                    title='Yıllık Büyüme Dağılımı',
                    color='Year'
                )
                
                fig.update_layout(
                    height=400,
                    yaxis_title='Büyüme Oranı (%)',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_tab2_geographic(self, df_long):
        """Tab 2: Coğrafi Analiz"""
        st.markdown('<h2 class="section-title">🌍 Coğrafi Analiz</h2>', unsafe_allow_html=True)
        
        # Metrik seçimi
        metric_options = ['USD', 'Units', 'Price', 'Growth']
        selected_metric = st.selectbox(
            "Analiz Metriği Seçin:",
            metric_options,
            index=0
        )
        
        # Harita
        map_fig = self.visualizer.create_geographic_map(df_long, selected_metric)
        
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True)
        
        # Ülke bazlı detay analiz
        st.markdown('<h3 class="subsection-title">📊 Ülke Bazlı Performans</h3>', unsafe_allow_html=True)
        
        # En iyi 10 ülke
        country_performance = df_long[
            df_long['Metric'].str.contains('USD', case=False, na=False)
        ].groupby('Country_Normalized')['Value'].agg(['sum', 'mean', 'std']).reset_index()
        
        country_performance = country_performance.sort_values('sum', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = px.bar(
                country_performance.head(15),
                x='Country_Normalized',
                y='sum',
                title='Top 15 Ülke - Toplam Satış',
                color='sum',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                height=400,
                xaxis_title='Ülke',
                yaxis_title='Toplam Satış (USD)',
                xaxis_tickangle=45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot (Büyüme vs Satış)
            if 'YoY_Growth' in df_long.columns:
                growth_by_country = df_long.groupby('Country_Normalized')['YoY_Growth'].mean().reset_index()
                
                merged_data = pd.merge(
                    country_performance,
                    growth_by_country,
                    on='Country_Normalized',
                    how='left'
                )
                
                fig = px.scatter(
                    merged_data.head(20),
                    x='sum',
                    y='YoY_Growth',
                    size='sum',
                    color='Country_Normalized',
                    hover_name='Country_Normalized',
                    title='Satış vs Büyüme',
                    size_max=50
                )
                
                fig.update_layout(
                    height=400,
                    xaxis_title='Toplam Satış (USD)',
                    yaxis_title='Ortalama Büyüme (%)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_tab3_competition(self, df_long):
        """Tab 3: Rekabet Analizi"""
        st.markdown('<h2 class="section-title">🏢 Rekabet Analizi</h2>', unsafe_allow_html=True)
        
        # Rekabet analizi grafikleri
        comp_fig = self.visualizer.create_competition_analysis(df_long)
        
        if comp_fig:
            st.plotly_chart(comp_fig, use_container_width=True)
        
        # Pazar payı analizi
        st.markdown('<h3 class="subsection-title">🎯 Pazar Payı Analizi</h3>', unsafe_allow_html=True)
        
        # Şirket bazında detaylı analiz
        corporation_data = df_long[
            df_long['Metric'].str.contains('USD', case=False, na=False)
        ].groupby('Corporation').agg({
            'Value': ['sum', 'mean', 'count'],
            'Country_Normalized': 'nunique',
            'Molecule': 'nunique'
        }).round(2)
        
        corporation_data.columns = ['Toplam_Satış', 'Ort_Satış', 'Kayıt_Sayısı', 'Ülke_Sayısı', 'Molekül_Sayısı']
        corporation_data = corporation_data.sort_values('Toplam_Satış', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(
                corporation_data.head(20),
                use_container_width=True,
                height=400
            )
        
        with col2:
            # Pazar konsantrasyonu
            total_sales = corporation_data['Toplam_Satış'].sum()
            corporation_data['Pazar_Payı'] = (corporation_data['Toplam_Satış'] / total_sales) * 100
            
            # CR4, CR8 hesaplama
            cr4 = corporation_data.head(4)['Pazar_Payı'].sum()
            cr8 = corporation_data.head(8)['Pazar_Payı'].sum()
            
            st.metric("CR4 (Top 4 Pazar Payı)", f"%{cr4:.1f}")
            st.metric("CR8 (Top 8 Pazar Payı)", f"%{cr8:.1f}")
            st.metric("Herfindahl-Hirschman Index", 
                     f"{sum((corporation_data['Pazar_Payı'] ** 2) / 100):.0f}")
            
            # Pazar yapısı değerlendirmesi
            if cr4 > 60:
                st.markdown('<div class="alert-box alert-warning">⚠️ Yüksek Konsantrasyon: Oligopol Pazar</div>', unsafe_allow_html=True)
            elif cr4 > 40:
                st.markdown('<div class="alert-box alert-success">✅ Orta Konsantrasyon: Rekabetçi Pazar</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-box alert-success">✅ Düşük Konsantrasyon: Tam Rekabet Pazarı</div>', unsafe_allow_html=True)
    
    def render_tab4_molecule(self, df_long):
        """Tab 4: Molekül Analizi"""
        st.markdown('<h2 class="section-title">💊 Molekül Analizi</h2>', unsafe_allow_html=True)
        
        # Molekül analizi grafikleri
        mol_fig = self.visualizer.create_molecule_analysis(df_long)
        
        if mol_fig:
            st.plotly_chart(mol_fig, use_container_width=True)
        
        # Ürün yaşam döngüsü analizi
        st.markdown('<h3 class="subsection-title">📈 Ürün Yaşam Döngüsü</h3>', unsafe_allow_html=True)
        
        # Molekül bazında zaman serisi
        molecule_list = df_long['Molecule'].unique()[:10]  # İlk 10 molekül
        
        selected_molecule = st.selectbox(
            "Molekül Seçin:",
            molecule_list
        )
        
        if selected_molecule:
            mol_timeseries = df_long[
                (df_long['Molecule'] == selected_molecule) &
                (df_long['Metric'].str.contains('USD', case=False, na=False))
            ].copy()
            
            if not mol_timeseries.empty:
                # Zaman serisi grafiği
                fig = px.line(
                    mol_timeseries,
                    x='Year',
                    y='Value',
                    color='Country_Normalized',
                    title=f'{selected_molecule} - Ülke Bazında Satış Trendi',
                    markers=True
                )
                
                fig.update_layout(
                    height=400,
                    xaxis_title='Yıl',
                    yaxis_title='Satış (USD)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Büyüme analizi
                if 'YoY_Growth' in mol_timeseries.columns:
                    growth_stats = mol_timeseries['YoY_Growth'].describe()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Ortalama Büyüme", f"%{growth_stats['mean']:.1f}")
                    
                    with col2:
                        st.metric("Maksimum Büyüme", f"%{growth_stats['max']:.1f}")
                    
                    with col3:
                        st.metric("Minimum Büyüme", f"%{growth_stats['min']:.1f}")
        
        # Molekül karlılık analizi
        st.markdown('<h3 class="subsection-title">💰 Karlılık Analizi</h3>', unsafe_allow_html=True)
        
        # Fiyat ve birim verilerini birleştir
        price_data = df_long[df_long['Metric'].str.contains('Price', case=False, na=False)]
        unit_data = df_long[df_long['Metric'].str.contains('Unit', case=False, na=False)]
        
        if not price_data.empty and not unit_data.empty:
            # Ortalama fiyat ve birim hesapla
            avg_price = price_data.groupby('Molecule')['Value'].mean()
            total_units = unit_data.groupby('Molecule')['Value'].sum()
            
            profitability_df = pd.DataFrame({
                'Ortalama_Fiyat': avg_price,
                'Toplam_Birim': total_units
            }).dropna()
            
            profitability_df['Toplam_Gelir'] = profitability_df['Ortalama_Fiyat'] * profitability_df['Toplam_Birim']
            profitability_df = profitability_df.sort_values('Toplam_Gelir', ascending=False)
            
            # Karlılık matrisi
            fig = px.scatter(
                profitability_df.head(30),
                x='Ortalama_Fiyat',
                y='Toplam_Birim',
                size='Toplam_Gelir',
                color='Toplam_Gelir',
                hover_name=profitability_df.index,
                title='Molekül Karlılık Matrisi',
                log_x=True,
                log_y=True,
                size_max=50
            )
            
            fig.update_layout(
                height=500,
                xaxis_title='Ortalama Fiyat (USD) - Log Scale',
                yaxis_title='Toplam Birim - Log Scale'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_tab5_timeseries(self, df_long):
        """Tab 5: Zaman Serisi Analizi"""
        st.markdown('<h2 class="section-title">📈 Zaman Serisi Analizi</h2>', unsafe_allow_html=True)
        
        # Zaman serisi grafiği
        ts_fig = self.visualizer.create_time_series_analysis(df_long)
        
        if ts_fig:
            st.plotly_chart(ts_fig, use_container_width=True)
        
        # Sezonallik analizi
        st.markdown('<h3 class="subsection-title">📊 Sezonallik ve Trend Analizi</h3>', unsafe_allow_html=True)
        
        # Toplam satış zaman serisi
        total_sales_ts = df_long[
            df_long['Metric'].str.contains('USD', case=False, na=False)
        ].groupby(['Year', 'Quarter'])['Value'].sum().reset_index()
        
        if len(total_sales_ts) > 8:  # Yeterli veri varsa
            # Zaman indeksi oluştur
            total_sales_ts['Period'] = total_sales_ts['Year'] + '-' + total_sales_ts['Quarter']
            total_sales_ts = total_sales_ts.sort_values(['Year', 'Quarter'])
            
            # Sezon ayrıştırma
            try:
                # Zaman serisi ayrıştırma
                decomposition = seasonal_decompose(
                    total_sales_ts.set_index('Period')['Value'],
                    model='additive',
                    period=4  # Çeyreklik veri
                )
                
                # Grafik oluştur
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Orijinal Seri', 'Trend', 'Sezonallik', 'Artık'),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=total_sales_ts['Period'],
                        y=decomposition.observed,
                        name='Orijinal'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=total_sales_ts['Period'],
                        y=decomposition.trend,
                        name='Trend'
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=total_sales_ts['Period'],
                        y=decomposition.seasonal,
                        name='Sezonallik'
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=total_sales_ts['Period'],
                        y=decomposition.resid,
                        name='Artık'
                    ),
                    row=4, col=1
                )
                
                fig.update_layout(
                    height=800,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Sezon ayrıştırma hatası: {str(e)}")
        
        # Korelasyon analizi
        st.markdown('<h3 class="subsection-title">🔗 Korelasyon Analizi</h3>', unsafe_allow_html=True)
        
        # Wide format hazırla
        df_wide = self.data_manager.get_wide_format(df_long)
        
        if df_wide is not None:
            # Sayısal sütunlar için korelasyon matrisi
            numeric_cols = df_wide.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                correlation_matrix = df_wide[numeric_cols].corr()
                
                # Heatmap
                fig = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu',
                    title='Korelasyon Matrisi'
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_tab6_ml_lab(self, df, df_long):
        """Tab 6: ML Laboratuvarı"""
        st.markdown('<h2 class="section-title">🤖 Makine Öğrenmesi Laboratuvarı</h2>', unsafe_allow_html=True)
        
        tab_a, tab_b, tab_c = st.tabs([
            "🔮 Satış Tahmini", 
            "🎯 Kümeleme Analizi", 
            "⚠️ Anomali Tespiti"
        ])
        
        with tab_a:
            self._render_forecasting_tab(df_long)
        
        with tab_b:
            self._render_clustering_tab(df_long)
        
        with tab_c:
            self._render_anomaly_tab(df_long)
    
    def _render_forecasting_tab(self, df_long):
        """Satış tahmini tab'ı"""
        st.markdown('<div class="ml-card">', unsafe_allow_html=True)
        st.markdown('### 🔮 Gelecek Tahmini (Random Forest)')
        
        # Filtreleme seçenekleri
        col1, col2 = st.columns(2)
        
        with col1:
            countries = ['Tümü'] + df_long['Country_Normalized'].unique().tolist()
            selected_country = st.selectbox("Ülke Seçin:", countries)
        
        with col2:
            molecules = ['Tümü'] + df_long['Molecule'].unique().tolist()
            selected_molecule = st.selectbox("Molekül Seçin:", molecules)
        
        if st.button("🚀 Tahminleme Yap", type="primary", use_container_width=True):
            with st.spinner("Tahmin modeli eğitiliyor..."):
                # Veriyi hazırla
                country = None if selected_country == 'Tümü' else selected_country
                molecule = None if selected_molecule == 'Tümü' else selected_molecule
                
                X, y, forecast_data = self.ml_manager.prepare_forecasting_data(
                    df_long, country, molecule
                )
                
                if X is not None and y is not None:
                    # Tahminleme yap
                    results = self.ml_manager.forecast_sales(
                        X, y, periods=self.forecast_periods
                    )
                    
                    if results:
                        # Sonuçları görselleştir
                        fig = go.Figure()
                        
                        # Geçmiş veri
                        fig.add_trace(go.Scatter(
                            x=forecast_data['Year'] + ' ' + forecast_data['Quarter'],
                            y=y,
                            mode='lines+markers',
                            name='Geçmiş Veri',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Test tahminleri
                        test_periods = len(results['y_test'])
                        test_dates = forecast_data['Year'].iloc[-test_periods:] + ' ' + forecast_data['Quarter'].iloc[-test_periods:]
                        
                        fig.add_trace(go.Scatter(
                            x=test_dates,
                            y=results['y_pred'],
                            mode='lines+markers',
                            name='Test Tahminleri',
                            line=dict(color='green', width=2, dash='dash')
                        ))
                        
                        # Gelecek tahminleri
                        future_dates = []
                        last_date = pd.to_datetime(forecast_data['Year'].iloc[-1] + forecast_data['Quarter'].iloc[-1].replace('Q', ''), 
                                                 format='%Y%q')
                        
                        for i in range(1, self.forecast_periods + 1):
                            future_date = last_date + pd.DateOffset(months=3*i)
                            future_dates.append(future_date.strftime('%Y-Q%q'))
                        
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=results['future_predictions'],
                            mode='lines+markers',
                            name=f'{self.forecast_periods} Dönem Tahmini',
                            line=dict(color='red', width=3)
                        ))
                        
                        # Güven aralığı
                        std_dev = np.std(results['future_predictions'])
                        fig.add_trace(go.Scatter(
                            x=future_dates + future_dates[::-1],
                            y=list(results['future_predictions'] + 1.96*std_dev) + 
                              list(results['future_predictions'] - 1.96*std_dev)[::-1],
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='%95 Güven Aralığı'
                        ))
                        
                        fig.update_layout(
                            title='Satış Tahmini ve Güven Aralığı',
                            xaxis_title='Dönem',
                            yaxis_title='Satış (USD)',
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Model performansı
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("MAE (Ortalama Mutlak Hata)", f"${results['mae']:,.0f}")
                        
                        with col2:
                            st.metric("R² Skoru", f"{results['r2']:.3f}")
                        
                        with col3:
                            total_growth = ((results['future_predictions'][-1] - y[-1]) / y[-1]) * 100
                            st.metric("Toplam Büyüme Tahmini", f"%{total_growth:.1f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_clustering_tab(self, df_long):
        """Kümeleme analizi tab'ı"""
        st.markdown('<div class="ml-card">', unsafe_allow_html=True)
        st.markdown('### 🎯 Kümeleme Analizi (K-Means)')
        
        # Wide formatı hazırla
        df_wide = self.data_manager.get_wide_format(df_long)
        
        if df_wide is not None:
            # Kümeleme yap
            clustering_results = self.ml_manager.perform_clustering(
                df_wide, n_clusters=self.n_clusters
            )
            
            if clustering_results:
                # Elbow method grafiği
                fig1 = go.Figure()
                
                fig1.add_trace(go.Scatter(
                    x=list(range(1, len(clustering_results['wcss']) + 1)),
                    y=clustering_results['wcss'],
                    mode='lines+markers',
                    name='WCSS'
                ))
                
                fig1.update_layout(
                    title='Elbow Method - Optimal Küme Sayısı',
                    xaxis_title='Küme Sayısı',
                    yaxis_title='WCSS (Within-Cluster Sum of Squares)',
                    height=400
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                # 3D kümeleme görselleştirmesi
                fig2 = go.Figure()
                
                # Her küme için scatter plot
                for cluster_id in range(self.n_clusters):
                    cluster_mask = clustering_results['clusters'] == cluster_id
                    
                    fig2.add_trace(go.Scatter3d(
                        x=clustering_results['pca_result'][cluster_mask, 0],
                        y=clustering_results['pca_result'][cluster_mask, 1],
                        z=clustering_results['pca_result'][cluster_mask, 2],
                        mode='markers',
                        name=f'Küme {cluster_id}',
                        marker=dict(
                            size=8,
                            color=cluster_id,
                            colorscale='Viridis',
                            opacity=0.8
                        )
                    ))
                
                fig2.update_layout(
                    title='3D Kümeleme Görselleştirmesi (PCA)',
                    scene=dict(
                        xaxis_title=f'PC1 (%{clustering_results["explained_variance"][0]*100:.1f})',
                        yaxis_title=f'PC2 (%{clustering_results["explained_variance"][1]*100:.1f})',
                        zaxis_title=f'PC3 (%{clustering_results["explained_variance"][2]*100:.1f})'
                    ),
                    height=600
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Küme istatistikleri
                st.markdown('##### 📊 Küme İstatistikleri')
                
                # Küme merkezlerini göster
                centers_df = pd.DataFrame(
                    clustering_results['centers'],
                    columns=[f'Özellik_{i}' for i in range(clustering_results['centers'].shape[1])]
                )
                
                st.dataframe(centers_df, use_container_width=True)
                
                # Küme dağılımı
                cluster_counts = pd.Series(clustering_results['clusters']).value_counts().sort_index()
                
                fig3 = px.pie(
                    values=cluster_counts.values,
                    names=[f'Küme {i}' for i in cluster_counts.index],
                    title='Küme Dağılımı'
                )
                
                st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_anomaly_tab(self, df_long):
        """Anomali tespiti tab'ı"""
        st.markdown('<div class="ml-card">', unsafe_allow_html=True)
        st.markdown('### ⚠️ Anomali Tespiti (Isolation Forest)')
        
        contamination = st.slider(
            "Anomali Oranı (Contamination)",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Beklenen anomali oranı"
        )
        
        if st.button("🔍 Anomali Tespit Et", type="primary", use_container_width=True):
            with st.spinner("Anomali tespiti yapılıyor..."):
                anomaly_results = self.ml_manager.detect_anomalies(
                    df_long, contamination=contamination
                )
                
                if anomaly_results:
                    # Anomali dağılımı
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Normal Kayıt", anomaly_results['normal_count'])
                    
                    with col2:
                        st.metric("Anomali Kayıt", anomaly_results['anomaly_count'])
                    
                    # Anomali skorları dağılımı
                    fig1 = px.histogram(
                        x=anomaly_results['scores'],
                        nbins=50,
                        title='Anomali Skorları Dağılımı',
                        color_discrete_sequence=['red']
                    )
                    
                    fig1.add_vline(
                        x=np.percentile(anomaly_results['scores'], contamination * 100),
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Anomali Eşiği"
                    )
                    
                    fig1.update_layout(
                        height=400,
                        xaxis_title='Anomali Skoru',
                        yaxis_title='Frekans'
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Anomalileri göster
                    anomaly_mask = anomaly_results['anomalies'] == -1
                    
                    if 'Year' in df_long.columns and 'Quarter' in df_long.columns:
                        # Zaman serisinde anomaliler
                        anomaly_data = df_long[anomaly_mask].copy()
                        
                        if not anomaly_data.empty:
                            anomaly_data['Period'] = anomaly_data['Year'] + ' ' + anomaly_data['Quarter']
                            
                            fig2 = px.scatter(
                                anomaly_data,
                                x='Period',
                                y='Value',
                                color='Country_Normalized',
                                size='Value',
                                hover_name='Molecule',
                                title='Tespit Edilen Anomaliler',
                                size_max=20
                            )
                            
                            fig2.update_layout(
                                height=500,
                                xaxis_title='Dönem',
                                yaxis_title='Değer',
                                xaxis_tickangle=45
                            )
                            
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # Anomali detayları
                            st.markdown('##### 📋 Anomali Detayları')
                            
                            anomaly_details = anomaly_data[[
                                'Country_Normalized', 'Corporation', 'Molecule',
                                'Year', 'Quarter', 'Value', 'Metric'
                            ]].sort_values('Value', ascending=False)
                            
                            st.dataframe(
                                anomaly_details,
                                use_container_width=True,
                                height=300
                            )
                    
                    # Anomali özellikleri
                    st.markdown('##### 🔍 Anomali Özellikleri')
                    
                    if anomaly_results['anomaly_count'] > 0:
                        normal_data = df_long[~anomaly_mask]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_normal = normal_data['Value'].mean()
                            avg_anomaly = anomaly_data['Value'].mean()
                            st.metric(
                                "Ortalama Değer Farkı",
                                f"${avg_anomaly - avg_normal:,.0f}",
                                f"%{((avg_anomaly - avg_normal) / avg_normal * 100):.1f}"
                            )
                        
                        with col2:
                            std_normal = normal_data['Value'].std()
                            std_anomaly = anomaly_data['Value'].std()
                            st.metric(
                                "Standart Sapma Farkı",
                                f"${std_anomaly - std_normal:,.0f}"
                            )
        
        st.markdown('</div>', unsafe_allow_html=True)

# ================================================
# 6. ANA UYGULAMA
# ================================================

def main():
    """Ana uygulama fonksiyonu"""
    
    # Başlık
    st.title("💊 PharmaIntelligence Pro")
    st.markdown("""
    <div style='color: #cbd5e1; font-size: 1.1rem; margin-bottom: 2rem;'>
    Kurumsal düzeyde ilaç pazarı analiz platformu. ML modelleri ile tahminleme, segmentasyon ve anomali tespiti.
    </div>
    """, unsafe_allow_html=True)
    
    # UI Manager oluştur
    ui_manager = UIManager()
    
    # Sidebar'ı render et
    uploaded_file = ui_manager.render_sidebar()
    
    # Veri yükleme
    if uploaded_file is not None:
        with st.spinner("Veri yükleniyor ve işleniyor..."):
            # Veriyi yükle
            raw_data = ui_manager.data_manager.load_data(uploaded_file)
            
            if raw_data is not None:
                # Veriyi işle
                df, df_long = ui_manager.data_manager.preprocess_data(raw_data)
                
                # Session state'e kaydet
                st.session_state.raw_data = raw_data
                st.session_state.df = df
                st.session_state.df_long = df_long
                st.session_state.data_loaded = True
    
    # Demo verisi yüklendi mi kontrol et
    elif 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Veri yüklendiyse tabları göster
    if st.session_state.get('data_loaded', False):
        df = st.session_state.df
        df_long = st.session_state.df_long
        
        # Tablar
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏠 Genel Bakış",
            "🌍 Coğrafi Analiz",
            "🏢 Rekabet Analizi",
            "💊 Molekül Analizi",
            "📈 Zaman Serisi",
            "🤖 ML Laboratuvarı"
        ])
        
        with tab1:
            ui_manager.render_tab1_overview(df, df_long)
        
        with tab2:
            ui_manager.render_tab2_geographic(df_long)
        
        with tab3:
            ui_manager.render_tab3_competition(df_long)
        
        with tab4:
            ui_manager.render_tab4_molecule(df_long)
        
        with tab5:
            ui_manager.render_tab5_timeseries(df_long)
        
        with tab6:
            ui_manager.render_tab6_ml_lab(df, df_long)
    
    else:
        # Hoşgeldiniz ekranı
        st.markdown("""
        <div style='text-align: center; padding: 5rem 2rem; background: linear-gradient(135deg, rgba(30, 58, 95, 0.5), rgba(20, 39, 78, 0.5)); border-radius: 20px; margin: 2rem 0;'>
            <h2 style='color: #f8fafc; margin-bottom: 1.5rem;'>🚀 PharmaIntelligence Pro'ya Hoşgeldiniz</h2>
            <p style='color: #cbd5e1; font-size: 1.2rem; margin-bottom: 2rem; max-width: 800px; margin-left: auto; margin-right: auto;'>
            İlaç pazarı verilerinizi yükleyin ve gelişmiş analitik özelliklerin kilidini açın.
            </p>
            
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin: 3rem 0;'>
                <div style='background: rgba(30, 58, 95, 0.7); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #2d7dd2;'>
                    <h3 style='color: #f8fafc; margin-bottom: 0.5rem;'>📈 Tahminleme</h3>
                    <p style='color: #cbd5e1;'>Random Forest ile gelecek satış tahminleri</p>
                </div>
                
                <div style='background: rgba(30, 58, 95, 0.7); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #2acaea;'>
                    <h3 style='color: #f8fafc; margin-bottom: 0.5rem;'>🎯 Segmentasyon</h3>
                    <p style='color: #cbd5e1;'>K-Means ile pazar segmentasyonu</p>
                </div>
                
                <div style='background: rgba(30, 58, 95, 0.7); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #2dd2a3;'>
                    <h3 style='color: #f8fafc; margin-bottom: 0.5rem;'>🌍 Coğrafi Analiz</h3>
                    <p style='color: #cbd5e1;'>İnteraktif dünya haritaları</p>
                </div>
            </div>
            
            <div style='margin-top: 3rem;'>
                <p style='color: #f2c94c; font-weight: bold;'>
                ⚡ Başlamak için sol taraftan veri yükleyin veya "Demo Verisi Yükle" butonuna tıklayın.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================================================
# 7. UYGULAMA BAŞLATMA
# ================================================

if __name__ == "__main__":
    try:
        # Bellek optimizasyonu
        gc.enable()
        
        # Session state'i başlat
        if 'raw_data' not in st.session_state:
            st.session_state.raw_data = None
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'df_long' not in st.session_state:
            st.session_state.df_long = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        
        # Uygulamayı başlat
        main()
        
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        st.error("Detaylı hata bilgisi:")
        st.code(traceback.format_exc())
        
        if st.button("🔄 Sayfayı Yenile", use_container_width=True):
            st.rerun()
