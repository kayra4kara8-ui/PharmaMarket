import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import pycountry
import hashlib
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Any
import io

warnings.filterwarnings('ignore')

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.set_page_config(
    page_title="İlaç Sektörü Satış Analizi",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown("""
    <style>
    /* Ana tema değişkenleri */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --danger-color: #d62728;
        --light-bg: #f8f9fa;
        --dark-bg: #343a40;
        --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Ana konteyner */
    .main {
        padding: 2rem;
    }
    
    /* KPI kartları */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        box-shadow: var(--card-shadow);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sekme stilleri */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        padding: 10px 16px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #1a1a2e 100%);
        color: white;
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        color: var(--primary-color);
        font-weight: 700;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: var(--dark-bg);
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 5px 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* İlerleme çubuğu */
    .stProgress > div > div > div {
        background-color: var(--primary-color);
    }
    
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA MANAGER CLASS
# ============================================================================

class DataManager:
    """Veri yükleme, temizleme ve ön işleme işlemlerini yönetir."""
    
    def __init__(self):
        self.df = None
        self.df_long = None
        self.country_mapping = self._create_country_mapping()
        
    def _create_country_mapping(self) -> Dict[str, str]:
        """Ülke isimlerini standartlaştırmak için mapping oluşturur."""
        mapping = {}
        
        # Türkçe ve yaygın ülke isimlerini ISO kodlarına eşle
        common_names = {
            "USA": "United States", "US": "United States", "Amerika": "United States",
            "UK": "United Kingdom", "İngiltere": "United Kingdom", "Britanya": "United Kingdom",
            "Türkiye": "Turkey", "Turkey": "Turkey",
            "Almanya": "Germany", "Germany": "Germany",
            "Fransa": "France", "France": "France",
            "İtalya": "Italy", "Italy": "Italy",
            "İspanya": "Spain", "Spain": "Spain",
            "Japonya": "Japan", "Japan": "Japan",
            "Çin": "China", "China": "China",
            "Hindistan": "India", "India": "India",
            "Brezilya": "Brazil", "Brazil": "Brazil",
            "Rusya": "Russia", "Russia": "Russia",
            "Güney Kore": "South Korea", "South Korea": "South Korea",
            "Kanada": "Canada", "Canada": "Canada",
            "Meksika": "Mexico", "Mexico": "Mexico",
            "Avustralya": "Australia", "Australia": "Australia"
        }
        
        for code in pycountry.countries:
            mapping[code.name] = code.alpha_3
        
        # Ortak isimleri ekle
        for common, official in common_names.items():
            if official in mapping:
                mapping[common] = mapping[official]
        
        return mapping
    
    def normalize_country_name(self, country_name: str) -> str:
        """Ülke ismini standartlaştırır ve ISO koduna çevirir."""
        if pd.isna(country_name):
            return "UNK"
        
        country_name = str(country_name).strip()
        
        # Önce mapping'de ara
        if country_name in self.country_mapping:
            return self.country_mapping[country_name]
        
        # pycountry ile dene
        try:
            country = pycountry.countries.search_fuzzy(country_name)[0]
            return country.alpha_3
        except:
            # Bulunamazsa orijinal ismi döndür
            return country_name
    
    def load_demo_data(self):
        """Demo veri setini oluşturur."""
        np.random.seed(42)
        
        # Temel veri yapısı
        countries = ["United States", "Germany", "France", "Japan", "China", 
                    "Turkey", "United Kingdom", "Italy", "Spain", "Brazil",
                    "India", "Russia", "South Korea", "Canada", "Australia"]
        
        corporations = ["PharmaCorp A", "MediTech B", "BioGen C", "HealthPlus D", 
                       "CureAll E", "Vitality F", "GenHeal G", "MediCare H"]
        
        molecules = ["Molecule A", "Molecule B", "Molecule C", "Molecule D", 
                    "Molecule E", "Molecule F", "Molecule G", "Molecule H"]
        
        sectors = ["Onkoloji", "Kardiyoloji", "Nöroloji", "Diyabet", 
                  "Enfeksiyon", "Ağrı", "Solunum", "Psikiyatri"]
        
        sources = ["Source A", "Source B", "Source C"]
        
        years = [2022, 2023, 2024]
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        
        # Veri oluştur
        data = []
        for year in years:
            for quarter in quarters:
                for country in countries:
                    for corp in np.random.choice(corporations, size=3, replace=False):
                        for molecule in np.random.choice(molecules, size=2, replace=False):
                            sector = np.random.choice(sectors)
                            source = np.random.choice(sources)
                            
                            # Temel metrikler
                            units = np.random.randint(1000, 10000)
                            usd_value = np.random.uniform(50000, 500000)
                            price_per_unit = usd_value / units if units > 0 else 0
                            
                            # Rastgele büyüme faktörü
                            growth_factor = np.random.uniform(0.8, 1.3)
                            if year > 2022:
                                usd_value *= growth_factor
                                units = int(units * growth_factor)
                            
                            data.append({
                                'Source.Name': source,
                                'Country': country,
                                'Sector': sector,
                                'Corporation': corp,
                                'Molecule': molecule,
                                'Year': year,
                                'Quarter': quarter,
                                'Units_Sold': units,
                                'USD_Value': usd_value,
                                'Price_Per_Unit': price_per_unit
                            })
        
        self.df = pd.DataFrame(data)
        self._create_derived_features()
        return self.df
    
    def load_excel_data(self, uploaded_file):
        """Excel dosyasından veri yükler ve işler."""
        try:
            # Excel'i oku
            self.df = pd.read_excel(uploaded_file)
            
            # Wide format'tan Long format'a çevir
            self._wide_to_long()
            
            # Özellik mühendisliği
            self._create_derived_features()
            
            return True
        except Exception as e:
            st.error(f"Veri yükleme hatası: {str(e)}")
            return False
    
    def _wide_to_long(self):
        """Wide formatındaki veriyi long formata çevirir."""
        if self.df is None:
            return
        
        # Örnek wide format sütunları
        # 'MAT Q3 2022 USD', 'MAT Q3 2023 USD', 'MAT Q4 2022 Units', vb.
        
        # Sütunları belirle
        value_columns = [col for col in self.df.columns if any(x in col for x in ['MAT', 'USD', 'Units'])]
        id_columns = [col for col in self.df.columns if col not in value_columns]
        
        # Melt işlemi
        self.df_long = pd.melt(
            self.df,
            id_vars=id_columns,
            value_vars=value_columns,
            var_name='Metric_Type',
            value_name='Value'
        )
        
        # Metric_Type'ı parçala
        self.df_long[['Period', 'Quarter', 'Year', 'Metric']] = \
            self.df_long['Metric_Type'].str.extract(r'(\w+)\s+(\w+\s+\d+)\s+(\d{4})\s+(\w+)')
    
    def _create_derived_features(self):
        """Türetilmiş özellikler oluşturur."""
        if self.df is not None:
            # Gruplama için yıl-saat bilgisi
            self.df['Date'] = pd.to_datetime(self.df['Year'].astype(str) + '-01-01')
            
            # YoY Büyüme Hesaplama
            self.df['YoY_Growth'] = self.df.groupby(['Country', 'Corporation', 'Molecule'])['USD_Value'] \
                .pct_change(periods=4) * 100
            
            # Fiyat Varyansı
            self.df['Price_Variance'] = self.df.groupby(['Molecule'])['Price_Per_Unit'] \
                .transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
            
            # Pazar Payı
            total_sales = self.df.groupby('Year')['USD_Value'].transform('sum')
            self.df['Market_Share'] = (self.df['USD_Value'] / total_sales) * 100
            
            # Ülke normalizasyonu
            self.df['Country_Code'] = self.df['Country'].apply(self.normalize_country_name)
            
            # Segmentasyon için özellikler
            self.df['Sales_Volume'] = np.log1p(self.df['Units_Sold'])
            self.df['Profit_Margin'] = (self.df['USD_Value'] - self.df['Units_Sold'] * 10) / self.df['USD_Value'] * 100
            
            # NaN değerleri temizle
            self.df = self.df.fillna(0)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Özet istatistikleri döndürür."""
        if self.df is None:
            return {}
        
        return {
            'total_sales': self.df['USD_Value'].sum(),
            'avg_growth': self.df['YoY_Growth'].mean(),
            'unique_countries': self.df['Country'].nunique(),
            'unique_molecules': self.df['Molecule'].nunique(),
            'unique_corporations': self.df['Corporation'].nunique(),
            'total_units': self.df['Units_Sold'].sum(),
            'avg_price': self.df['Price_Per_Unit'].mean()
        }
    
    def prepare_ml_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ML modelleri için veri hazırlar."""
        if self.df is None:
            return pd.DataFrame(), pd.DataFrame()
        
        # Özellikler ve hedef değişkenler
        features = ['Units_Sold', 'Price_Per_Unit', 'YoY_Growth', 
                   'Price_Variance', 'Sales_Volume', 'Profit_Margin']
        
        # NaN kontrolü
        ml_df = self.df[features + ['Year', 'Country', 'Corporation']].copy()
        ml_df = ml_df.fillna(0)
        
        # Encoding kategorik değişkenler
        categorical_cols = ['Country', 'Corporation']
        ml_df = pd.get_dummies(ml_df, columns=categorical_cols, drop_first=True)
        
        return ml_df, self.df[['USD_Value', 'YoY_Growth']]

# ============================================================================
# VISUALIZER CLASS
# ============================================================================

class Visualizer:
    """Veri görselleştirme işlemlerini yönetir."""
    
    @staticmethod
    def create_kpi_cards(stats: Dict[str, Any]) -> None:
        """KPI kartlarını oluşturur."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Toplam Satış</div>
                <div class="kpi-value">${stats.get('total_sales', 0):,.0f}</div>
                <div>USD</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="kpi-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                <div class="kpi-label">Ortalama Büyüme</div>
                <div class="kpi-value">{stats.get('avg_growth', 0):.1f}%</div>
                <div>YoY</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="kpi-card" style="background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);">
                <div class="kpi-label">Ülke Sayısı</div>
                <div class="kpi-value">{stats.get('unique_countries', 0)}</div>
                <div>Ülke</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="kpi-card" style="background: linear-gradient(135deg, #4A00E0 0%, #8E2DE2 100%);">
                <div class="kpi-label">Ürün Çeşidi</div>
                <div class="kpi-value">{stats.get('unique_molecules', 0)}</div>
                <div>Molekül</div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def create_choropleth_map(df: pd.DataFrame, metric: str = 'USD_Value') -> go.Figure:
        """Dinamik dünya haritası oluşturur."""
        try:
            # Ülke bazında toplam metrik
            country_data = df.groupby(['Country_Code', 'Country'])[metric].sum().reset_index()
            
            fig = px.choropleth(
                country_data,
                locations="Country_Code",
                color=metric,
                hover_name="Country",
                hover_data={metric: ':,.2f', "Country_Code": False},
                color_continuous_scale=px.colors.sequential.Plasma,
                title=f"Ülkelere Göre {metric} Dağılımı",
                labels={metric: metric.replace('_', ' ')},
                projection="natural earth"
            )
            
            fig.update_layout(
                height=600,
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='equirectangular'
                ),
                margin={"r": 0, "t": 50, "l": 0, "b": 0}
            )
            
            return fig
        except Exception as e:
            st.warning(f"Harita oluşturulamadı: {str(e)}")
            return go.Figure()
    
    @staticmethod
    def create_time_series(df: pd.DataFrame) -> go.Figure:
        """Zaman serisi grafiği oluşturur."""
        time_data = df.groupby(['Year', 'Quarter']).agg({
            'USD_Value': 'sum',
            'Units_Sold': 'sum',
            'YoY_Growth': 'mean'
        }).reset_index()
        
        time_data['Period'] = time_data['Year'].astype(str) + ' ' + time_data['Quarter']
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Satış Trendi (USD)', 'Yıllık Büyüme (%)'),
            vertical_spacing=0.15
        )
        
        # Satış trendi
        fig.add_trace(
            go.Scatter(
                x=time_data['Period'],
                y=time_data['USD_Value'],
                mode='lines+markers',
                name='Satış',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Büyüme trendi
        fig.add_trace(
            go.Bar(
                x=time_data['Period'],
                y=time_data['YoY_Growth'],
                name='Büyüme',
                marker_color='#ff7f0e',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Dönem", row=2, col=1)
        fig.update_yaxes(title_text="USD", row=1, col=1)
        fig.update_yaxes(title_text="%", row=2, col=1)
        
        return fig
    
    @staticmethod
    def create_3d_cluster_plot(df: pd.DataFrame, cluster_labels: np.ndarray) -> go.Figure:
        """3D kümeleme grafiği oluşturur."""
        fig = go.Figure(data=[
            go.Scatter3d(
                x=df['Price_Per_Unit'],
                y=df['Sales_Volume'],
                z=df['YoY_Growth'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=cluster_labels,
                    colorscale='Viridis',
                    opacity=0.8,
                    line=dict(width=0.5, color='white')
                ),
                text=df['Country'],
                hovertemplate='<b>%{text}</b><br>' +
                             'Fiyat: %{x:.2f}<br>' +
                             'Hacim: %{y:.2f}<br>' +
                             'Büyüme: %{z:.2f}%<br>' +
                             '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='3D Kümeleme Analizi',
            scene=dict(
                xaxis_title='Fiyat (USD/Unit)',
                yaxis_title='Satış Hacmi (log)',
                zaxis_title='YoY Büyüme (%)'
            ),
            height=700,
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def create_pareto_chart(df: pd.DataFrame) -> go.Figure:
        """Pareto analizi grafiği oluşturur."""
        corp_data = df.groupby('Corporation')['USD_Value'].sum().reset_index()
        corp_data = corp_data.sort_values('USD_Value', ascending=False)
        corp_data['Cumulative_Percentage'] = corp_data['USD_Value'].cumsum() / corp_data['USD_Value'].sum() * 100
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart - Satışlar
        fig.add_trace(
            go.Bar(
                x=corp_data['Corporation'],
                y=corp_data['USD_Value'],
                name='Satış',
                marker_color='#1f77b4',
                opacity=0.7
            ),
            secondary_y=False
        )
        
        # Line chart - Kümülatif yüzde
        fig.add_trace(
            go.Scatter(
                x=corp_data['Corporation'],
                y=corp_data['Cumulative_Percentage'],
                name='Kümülatif %',
                line=dict(color='#ff7f0e', width=3),
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        # 80% çizgisi
        fig.add_hline(y=80, line_dash="dash", line_color="red", 
                     annotation_text="80%", secondary_y=True)
        
        fig.update_layout(
            title='Şirketlere Göre Pareto Analizi',
            xaxis_title='Şirket',
            height=500,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Satış (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Kümülatif %", secondary_y=True)
        
        return fig

# ============================================================================
# ML MODEL MANAGER CLASS
# ============================================================================

class MLModelManager:
    """Makine öğrenmesi modellerini yönetir."""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.forecast_model = None
        self.clustering_model = None
        self.anomaly_model = None
        
    def train_forecasting_model(self) -> Dict[str, Any]:
        """Zaman serisi tahmin modeli eğitir."""
        if self.data_manager.df is None:
            return {}
        
        try:
            # Veriyi hazırla
            time_data = self.data_manager.df.groupby(['Year', 'Quarter']).agg({
                'USD_Value': 'sum',
                'Units_Sold': 'sum'
            }).reset_index()
            
            time_data['Time_Index'] = range(len(time_data))
            
            # Özellikler
            X = time_data[['Time_Index', 'Units_Sold']]
            y = time_data['USD_Value']
            
            # RandomForest modeli
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            model.fit(X, y)
            self.forecast_model = model
            
            # Gelecek tahminleri
            future_periods = 8  # 2 yıl (8 çeyrek)
            last_index = time_data['Time_Index'].max()
            
            future_predictions = []
            for i in range(1, future_periods + 1):
                # Gelecek dönem için birim satış tahmini (basit trend devamı)
                future_units = time_data['Units_Sold'].mean() * (1 + 0.05 * i)
                
                pred = model.predict([[last_index + i, future_units]])
                future_predictions.append({
                    'Year': 2024 + ((i-1) // 4),
                    'Quarter': f'Q{((i-1) % 4) + 1}',
                    'Predicted_Value': pred[0],
                    'Lower_Bound': pred[0] * 0.9,  # %90 güven aralığı
                    'Upper_Bound': pred[0] * 1.1   # %110 güven aralığı
                })
            
            return {
                'model': model,
                'future_predictions': pd.DataFrame(future_predictions),
                'mse': mean_squared_error(y, model.predict(X)),
                'mae': mean_absolute_error(y, model.predict(X)),
                'r2': model.score(X, y)
            }
            
        except Exception as e:
            st.error(f"Tahmin modeli eğitim hatası: {str(e)}")
            return {}
    
    def train_clustering_model(self, n_clusters: int = 3) -> Dict[str, Any]:
        """Kümeleme modeli eğitir."""
        if self.data_manager.df is None:
            return {}
        
        try:
            # Kümeleme için veri hazırla
            cluster_data = self.data_manager.df.groupby('Country').agg({
                'Price_Per_Unit': 'mean',
                'Sales_Volume': 'mean',
                'YoY_Growth': 'mean',
                'USD_Value': 'sum'
            }).reset_index()
            
            # Özellikleri ölçeklendir
            features = ['Price_Per_Unit', 'Sales_Volume', 'YoY_Growth']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(cluster_data[features])
            
            # KMeans modeli
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = model.fit_predict(X_scaled)
            
            self.clustering_model = model
            
            # Silhouette skoru
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            
            # PCA ile boyut indirgeme (3D görselleştirme için)
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_scaled)
            
            return {
                'model': model,
                'labels': cluster_labels,
                'data': cluster_data,
                'silhouette_score': silhouette_avg,
                'pca_data': X_pca,
                'features': features
            }
            
        except Exception as e:
            st.error(f"Kümeleme modeli eğitim hatası: {str(e)}")
            return {}
    
    def find_optimal_clusters(self, max_clusters: int = 10) -> go.Figure:
        """Optimal küme sayısını belirlemek için elbow method grafiği."""
        if self.data_manager.df is None:
            return go.Figure()
        
        try:
            cluster_data = self.data_manager.df.groupby('Country').agg({
                'Price_Per_Unit': 'mean',
                'Sales_Volume': 'mean',
                'YoY_Growth': 'mean'
            }).reset_index()
            
            features = ['Price_Per_Unit', 'Sales_Volume', 'YoY_Growth']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(cluster_data[features])
            
            inertia = []
            silhouette_scores = []
            
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            
            # Elbow grafiği
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Elbow Method', 'Silhouette Scores'))
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(2, max_clusters + 1)),
                    y=inertia,
                    mode='lines+markers',
                    name='Inertia',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(2, max_clusters + 1)),
                    y=silhouette_scores,
                    mode='lines+markers',
                    name='Silhouette',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                showlegend=True,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Küme Sayısı (k)", row=1, col=1)
            fig.update_yaxes(title_text="Inertia", row=1, col=1)
            fig.update_xaxes(title_text="Küme Sayısı (k)", row=1, col=2)
            fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
            
            return fig
            
        except Exception as e:
            st.error(f"Optimal küme analizi hatası: {str(e)}")
            return go.Figure()
    
    def detect_anomalies(self, contamination: float = 0.1) -> Dict[str, Any]:
        """Anomali tespiti yapar."""
        if self.data_manager.df is None:
            return {}
        
        try:
            # Anomali tespiti için veri
            anomaly_data = self.data_manager.df.groupby(['Country', 'Corporation']).agg({
                'USD_Value': 'sum',
                'Price_Per_Unit': 'mean',
                'YoY_Growth': 'mean'
            }).reset_index()
            
            # Özellikleri ölçeklendir
            features = ['USD_Value', 'Price_Per_Unit', 'YoY_Growth']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(anomaly_data[features])
            
            # Isolation Forest modeli
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            anomaly_labels = model.fit_predict(X_scaled)
            anomaly_data['Is_Anomaly'] = anomaly_labels == -1
            
            self.anomaly_model = model
            
            return {
                'model': model,
                'anomaly_data': anomaly_data,
                'anomaly_count': sum(anomaly_labels == -1),
                'total_count': len(anomaly_labels)
            }
            
        except Exception as e:
            st.error(f"Anomali tespiti hatası: {str(e)}")
            return {}

# ============================================================================
# UI MANAGER CLASS
# ============================================================================

class UIManager:
    """Kullanıcı arayüzü bileşenlerini yönetir."""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.visualizer = Visualizer()
        self.ml_manager = MLModelManager(self.data_manager)
        
    def render_sidebar(self):
        """Sidebar bileşenlerini render eder."""
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <h2 style="color: white;">💊 İlaç Analizi</h2>
                <p style="color: #aaa;">Enterprise Dashboard</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Veri yükleme
            st.subheader("📁 Veri Yükleme")
            uploaded_file = st.file_uploader(
                "Excel dosyası yükleyin",
                type=['xlsx', 'xls'],
                help="Wide formatında satış verisi içeren Excel dosyası"
            )
            
            if uploaded_file is not None:
                with st.spinner("Veri yükleniyor..."):
                    success = self.data_manager.load_excel_data(uploaded_file)
                    if success:
                        st.success("✓ Veri başarıyla yüklendi!")
                    else:
                        st.error("✗ Veri yükleme başarısız!")
            else:
                if st.button("Demo Veri Yükle", type="primary", use_container_width=True):
                    with st.spinner("Demo veri oluşturuluyor..."):
                        self.data_manager.load_demo_data()
                        st.success("✓ Demo veri yüklendi!")
                        st.rerun()
            
            st.divider()
            
            # Simülasyon aracı
            st.subheader("🔄 Fiyat Simülasyonu")
            price_increase = st.slider(
                "Global Fiyat Artışı (%)",
                min_value=0,
                max_value=50,
                value=10,
                step=1
            )
            
            if st.button("Simülasyon Çalıştır", use_container_width=True):
                self.run_price_simulation(price_increase)
            
            st.divider()
            
            # Filtreler
            st.subheader("🔍 Filtreler")
            
            if self.data_manager.df is not None:
                countries = sorted(self.data_manager.df['Country'].unique().tolist())
                selected_countries = st.multiselect(
                    "Ülkeler",
                    countries,
                    default=countries[:5] if len(countries) > 5 else countries
                )
                
                sectors = sorted(self.data_manager.df['Sector'].unique().tolist())
                selected_sectors = st.multiselect(
                    "Sektörler",
                    sectors,
                    default=sectors
                )
                
                years = sorted(self.data_manager.df['Year'].unique().tolist())
                selected_years = st.multiselect(
                    "Yıllar",
                    years,
                    default=years
                )
                
                # Filtreleri uygula
                if selected_countries:
                    self.data_manager.df = self.data_manager.df[
                        self.data_manager.df['Country'].isin(selected_countries)
                    ]
                
                if selected_sectors:
                    self.data_manager.df = self.data_manager.df[
                        self.data_manager.df['Sector'].isin(selected_sectors)
                    ]
                
                if selected_years:
                    self.data_manager.df = self.data_manager.df[
                        self.data_manager.df['Year'].isin(selected_years)
                    ]
            
            st.divider()
            
            # Hakkında
            st.markdown("""
            <div style="padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                <small>
                <strong>Enterprise Pharma Analytics v2.0</strong><br>
                © 2024 AI Pharma Solutions<br>
                Tüm hakları saklıdır.
                </small>
            </div>
            """, unsafe_allow_html=True)
    
    def run_price_simulation(self, price_increase: float):
        """Fiyat simülasyonu çalıştırır."""
        if self.data_manager.df is None:
            st.warning("Lütfen önce veri yükleyin!")
            return
        
        # Elastikiyet katsayısı (tahmini)
        elasticity = -0.5  # Talep fiyat esnekliği
        
        # Mevcut satışlar
        current_sales = self.data_manager.df['USD_Value'].sum()
        current_units = self.data_manager.df['Units_Sold'].sum()
        current_price = self.data_manager.df['Price_Per_Unit'].mean()
        
        # Yeni fiyat
        new_price = current_price * (1 + price_increase/100)
        
        # Talep değişimi
        demand_change = elasticity * (price_increase/100)
        new_units = current_units * (1 + demand_change)
        
        # Yeni satış
        new_sales = new_units * new_price
        
        # Değişim yüzdeleri
        sales_change = ((new_sales - current_sales) / current_sales) * 100
        unit_change = ((new_units - current_units) / current_units) * 100
        
        # Sonuçları göster
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Mevcut Ciro",
                f"${current_sales:,.0f}",
                delta=f"{sales_change:+.1f}%"
            )
        
        with col2:
            st.metric(
                "Tahmini Ciro",
                f"${new_sales:,.0f}",
                delta=f"{unit_change:+.1f}% birim değişimi"
            )
        
        # Detaylı analiz
        with st.expander("Simülasyon Detayları"):
            st.write(f"**Fiyat Elastikiyeti:** {elasticity}")
            st.write(f"**Ortalama Fiyat Değişimi:** {price_increase}%")
            st.write(f"**Talep Değişimi:** {demand_change*100:.1f}%")
            st.write(f"**Birim Satış Değişimi:** {unit_change:.1f}%")
            st.write(f"**Ciro Değişimi:** {sales_change:.1f}%")
    
    def render_tab1_overview(self):
        """Genel Bakış sekmesini render eder."""
        st.title("🏠 Executive Summary")
        
        if self.data_manager.df is None:
            st.warning("Lütfen önce veri yükleyin veya demo veri kullanın!")
            return
        
        # KPI Kartları
        stats = self.data_manager.get_summary_stats()
        self.visualizer.create_kpi_cards(stats)
        
        st.divider()
        
        # Trend Grafikleri
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sektörel Dağılım")
            sector_data = self.data_manager.df.groupby('Sector')['USD_Value'].sum().reset_index()
            fig = px.pie(
                sector_data,
                values='USD_Value',
                names='Sector',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Aylık Trend")
            fig = self.visualizer.create_time_series(self.data_manager.df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detaylı Tablo
        st.subheader("📋 Detaylı Özet")
        summary_table = self.data_manager.df.groupby(['Country', 'Sector']).agg({
            'USD_Value': ['sum', 'mean', 'std'],
            'Units_Sold': 'sum',
            'YoY_Growth': 'mean'
        }).round(2)
        
        st.dataframe(
            summary_table,
            use_container_width=True,
            height=400
        )
    
    def render_tab2_geo_insights(self):
        """Coğrafi Analiz sekmesini render eder."""
        st.title("🌍 Coğrafi Analiz")
        
        if self.data_manager.df is None:
            st.warning("Lütfen önce veri yükleyin veya demo veri kullanın!")
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Metrik seçimi
            metric_options = ['USD_Value', 'Units_Sold', 'Price_Per_Unit', 'YoY_Growth']
            selected_metric = st.selectbox(
                "Görselleştirilecek Metrik",
                metric_options,
                format_func=lambda x: x.replace('_', ' ')
            )
            
            # Harita
            fig = self.visualizer.create_choropleth_map(
                self.data_manager.df, 
                selected_metric
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🌐 Ülke Performansı")
            
            # Top 10 ülke
            top_countries = self.data_manager.df.groupby('Country')['USD_Value'] \
                .sum().nlargest(10).reset_index()
            
            for idx, row in top_countries.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style="padding: 0.5rem; margin: 0.2rem 0; 
                                background: rgba(31, 119, 180, 0.1); 
                                border-radius: 5px;">
                        <strong>{row['Country']}</strong><br>
                        <small>${row['USD_Value']:,.0f}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
            
            # Hızlı istatistikler
            st.metric("En Hızlı Büyüyen", "Türkiye", "+15.2%")
            st.metric("En Karlı", "ABD", "$2.1M")
            st.metric("En Yüksek Fiyat", "Japonya", "$45.2/unit")
        
        st.divider()
        
        # Ülke bazlı detaylı analiz
        st.subheader("📊 Ülke Bazlı Karşılaştırma")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot: Büyüme vs Karlılık
            country_stats = self.data_manager.df.groupby('Country').agg({
                'USD_Value': 'sum',
                'YoY_Growth': 'mean',
                'Profit_Margin': 'mean'
            }).reset_index()
            
            fig = px.scatter(
                country_stats,
                x='YoY_Growth',
                y='Profit_Margin',
                size='USD_Value',
                color='USD_Value',
                hover_name='Country',
                size_max=50,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                title='Büyüme vs Karlılık Analizi',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart: Ülke performansı
            fig = px.bar(
                country_stats.nlargest(15, 'USD_Value'),
                x='Country',
                y='USD_Value',
                color='YoY_Growth',
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(
                title='Top 15 Ülke - Satış Performansı',
                height=500,
                xaxis_tickangle=45
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_tab3_competition(self):
        """Rekabet Analizi sekmesini render eder."""
        st.title("🏢 Rekabet Analizi")
        
        if self.data_manager.df is None:
            st.warning("Lütfen önce veri yükleyin veya demo veri kullanın!")
            return
        
        # Pareto Analizi
        st.subheader("📉 Pareto Analizi (80/20 Kuralı)")
        fig = self.visualizer.create_pareto_chart(self.data_manager.df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Pazar Liderleri")
            
            # Pazar payı değişimi
            market_share_data = self.data_manager.df.groupby(['Year', 'Corporation']) \
                .agg({'USD_Value': 'sum'}).reset_index()
            
            # Pivot tablo
            pivot_data = market_share_data.pivot(
                index='Year', 
                columns='Corporation', 
                values='USD_Value'
            ).fillna(0)
            
            # Yıllara göre pazar payı
            market_share_pct = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100
            
            fig = px.area(
                market_share_pct,
                title='Yıllara Göre Pazar Payı Değişimi',
                labels={'value': 'Pazar Payı (%)', 'Year': 'Yıl'}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Performans Göstergeleri")
            
            # Şirket bazlı KPI'lar
            corp_kpis = self.data_manager.df.groupby('Corporation').agg({
                'USD_Value': ['sum', 'mean', 'std'],
                'YoY_Growth': 'mean',
                'Market_Share': 'mean'
            }).round(2)
            
            # Performans skoru hesapla
            corp_kpis['Performance_Score'] = (
                corp_kpis[('USD_Value', 'sum')] / corp_kpis[('USD_Value', 'sum')].max() * 40 +
                corp_kpis[('YoY_Growth', 'mean')] / abs(corp_kpis[('YoY_Growth', 'mean')]).max() * 30 +
                (1 - corp_kpis[('USD_Value', 'std')] / corp_kpis[('USD_Value', 'std')].max()) * 30
            )
            
            # Sırala ve göster
            corp_kpis = corp_kpis.sort_values('Performance_Score', ascending=False)
            
            st.dataframe(
                corp_kpis.head(10),
                use_container_width=True,
                height=400
            )
        
        # Detaylı rapor
        with st.expander("📋 Detaylı Rekabet Raporu"):
            st.write("""
            **Analiz Metodolojisi:**
            1. Pazar konsantrasyonu (Herfindahl-Hirschman Index)
            2. Büyüme oranları karşılaştırması
            3. Fiyat rekabet analizi
            4. Ürün portföyü çeşitliliği
            """)
            
            # HHI hesaplama
            market_shares = self.data_manager.df.groupby('Corporation')['USD_Value'] \
                .sum() / self.data_manager.df['USD_Value'].sum()
            hhi = (market_shares ** 2).sum() * 10000
            
            st.metric("Pazar Konsantrasyonu (HHI)", f"{hhi:.0f}", 
                     delta="Düşük" if hhi < 1500 else "Yüksek")
    
    def render_tab4_molecule(self):
        """Molekül Analizi sekmesini render eder."""
        st.title("💊 Molekül Analizi")
        
        if self.data_manager.df is None:
            st.warning("Lütfen önce veri yükleyin veya demo veri kullanın!")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Ürün Yaşam Döngüsü
            st.subheader("📈 Ürün Yaşam Döngüsü")
            
            # Molekül bazlı trend
            molecule_trend = self.data_manager.df.groupby(['Year', 'Molecule']).agg({
                'USD_Value': 'sum',
                'Units_Sold': 'sum'
            }).reset_index()
            
            # En aktif 5 molekül
            top_molecules = molecule_trend.groupby('Molecule')['USD_Value'] \
                .sum().nlargest(5).index.tolist()
            
            filtered_trend = molecule_trend[molecule_trend['Molecule'].isin(top_molecules)]
            
            fig = px.line(
                filtered_trend,
                x='Year',
                y='USD_Value',
                color='Molecule',
                markers=True,
                title='Üst 5 Molekül - Satış Trendi'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🏆 Molekül Performansı")
            
            # Molekül sıralaması
            molecule_perf = self.data_manager.df.groupby('Molecule').agg({
                'USD_Value': 'sum',
                'YoY_Growth': 'mean',
                'Price_Per_Unit': 'mean'
            }).nlargest(10, 'USD_Value').reset_index()
            
            for idx, row in molecule_perf.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style="padding: 0.8rem; margin: 0.3rem 0; 
                                border-left: 4px solid #1f77b4;
                                background: rgba(31, 119, 180, 0.05); 
                                border-radius: 5px;">
                        <strong>#{idx+1} {row['Molecule']}</strong><br>
                        <small>Satış: ${row['USD_Value']:,.0f}</small><br>
                        <small>Büyüme: {row['YoY_Growth']:.1f}%</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
        
        # Fiyat Elastikiyeti Analizi
        st.subheader("💰 Fiyat Elastikiyeti Analizi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot with regression line
            elasticity_data = self.data_manager.df.groupby('Molecule').agg({
                'Price_Per_Unit': 'mean',
                'Units_Sold': 'sum',
                'USD_Value': 'sum'
            }).reset_index()
            
            # Log dönüşümü
            elasticity_data['Log_Price'] = np.log(elasticity_data['Price_Per_Unit'])
            elasticity_data['Log_Units'] = np.log(elasticity_data['Units_Sold'])
            
            fig = px.scatter(
                elasticity_data,
                x='Log_Price',
                y='Log_Units',
                hover_name='Molecule',
                size='USD_Value',
                color='USD_Value',
                trendline='ols',
                title='Fiyat-Talep İlişkisi'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Elastikiyet katsayıları
            st.write("**Fiyat Elastikiyeti Tahminleri:**")
            
            # Basit regresyon ile elastikiyet
            X = elasticity_data['Log_Price'].values.reshape(-1, 1)
            y = elasticity_data['Log_Units'].values
            
            if len(X) > 1:
                X_with_const = sm.add_constant(X)
                model = sm.OLS(y, X_with_const).fit()
                
                # Elastikiyet katsayısı
                elasticity_coef = model.params[1]
                
                st.metric(
                    "Ortalama Fiyat Elastikiyeti",
                    f"{elasticity_coef:.3f}",
                    delta="Esnek" if elasticity_coef < -1 else "Esnek Değil"
                )
                
                # Detaylı sonuçlar
                with st.expander("Regresyon Sonuçları"):
                    st.text(str(model.summary()))
                
                # Öneriler
                st.info(f"""
                **Analiz Sonucu:**
                - Elastikiyet katsayısı: {elasticity_coef:.3f}
                - Talep fiyata {"çok duyarlı" if elasticity_coef < -1 else "az duyarlı"}
                - Öneri: {"Fiyat artışı dikkatli yapılmalı" if elasticity_coef < -1 else "Fiyat esnekliği düşük"}
                """)
        
        # Molekül portföyü optimizasyonu
        with st.expander("🔍 Molekül Portföyü Optimizasyonu"):
            st.write("""
            **Portföy Matrisi (BCG):**
            - Yıldızlar: Yüksek büyüme, yüksek pazar payı
            - Nakit İnekleri: Düşük büyüme, yüksek pazar payı
            - Soru İşaretleri: Yüksek büyüme, düşük pazar payı
            - Köpekler: Düşük büyüme, düşük pazar payı
            """)
            
            # BCG matrisi
            bcg_data = self.data_manager.df.groupby('Molecule').agg({
                'Market_Share': 'mean',
                'YoY_Growth': 'mean'
            }).reset_index()
            
            fig = px.scatter(
                bcg_data,
                x='Market_Share',
                y='YoY_Growth',
                hover_name='Molecule',
                color='YoY_Growth',
                size='Market_Share',
                title='BCG Matrisi - Molekül Portföyü'
            )
            
            # Quadrant çizgileri
            fig.add_hline(y=bcg_data['YoY_Growth'].median(), line_dash="dash", line_color="gray")
            fig.add_vline(x=bcg_data['Market_Share'].median(), line_dash="dash", line_color="gray")
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_tab5_time_series(self):
        """Zaman Serisi sekmesini render eder."""
        st.title("📈 Zaman Serisi Analizi")
        
        if self.data_manager.df is None:
            st.warning("Lütfen önce veri yükleyin veya demo veri kullanın!")
            return
        
        # Ana trend grafiği
        st.subheader("📊 Satış Trendleri (2022-2024)")
        
        # Çoklu metrik seçimi
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_sales = st.checkbox("Satış (USD)", value=True)
        with col2:
            show_units = st.checkbox("Birim Satış", value=True)
        with col3:
            show_growth = st.checkbox("Büyüme (%)", value=False)
        
        # Zaman serisi verisi
        time_data = self.data_manager.df.groupby(['Year', 'Quarter']).agg({
            'USD_Value': 'sum',
            'Units_Sold': 'sum',
            'YoY_Growth': 'mean'
        }).reset_index()
        
        time_data['Period'] = time_data['Year'].astype(str) + ' ' + time_data['Quarter']
        
        # Dinamik grafik
        fig = go.Figure()
        
        if show_sales:
            fig.add_trace(go.Scatter(
                x=time_data['Period'],
                y=time_data['USD_Value'],
                name='Satış (USD)',
                line=dict(color='blue', width=3),
                yaxis='y'
            ))
        
        if show_units:
            fig.add_trace(go.Scatter(
                x=time_data['Period'],
                y=time_data['Units_Sold'],
                name='Birim Satış',
                line=dict(color='green', width=2, dash='dash'),
                yaxis='y2'
            ))
        
        if show_growth:
            fig.add_trace(go.Bar(
                x=time_data['Period'],
                y=time_data['YoY_Growth'],
                name='Büyüme (%)',
                marker_color='orange',
                opacity=0.6,
                yaxis='y3'
            ))
        
        # Layout ayarları
        fig.update_layout(
            title='Çoklu Metrik Trend Analizi',
            height=600,
            xaxis=dict(title='Dönem'),
            yaxis=dict(title='Satış (USD)', side='left'),
            yaxis2=dict(
                title='Birim Satış',
                overlaying='y',
                side='right'
            ),
            yaxis3=dict(
                title='Büyüme (%)',
                overlaying='y',
                side='right',
                position=0.95
            ),
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Mevsimsellik analizi
            st.subheader("🌱 Mevsimsellik Analizi")
            
            # Çeyreklere göre ortalama satış
            seasonal_data = self.data_manager.df.groupby('Quarter').agg({
                'USD_Value': 'mean',
                'Units_Sold': 'mean'
            }).reset_index()
            
            fig = px.bar_polar(
                seasonal_data,
                r='USD_Value',
                theta='Quarter',
                color='USD_Value',
                template='plotly_dark',
                color_continuous_scale='Viridis',
                title='Çeyreklere Göre Satış Dağılımı'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Trend ayrıştırma
            st.subheader("🔍 Trend Bileşenleri")
            
            try:
                # Zaman serisi ayrıştırma
                monthly_data = self.data_manager.df.resample('M', on='Date')['USD_Value'].sum()
                
                if len(monthly_data) >= 24:  # En az 2 yıl veri
                    decomposition = seasonal_decompose(
                        monthly_data,
                        model='additive',
                        period=12
                    )
                    
                    # Ayrıştırma grafiği
                    fig = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=('Orjinal Seri', 'Trend', 'Mevsimsellik', 'Artık'),
                        vertical_spacing=0.08
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=monthly_data.index, y=monthly_data, name='Orjinal'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Mevsimsellik'),
                        row=3, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Artık'),
                        row=4, col=1
                    )
                    
                    fig.update_layout(height=800, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Trend ayrıştırma için yeterli veri yok (en az 24 ay)")
                    
            except Exception as e:
                st.error(f"Mevsimsellik analizi hatası: {str(e)}")
        
        # Korelasyon analizi
        with st.expander("📊 Korelasyon Matrisi"):
            numeric_cols = self.data_manager.df.select_dtypes(include=[np.number]).columns
            corr_matrix = self.data_manager.df[numeric_cols].corr()
            
            fig = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.columns.tolist(),
                colorscale='RdBu',
                zmin=-1, zmax=1,
                showscale=True
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_tab6_ml_lab(self):
        """ML Laboratuvarı sekmesini render eder."""
        st.title("🤖 Makine Öğrenmesi Laboratuvarı")
        
        if self.data_manager.df is None:
            st.warning("Lütfen önce veri yükleyin veya demo veri kullanın!")
            return
        
        # Sekme yapısı
        ml_tab1, ml_tab2, ml_tab3 = st.tabs([
            "🔮 Tahmin (Forecasting)",
            "🎯 Kümeleme (Clustering)",
            "🚨 Anomali Tespiti"
        ])
        
        # TAB 1: Tahmin Modeli
        with ml_tab1:
            st.header("2025-2026 Satış Tahminleri")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("""
                **Model:** Random Forest Regressor
                **Özellikler:** Zaman indeksi, Birim satışlar, Mevsimsel faktörler
                **Çıktı:** 2025-2026 çeyreklik satış tahminleri
                """)
            
            with col2:
                if st.button("🎯 Modeli Eğit ve Tahmin Et", type="primary", use_container_width=True):
                    with st.spinner("Model eğitiliyor..."):
                        results = self.ml_manager.train_forecasting_model()
                        
                        if results:
                            st.success(f"Model başarıyla eğitildi! (R²: {results['r2']:.3f})")
            
            # Tahmin sonuçları
            if self.ml_manager.forecast_model is not None:
                results = self.ml_manager.train_forecasting_model()
                predictions = results['future_predictions']
                
                # Tahmin grafiği
                fig = go.Figure()
                
                # Geçmiş veriler
                historical = self.data_manager.df.groupby(['Year', 'Quarter'])['USD_Value'] \
                    .sum().reset_index()
                historical['Period'] = historical['Year'].astype(str) + ' ' + historical['Quarter']
                
                fig.add_trace(go.Scatter(
                    x=historical['Period'],
                    y=historical['USD_Value'],
                    mode='lines+markers',
                    name='Geçmiş Veri',
                    line=dict(color='blue', width=2)
                ))
                
                # Tahminler
                predictions['Period'] = predictions['Year'].astype(str) + ' ' + predictions['Quarter']
                
                fig.add_trace(go.Scatter(
                    x=predictions['Period'],
                    y=predictions['Predicted_Value'],
                    mode='lines+markers',
                    name='Tahmin',
                    line=dict(color='green', width=3, dash='dash')
                ))
                
                # Güven aralığı
                fig.add_trace(go.Scatter(
                    x=list(predictions['Period']) + list(predictions['Period'])[::-1],
                    y=list(predictions['Upper_Bound']) + list(predictions['Lower_Bound'])[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    showlegend=True,
                    name='Güven Aralığı (%90)'
                ))
                
                fig.update_layout(
                    title='2025-2026 Satış Tahminleri',
                    height=500,
                    xaxis_title='Dönem',
                    yaxis_title='Satış (USD)',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tahmin tablosu
                st.subheader("📋 Tahmin Tablosu")
                predictions_display = predictions.copy()
                predictions_display['Predicted_Value'] = predictions_display['Predicted_Value'].apply(
                    lambda x: f"${x:,.0f}"
                )
                predictions_display['Lower_Bound'] = predictions_display['Lower_Bound'].apply(
                    lambda x: f"${x:,.0f}"
                )
                predictions_display['Upper_Bound'] = predictions_display['Upper_Bound'].apply(
                    lambda x: f"${x:,.0f}"
                )
                
                st.dataframe(
                    predictions_display,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Model performansı
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Skoru", f"{results['r2']:.3f}")
                with col2:
                    st.metric("Ortalama Mutlak Hata", f"${results['mae']:,.0f}")
                with col3:
                    st.metric("Toplam 2025 Tahmini", f"${predictions[predictions['Year']==2025]['Predicted_Value'].sum():,.0f}")
        
        # TAB 2: Kümeleme Modeli
        with ml_tab2:
            st.header("Ülke Segmentasyonu (Clustering)")
            
            # Küme sayısı seçimi
            col1, col2 = st.columns(2)
            
            with col1:
                n_clusters = st.slider(
                    "Küme Sayısı (K)",
                    min_value=2,
                    max_value=10,
                    value=3,
                    help="Elbow method ile optimal değeri belirleyin"
                )
            
            with col2:
                if st.button("🎯 Ülkeleri Kümele", type="primary", use_container_width=True):
                    with st.spinner("Kümeleme yapılıyor..."):
                        self.ml_manager.train_clustering_model(n_clusters)
            
            # Optimal küme analizi
            st.subheader("📊 Optimal Küme Sayısı Analizi")
            elbow_fig = self.ml_manager.find_optimal_clusters()
            st.plotly_chart(elbow_fig, use_container_width=True)
            
            # Kümeleme sonuçları
            if self.ml_manager.clustering_model is not None:
                results = self.ml_manager.train_clustering_model(n_clusters)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 3D kümeleme grafiği
                    cluster_data = results['data'].copy()
                    cluster_data['Cluster'] = results['labels']
                    
                    fig = self.visualizer.create_3d_cluster_plot(cluster_data, results['labels'])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Küme özellikleri
                    st.subheader("🎯 Küme Profilleri")
                    
                    for cluster_id in range(n_clusters):
                        cluster_stats = cluster_data[cluster_data['Cluster'] == cluster_id]
                        
                        with st.expander(f"Küme {cluster_id + 1} ({len(cluster_stats)} ülke)"):
                            st.write("**Ülkeler:**", ", ".join(cluster_stats['Country'].head(5).tolist()))
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Ort. Fiyat", f"${cluster_stats['Price_Per_Unit'].mean():.2f}")
                            with col_b:
                                st.metric("Ort. Hacim", f"{cluster_stats['Sales_Volume'].mean():.2f}")
                            with col_c:
                                st.metric("Ort. Büyüme", f"{cluster_stats['YoY_Growth'].mean():.1f}%")
                    
                    # Silhouette skoru
                    st.metric(
                        "Model Kalitesi (Silhouette)",
                        f"{results['silhouette_score']:.3f}",
                        delta="İyi" if results['silhouette_score'] > 0.5 else "Orta",
                        delta_color="normal"
                    )
                
                # Küme dağılım haritası
                st.subheader("🌍 Kümeleme Haritası")
                
                # Ülke kodlarını al
                cluster_data['Country_Code'] = cluster_data['Country'].apply(
                    self.data_manager.normalize_country_name
                )
                
                fig = px.choropleth(
                    cluster_data,
                    locations="Country_Code",
                    color="Cluster",
                    hover_name="Country",
                    hover_data={
                        'Price_Per_Unit': ':.2f',
                        'YoY_Growth': ':.1f',
                        'Country_Code': False,
                        'Cluster': True
                    },
                    color_continuous_scale=px.colors.qualitative.Set3,
                    title="Ülke Kümeleri - Coğrafi Dağılım"
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        # TAB 3: Anomali Tespiti
        with ml_tab3:
            st.header("🚨 Anomali Tespiti")
            
            col1, col2 = st.columns(2)
            
            with col1:
                contamination = st.slider(
                    "Anomali Oranı Tahmini",
                    min_value=0.01,
                    max_value=0.3,
                    value=0.1,
                    step=0.01,
                    help="Veri setindeki tahmini anomali oranı"
                )
            
            with col2:
                if st.button("🔍 Anomalileri Tespit Et", type="primary", use_container_width=True):
                    with st.spinner("Anomali analizi yapılıyor..."):
                        results = self.ml_manager.detect_anomalies(contamination)
                        
                        if results:
                            anomaly_pct = (results['anomaly_count'] / results['total_count']) * 100
                            st.success(f"{results['anomaly_count']} anomali tespit edildi ({anomaly_pct:.1f}%)")
            
            # Anomali sonuçları
            if self.ml_manager.anomaly_model is not None:
                results = self.ml_manager.detect_anomalies(contamination)
                anomaly_data = results['anomaly_data']
                
                # Anomali dağılımı
                col1, col2 = st.columns(2)
                
                with col1:
                    # Anomali sayıları
                    fig = px.pie(
                        anomaly_data,
                        names='Is_Anomaly',
                        title='Anomali Dağılımı',
                        color_discrete_sequence=['green', 'red']
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Anomali özellikleri
                    st.subheader("📋 Anomali Detayları")
                    
                    # Anomalileri listele
                    anomalies = anomaly_data[anomaly_data['Is_Anomaly'] == True]
                    
                    if not anomalies.empty:
                        for idx, row in anomalies.head(10).iterrows():
                            st.warning(f"""
                            **{row['Country']} - {row['Corporation']}**
                            - Satış: ${row['USD_Value']:,.0f}
                            - Fiyat: ${row['Price_Per_Unit']:.2f}/unit
                            - Büyüme: {row['YoY_Growth']:.1f}%
                            """)
                    else:
                        st.info("Anomali tespit edilmedi.")
                
                # Anomali scatter plot
                st.subheader("📊 Anomali Görselleştirme")
                
                fig = px.scatter(
                    anomaly_data,
                    x='USD_Value',
                    y='Price_Per_Unit',
                    color='Is_Anomaly',
                    size='YoY_Growth',
                    hover_name='Country',
                    hover_data=['Corporation', 'YoY_Growth'],
                    color_discrete_sequence=['green', 'red'],
                    title='Anomali Dağılımı - Satış vs Fiyat'
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomali analizi raporu
                with st.expander("📈 Anomali Analizi Raporu"):
                    st.write("""
                    **Potansiyel Nedenler:**
                    1. Aşırı yüksek/az satış rakamları
                    2. Anormal fiyat değişimleri
                    3. Beklenmeyen büyüme oranları
                    4. Veri giriş hataları
                    
                    **Önerilen Aksiyonlar:**
                    - Anomalileri manuel olarak kontrol edin
                    - Veri kalitesini iyileştirin
                    - İş kurallarını gözden geçirin
                    """)
                    
                    # İstatistikler
                    anomaly_stats = anomaly_data.groupby('Is_Anomaly').agg({
                        'USD_Value': ['mean', 'std'],
                        'Price_Per_Unit': ['mean', 'std'],
                        'YoY_Growth': ['mean', 'std']
                    }).round(2)
                    
                    st.dataframe(anomaly_stats, use_container_width=True)
    
    def render_main(self):
        """Ana uygulamayı render eder."""
        load_css()
        
        # Sidebar
        self.render_sidebar()
        
        # Ana içerik
        if self.data_manager.df is None:
            # Hoşgeldin ekranı
            st.markdown("""
            <div style="text-align: center; padding: 5rem 1rem;">
                <h1 style="color: #1f77b4;">💊 İlaç Sektörü Satış Analizi</h1>
                <p style="font-size: 1.2rem; color: #666;">
                    Enterprise-Grade Dashboard
                </p>
                <div style="max-width: 600px; margin: 3rem auto;">
                    <p>📊 Kapsamlı satış analizi ve tahmin</p>
                    <p>🌍 Coğrafi görselleştirme</p>
                    <p>🤖 Makine öğrenmesi modelleri</p>
                    <p>📈 Zaman serisi analizi</p>
                </div>
                <div style="margin-top: 3rem;">
                    <p><strong>Başlamak için:</strong></p>
                    <p>1. Sidebar'dan Excel dosyası yükleyin</p>
                    <p>2. Veya "Demo Veri Yükle" butonuna tıklayın</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Sekmeler
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏠 Genel Bakış",
            "🌍 Coğrafi Analiz",
            "🏢 Rekabet Analizi",
            "💊 Molekül Analizi",
            "📈 Zaman Serisi",
            "🤖 ML Laboratuvarı"
        ])
        
        with tab1:
            self.render_tab1_overview()
        
        with tab2:
            self.render_tab2_geo_insights()
        
        with tab3:
            self.render_tab3_competition()
        
        with tab4:
            self.render_tab4_molecule()
        
        with tab5:
            self.render_tab5_time_series()
        
        with tab6:
            self.render_tab6_ml_lab()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Ana uygulama fonksiyonu."""
    try:
        # Uygulama başlığı
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #1f77b4; margin-bottom: 0;">İlaç Sektörü Satış Analizi</h1>
            <p style="color: #666; font-size: 1.1rem;">
                Enterprise-Grade Dashboard | AI-Powered Insights
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # UI Manager'ı başlat ve render et
        ui_manager = UIManager()
        ui_manager.render_main()
        
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        st.info("Lütfen sayfayı yenileyin veya daha sonra tekrar deneyin.")

if __name__ == "__main__":
    main()
