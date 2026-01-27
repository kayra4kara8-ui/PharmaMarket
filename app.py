"""
PHARMA MARKET - PROFESYONEL EXCEL VERİ ANALİZ PLATFORMU
Versiyon: 3.0.0
Tarih: 2026-01-27
Yazar: Pharma Analytics Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import io
import base64
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import math
import hashlib
import time
from collections import Counter
import altair as alt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(
    page_title="Pharma Market Analytics Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://pharmamarket.com/support',
        'Report a bug': 'https://pharmamarket.com/bug',
        'About': "Pharma Market Analytics v3.0 - Profesyonel Veri Analiz Platformu"
    }
)

# CSS stil enjeksiyonu
def inject_custom_css():
    """Özel CSS stilleri ekle"""
    st.markdown("""
    <style>
    /* Ana stil */
    .main-header {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem !important;
        padding: 1rem;
        border-radius: 15px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
    }
    
    .sub-header {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #2d3748 !important;
        border-left: 5px solid #4c51bf;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0 !important;
        background: linear-gradient(90deg, #f7fafc 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 0 10px 10px 0;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        margin: 0.5rem 0 !important;
    }
    
    .metric-label {
        font-size: 1rem !important;
        opacity: 0.9;
        margin-bottom: 0.5rem !important;
    }
    
    .success-badge {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .warning-badge {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .error-badge {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .info-badge {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .tab-button {
        background: #f7fafc;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        margin-right: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 600;
        color: #4a5568;
    }
    
    .tab-button:hover {
        background: #edf2f7;
        border-color: #cbd5e0;
    }
    
    .tab-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }
    
    /* Sidebar stil */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    /* Dataframe stil */
    .dataframe {
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button stil */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Selectbox stil */
    .stSelectbox > div > div > div {
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.25rem;
    }
    
    /* Slider stil */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tab içeriği */
    .tab-content {
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
        margin-top: 1rem;
    }
    
    /* Animasyonlar */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    </style>
    """, unsafe_allow_html=True)

# Session state başlatma
def initialize_session_state():
    """Session state değişkenlerini başlat"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'data_profile' not in st.session_state:
        st.session_state.data_profile = {}
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = {}
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    if 'report_data' not in st.session_state:
        st.session_state.report_data = {}
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = 'ready'
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = {
            'theme': 'light',
            'language': 'tr',
            'auto_save': True,
            'notifications': True
        }

# Veri yükleme fonksiyonları
@st.cache_data(ttl=3600, show_spinner="Excel dosyası yükleniyor...")
def load_excel_file(uploaded_file):
    """Excel dosyasını yükle ve DataFrame'e dönüştür"""
    try:
        # Dosya türüne göre okuma
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            st.error("Desteklenmeyen dosya formatı. Lütfen .xlsx, .xls, .csv veya .parquet formatında yükleyin.")
            return None
        
        # Sütun isimlerini temizle
        df.columns = [str(col).strip().upper().replace(' ', '_').replace('.', '_') 
                     for col in df.columns]
        
        # Veri tiplerini optimize et
        df = optimize_data_types(df)
        
        # Ön işleme
        df = preprocess_dataframe(df)
        
        return df
    
    except Exception as e:
        st.error(f"Dosya yükleme hatası: {str(e)}")
        return None

def optimize_data_types(df):
    """DataFrame veri tiplerini optimize et"""
    for col in df.columns:
        # Sayısal kolonlar için optimizasyon
        if pd.api.types.is_numeric_dtype(df[col]):
            col_min = df[col].min()
            col_max = df[col].max()
            
            # Integer optimizasyonu
            if pd.api.types.is_integer_dtype(df[col]):
                if col_min >= 0:
                    if col_max < 256:
                        df[col] = df[col].astype('uint8')
                    elif col_max < 65536:
                        df[col] = df[col].astype('uint16')
                    elif col_max < 4294967296:
                        df[col] = df[col].astype('uint32')
                else:
                    if col_min > -128 and col_max < 128:
                        df[col] = df[col].astype('int8')
                    elif col_min > -32768 and col_max < 32768:
                        df[col] = df[col].astype('int16')
                    elif col_min > -2147483648 and col_max < 2147483648:
                        df[col] = df[col].astype('int32')
            
            # Float optimizasyonu
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype('float32')
        
        # Kategorik kolonlar için optimizasyon
        elif df[col].dtype == 'object':
            unique_count = df[col].nunique()
            total_count = len(df[col])
            
            if unique_count / total_count < 0.5:  # %50'den az benzersiz değer
                df[col] = df[col].astype('category')
    
    return df

def preprocess_dataframe(df):
    """DataFrame ön işleme"""
    # NaN değerleri temizle
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('BİLİNMİYOR')
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
    
    # Tarih kolonlarını işle
    date_patterns = ['TARİH', 'DATE', 'TARIH', 'TARİH_', 'DATE_']
    for col in df.columns:
        if any(pattern in col for pattern in date_patterns):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    return df

# Veri analiz fonksiyonları
def analyze_missing_values(df):
    """Eksik veri analizi"""
    missing_stats = {}
    
    # np.product yerine np.prod kullan
    total_cells = np.prod(df.shape)
    total_missing = df.isnull().sum().sum()
    
    missing_stats['total_cells'] = total_cells
    missing_stats['total_missing'] = total_missing
    missing_stats['missing_percentage'] = (total_missing / total_cells * 100) if total_cells > 0 else 0
    
    # Kolon bazında eksik değerler
    missing_per_column = df.isnull().sum()
    missing_stats['missing_columns'] = missing_per_column[missing_per_column > 0].to_dict()
    
    # Eksik değer pattern'leri
    missing_stats['missing_patterns'] = {}
    for col in df.columns:
        if df[col].isnull().any():
            missing_stats['missing_patterns'][col] = {
                'count': df[col].isnull().sum(),
                'percentage': df[col].isnull().mean() * 100,
                'suggested_fix': 'median' if pd.api.types.is_numeric_dtype(df[col]) else 'mode'
            }
    
    return missing_stats

def analyze_data_quality(df):
    """Veri kalitesi analizi"""
    quality_stats = {}
    
    # Veri tipleri
    quality_stats['data_types'] = df.dtypes.astype(str).to_dict()
    
    # Benzersiz değer sayıları
    quality_stats['unique_counts'] = df.nunique().to_dict()
    
    # Outlier analizi
    quality_stats['outliers'] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols[:10]:  # İlk 10 sayısal kolon
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            quality_stats['outliers'][col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        except:
            pass
    
    # Veri dağılımı
    quality_stats['distributions'] = {}
    for col in numeric_cols[:5]:
        try:
            quality_stats['distributions'][col] = {
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
        except:
            pass
    
    return quality_stats

def calculate_basic_statistics(df):
    """Temel istatistikleri hesapla"""
    stats = {}
    
    # Sayısal kolonlar için istatistikler
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        stats[col] = {
            'count': df[col].count(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            '25%': df[col].quantile(0.25),
            '50%': df[col].median(),
            '75%': df[col].quantile(0.75),
            'max': df[col].max(),
            'variance': df[col].var(),
            'range': df[col].max() - df[col].min()
        }
    
    # Kategorik kolonlar için istatistikler
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols[:10]:  # İlk 10 kategorik kolon
        stats[col] = {
            'unique_count': df[col].nunique(),
            'mode': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'top_frequencies': df[col].value_counts().head(5).to_dict()
        }
    
    return stats

def generate_data_profile(df):
    """Kapsamlı veri profili oluştur"""
    profile = {}
    
    # Temel bilgiler
    profile['shape'] = df.shape
    profile['memory_usage'] = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    
    # Veri tipleri dağılımı
    profile['dtype_distribution'] = df.dtypes.value_counts().to_dict()
    
    # İstatistikler
    profile['statistics'] = calculate_basic_statistics(df)
    
    # Eksik değerler
    profile['missing_values'] = analyze_missing_values(df)
    
    # Veri kalitesi
    profile['data_quality'] = analyze_data_quality(df)
    
    # Korelasyon analizi
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        profile['correlation_matrix'] = numeric_df.corr().to_dict()
        
        # En yüksek korelasyonlar
        corr_matrix = numeric_df.corr()
        high_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.7:
                    high_correlations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': corr_value,
                        'type': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                    })
        
        profile['high_correlations'] = sorted(
            high_correlations, 
            key=lambda x: abs(x['correlation']), 
            reverse=True
        )[:20]
    
    return profile

# Görselleştirme fonksiyonları
def create_correlation_heatmap(df):
    """Korelasyon heatmap oluştur"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        return None
    
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Korelasyon"),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Korelasyon Matrisi',
        height=600,
        width=800,
        xaxis_title="Değişkenler",
        yaxis_title="Değişkenler",
        template='plotly_white'
    )
    
    return fig

def create_distribution_plot(df, column):
    """Dağılım grafiği oluştur"""
    if column not in df.columns:
        return None
    
    if pd.api.types.is_numeric_dtype(df[column]):
        # Histogram ve KDE
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df[column].dropna(),
            name='Histogram',
            nbinsx=50,
            opacity=0.7,
            marker_color='#667eea'
        ))
        
        # KDE eğrisi
        from scipy import stats
        kde = stats.gaussian_kde(df[column].dropna())
        x_range = np.linspace(df[column].min(), df[column].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde(x_range),
            name='KDE',
            line=dict(color='#764ba2', width=2)
        ))
        
        fig.update_layout(
            title=f'{column} Dağılımı',
            xaxis_title=column,
            yaxis_title='Frekans',
            template='plotly_white',
            height=400,
            showlegend=True
        )
    
    else:
        # Kategorik dağılım
        value_counts = df[column].value_counts().head(20)
        
        fig = go.Figure(data=[go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            marker_color='#667eea'
        )])
        
        fig.update_layout(
            title=f'{column} Dağılımı (Top 20)',
            xaxis_title=column,
            yaxis_title='Frekans',
            template='plotly_white',
            height=400,
            xaxis={'categoryorder': 'total descending'}
        )
    
    return fig

def create_time_series_plot(df, date_column, value_column):
    """Zaman serisi grafiği oluştur"""
    if date_column not in df.columns or value_column not in df.columns:
        return None
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except:
            return None
    
    # Tarihe göre grupla
    time_series = df.groupby(date_column)[value_column].sum().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_series[date_column],
        y=time_series[value_column],
        mode='lines+markers',
        name=value_column,
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2')
    ))
    
    # Hareketli ortalama (30 gün)
    if len(time_series) > 30:
        time_series['moving_avg'] = time_series[value_column].rolling(window=30).mean()
        
        fig.add_trace(go.Scatter(
            x=time_series[date_column],
            y=time_series['moving_avg'],
            mode='lines',
            name='30 Günlük Ortalama',
            line=dict(color='#48bb78', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f'{value_column} Zaman Serisi',
        xaxis_title='Tarih',
        yaxis_title=value_column,
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_box_plot(df, category_column, value_column):
    """Box plot oluştur"""
    if category_column not in df.columns or value_column not in df.columns:
        return None
    
    # Top 20 kategori
    top_categories = df[category_column].value_counts().head(20).index
    filtered_df = df[df[category_column].isin(top_categories)]
    
    fig = go.Figure()
    
    for category in top_categories:
        category_data = filtered_df[filtered_df[category_column] == category][value_column]
        
        fig.add_trace(go.Box(
            y=category_data,
            name=category,
            marker_color='#667eea',
            boxmean='sd'
        ))
    
    fig.update_layout(
        title=f'{value_column} Dağılımı - {category_column} Bazında',
        xaxis_title=category_column,
        yaxis_title=value_column,
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig

def create_scatter_plot(df, x_column, y_column, color_column=None):
    """Scatter plot oluştur"""
    if x_column not in df.columns or y_column not in df.columns:
        return None
    
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color_column,
        title=f'{y_column} vs {x_column}',
        opacity=0.7,
        color_continuous_scale='Viridis' if color_column and pd.api.types.is_numeric_dtype(df[color_column]) else None
    )
    
    fig.update_layout(
        template='plotly_white',
        height=500,
        hovermode='closest'
    )
    
    # Trend çizgisi
    if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
        z = np.polyfit(df[x_column].dropna(), df[y_column].dropna(), 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=df[x_column],
            y=p(df[x_column]),
            mode='lines',
            name='Trend Çizgisi',
            line=dict(color='red', dash='dash')
        ))
    
    return fig

# İleri analiz fonksiyonları
def perform_cluster_analysis(df, n_clusters=3):
    """Kümeleme analizi yap"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        return None
    
    # NaN değerleri temizle
    numeric_df = numeric_df.dropna()
    
    if len(numeric_df) < n_clusters:
        return None
    
    # Standardizasyon
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # K-Means kümeleme
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # PCA ile 2D'ye indirgeme
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    results = {
        'clusters': clusters.tolist(),
        'pca_result': pca_result.tolist(),
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'inertia': kmeans.inertia_,
        'explained_variance': pca.explained_variance_ratio_.tolist()
    }
    
    # Küme özellikleri
    df_clustered = numeric_df.copy()
    df_clustered['cluster'] = clusters
    
    cluster_stats = df_clustered.groupby('cluster').agg(['mean', 'std', 'count']).to_dict()
    results['cluster_statistics'] = cluster_stats
    
    return results

def perform_outlier_detection(df, method='iqr'):
    """Outlier tespiti yap"""
    numeric_df = df.select_dtypes(include=[np.number])
    outliers = {}
    
    for column in numeric_df.columns:
        data = numeric_df[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(data))
            column_outliers = df[z_scores > 3]
        
        outliers[column] = {
            'count': len(column_outliers),
            'indices': column_outliers.index.tolist(),
            'percentage': len(column_outliers) / len(df) * 100,
            'values': column_outliers[column].tolist() if len(column_outliers) > 0 else []
        }
    
    return outliers

def calculate_advanced_metrics(df):
    """İleri düzey metrikler hesapla"""
    metrics = {}
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    for column in numeric_df.columns:
        data = numeric_df[column].dropna()
        
        if len(data) > 0:
            metrics[column] = {
                'coefficient_of_variation': (data.std() / data.mean()) * 100 if data.mean() != 0 else 0,
                'quartile_coefficient_of_dispersion': (data.quantile(0.75) - data.quantile(0.25)) / 
                                                     (data.quantile(0.75) + data.quantile(0.25)) if 
                                                     (data.quantile(0.75) + data.quantile(0.25)) != 0 else 0,
                'entropy': stats.entropy(data.value_counts(normalize=True)) if len(data) > 0 else 0,
                'gini_coefficient': calculate_gini_coefficient(data),
                'normalized_entropy': calculate_normalized_entropy(data)
            }
    
    return metrics

def calculate_gini_coefficient(data):
    """Gini katsayısı hesapla"""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    index = np.arange(1, n + 1)
    
    gini = (np.sum((2 * index - n - 1) * sorted_data)) / (n * np.sum(sorted_data))
    return gini

def calculate_normalized_entropy(data):
    """Normalize edilmiş entropi hesapla"""
    value_counts = data.value_counts(normalize=True)
    entropy = stats.entropy(value_counts)
    max_entropy = np.log(len(value_counts)) if len(value_counts) > 0 else 0
    
    return entropy / max_entropy if max_entropy > 0 else 0

# Raporlama fonksiyonları
def generate_html_report(df, profile, analysis_results):
    """HTML raporu oluştur"""
    report = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pharma Market Analiz Raporu</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
            .section {{ margin-bottom: 40px; padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f7fafc; 
                      border-radius: 8px; min-width: 150px; text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
            .metric-label {{ font-size: 14px; color: #718096; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
            th {{ background-color: #f7fafc; font-weight: bold; }}
            .insight {{ background: #e6fffa; padding: 15px; border-left: 4px solid #38b2ac; margin: 10px 0; }}
            .warning {{ background: #fffaf0; padding: 15px; border-left: 4px solid #ed8936; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏥 Pharma Market Analiz Raporu</h1>
            <p>Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Veri Özeti</h2>
            <div class="metric">
                <div class="metric-value">{df.shape[0]:,}</div>
                <div class="metric-label">Toplam Satır</div>
            </div>
            <div class="metric">
                <div class="metric-value">{df.shape[1]}</div>
                <div class="metric-label">Toplam Sütun</div>
            </div>
            <div class="metric">
                <div class="metric-value">{profile['memory_usage']:.2f} MB</div>
                <div class="metric-label">Bellek Kullanımı</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔍 Veri Kalitesi</h2>
            <p>Eksik Veri Oranı: {profile['missing_values']['missing_percentage']:.2f}%</p>
            <p>Toplam Eksik Hücre: {profile['missing_values']['total_missing']:,}</p>
    """
    
    # Eksik değerler tablosu
    if profile['missing_values']['missing_columns']:
        report += """
            <h3>Eksik Değerler Olan Sütunlar:</h3>
            <table>
                <tr><th>Sütun</th><th>Eksik Sayı</th><th>Yüzde</th></tr>
        """
        for col, count in profile['missing_values']['missing_columns'].items():
            percentage = (count / len(df)) * 100
            report += f"<tr><td>{col}</td><td>{count:,}</td><td>{percentage:.2f}%</td></tr>"
        report += "</table>"
    
    # Outlier'lar
    if 'outliers' in profile['data_quality']:
        report += """
            <div class="section">
                <h2>⚠️ Outlier Tespiti</h2>
        """
        for col, outlier_info in profile['data_quality']['outliers'].items():
            if outlier_info['count'] > 0:
                report += f"""
                <div class="warning">
                    <strong>{col}</strong>: {outlier_info['count']} outlier tespit edildi 
                    ({outlier_info['percentage']:.2f}%)
                </div>
                """
    
    # İstatistikler
    report += """
        <div class="section">
            <h2>📈 Temel İstatistikler</h2>
            <table>
                <tr><th>Sütun</th><th>Ortalama</th><th>Std Sapma</th><th>Min</th><th>Max</th></tr>
    """
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in list(numeric_cols)[:10]:  # İlk 10 sayısal sütun
        if col in profile['statistics']:
            stats = profile['statistics'][col]
            report += f"""
                <tr>
                    <td>{col}</td>
                    <td>{stats.get('mean', 0):.2f}</td>
                    <td>{stats.get('std', 0):.2f}</td>
                    <td>{stats.get('min', 0):.2f}</td>
                    <td>{stats.get('max', 0):.2f}</td>
                </tr>
            """
    
    report += """
            </table>
        </div>
        
        <div class="section">
            <h2>💡 Öneriler</h2>
    """
    
    # Öneriler
    if profile['missing_values']['missing_percentage'] > 5:
        report += """
            <div class="insight">
                <strong>Eksik Veri Önerisi:</strong> Veri setinde %5'ten fazla eksik veri bulunuyor. 
                Eksik verileri doldurmak için imputation yöntemleri uygulanmalı.
            </div>
        """
    
    if 'outliers' in profile['data_quality']:
        outlier_count = sum(info['count'] for info in profile['data_quality']['outliers'].values())
        if outlier_count > 0:
            report += f"""
                <div class="insight">
                    <strong>Outlier Önerisi:</strong> {outlier_count} outlier tespit edildi. 
                    Bu değerler analizden önce temizlenmeli veya dönüştürülmeli.
                </div>
            """
    
    report += """
        </div>
        
        <div class="section">
            <h2>📋 Kolon Bilgileri</h2>
            <table>
                <tr><th>Sütun</th><th>Veri Tipi</th><th>Benzersiz Değer</th><th>Eksik Değer</th></tr>
    """
    
    for col in df.columns[:20]:  # İlk 20 sütun
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        missing_count = df[col].isnull().sum()
        
        report += f"""
            <tr>
                <td>{col}</td>
                <td>{dtype}</td>
                <td>{unique_count:,}</td>
                <td>{missing_count:,}</td>
            </tr>
        """
    
    report += """
            </table>
        </div>
        
        <footer style="margin-top: 50px; padding: 20px; text-align: center; color: #718096; border-top: 1px solid #e2e8f0;">
            <p>Pharma Market Analytics Platform v3.0 © 2026</p>
            <p>Bu rapor otomatik olarak oluşturulmuştur.</p>
        </footer>
    </body>
    </html>
    """
    
    return report

def export_to_excel(df, profile, analysis_results):
    """Excel'e export"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Ana veri
        df.to_excel(writer, sheet_name='Ana_Veri', index=False)
        
        # İstatistikler
        stats_df = pd.DataFrame(profile['statistics']).T
        stats_df.to_excel(writer, sheet_name='İstatistikler')
        
        # Eksik değerler
        missing_df = pd.DataFrame({
            'Sütun': list(profile['missing_values']['missing_columns'].keys()),
            'Eksik_Sayı': list(profile['missing_values']['missing_columns'].values()),
            'Yüzde': [count/len(df)*100 for count in profile['missing_values']['missing_columns'].values()]
        })
        missing_df.to_excel(writer, sheet_name='Eksik_Değerler', index=False)
        
        # Outlier'lar
        if 'outliers' in profile['data_quality']:
            outliers_data = []
            for col, info in profile['data_quality']['outliers'].items():
                if info['count'] > 0:
                    outliers_data.append({
                        'Sütun': col,
                        'Outlier_Sayısı': info['count'],
                        'Yüzde': info['percentage'],
                        'Alt_Sınır': info.get('lower_bound', 0),
                        'Üst_Sınır': info.get('upper_bound', 0)
                    })
            
            if outliers_data:
                outliers_df = pd.DataFrame(outliers_data)
                outliers_df.to_excel(writer, sheet_name='Outlierlar', index=False)
    
    return output.getvalue()

# Ana uygulama bileşenleri
def render_sidebar():
    """Sidebar bileşenlerini render et"""
    with st.sidebar:
        st.markdown("## ⚙️ Ayarlar")
        
        # Dosya yükleme
        uploaded_file = st.file_uploader(
            "📤 Excel/CSV Dosyası Yükle",
            type=['xlsx', 'xls', 'csv', 'parquet'],
            help="Excel, CSV veya Parquet formatında dosya yükleyin"
        )
        
        if uploaded_file is not None:
            if st.session_state.uploaded_file != uploaded_file:
                with st.spinner("Veri yükleniyor..."):
                    df = load_excel_file(uploaded_file)
                    
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.uploaded_file = uploaded_file
                        st.session_state.data_profile = generate_data_profile(df)
                        st.session_state.selected_columns = list(df.columns)
                        
                        st.success(f"✓ {uploaded_file.name} başarıyla yüklendi!")
                        
                        # Temel bilgiler
                        st.info(f"""
                        **Veri Bilgileri:**
                        - Satır Sayısı: {df.shape[0]:,}
                        - Sütun Sayısı: {df.shape[1]}
                        - Bellek Kullanımı: {st.session_state.data_profile['memory_usage']:.2f} MB
                        """)
        
        # Sütun seçimi
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("## 🎯 Sütun Seçimi")
            
            all_columns = list(st.session_state.df.columns)
            selected_columns = st.multiselect(
                "Analiz için sütunları seçin:",
                options=all_columns,
                default=st.session_state.selected_columns,
                help="Analiz edilecek sütunları seçin"
            )
            
            st.session_state.selected_columns = selected_columns
            
            # Satır sınırı
            st.markdown("---")
            st.markdown("## 📊 Görüntüleme Ayarları")
            
            max_rows = len(st.session_state.df)
            if max_rows > 100:
                row_limit = st.slider(
                    "Görüntülenecek Satır Sayısı",
                    min_value=100,
                    max_value=min(10000, max_rows),
                    value=min(1000, max_rows),
                    step=100,
                    help="Tabloda görüntülenecek maksimum satır sayısı"
                )
                st.session_state.row_limit = row_limit
            else:
                st.session_state.row_limit = max_rows
            
            # Analiz ayarları
            st.markdown("---")
            st.markdown("## 🔧 Analiz Ayarları")
            
            st.session_state.analysis_settings = {
                'outlier_method': st.selectbox(
                    "Outlier Tespit Yöntemi",
                    ['iqr', 'zscore'],
                    index=0,
                    help="Outlier tespiti için kullanılacak yöntem"
                ),
                'cluster_count': st.slider(
                    "Küme Sayısı",
                    min_value=2,
                    max_value=10,
                    value=4,
                    help="Kümeleme analizi için küme sayısı"
                ),
                'confidence_level': st.slider(
                    "Güven Düzeyi (%)",
                    min_value=90,
                    max_value=99,
                    value=95,
                    step=1,
                    help="İstatistiksel analizler için güven düzeyi"
                )
            }
            
            # İşlem butonları
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Veriyi Yenile", use_container_width=True):
                    st.session_state.df = load_excel_file(st.session_state.uploaded_file)
                    st.session_state.data_profile = generate_data_profile(st.session_state.df)
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Veriyi Temizle", use_container_width=True):
                    st.session_state.df = None
                    st.session_state.uploaded_file = None
                    st.session_state.data_profile = {}
                    st.rerun()

def render_main_content():
    """Ana içeriği render et"""
    if st.session_state.df is None:
        render_welcome_screen()
    else:
        render_data_analysis()

def render_welcome_screen():
    """Hoş geldiniz ekranını render et"""
    st.markdown('<h1 class="main-header">🏥 Pharma Market Analytics Platform</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #6B7280; margin-bottom: 3rem;'>
        <p style='font-size: 1.2rem;'>Profesyonel Excel veri işleme, analiz ve görselleştirme platformu</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Özellikler grid'i
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>📊 Veri Analizi</h3>
            <p>• Eksik veri analizi</p>
            <p>• Outlier tespiti</p>
            <p>• İstatistiksel analiz</p>
            <p>• Korelasyon analizi</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>📈 Görselleştirme</h3>
            <p>• İnteraktif grafikler</p>
            <p>• Zaman serisi analizi</p>
            <p>• Dağılım grafikleri</p>
            <p>• Heatmap'ler</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h3>📋 Raporlama</h3>
            <p>• HTML raporları</p>
            <p>• Excel export</p>
            <p>• PDF raporları</p>
            <p>• Otomatik öngörüler</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Hızlı başlangıç
    st.markdown("---")
    st.markdown("## 🚀 Hızlı Başlangıç")
    
    # Örnek veri setleri
    st.markdown("### Örnek Veri Setleri ile Başlayın:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💊 Satış Verileri", use_container_width=True):
            # Örnek veri oluştur
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
            sample_data = pd.DataFrame({
                'TARIH': np.random.choice(dates, 1000),
                'URUN_KATEGORI': np.random.choice(['Ağrı Kesici', 'Vitamin', 'Antibiyotik', 'Krem', 'Şurup'], 1000),
                'SATIS_MIKTAR': np.random.randint(10, 1000, 1000),
                'SATIS_TUTAR': np.random.uniform(100, 5000, 1000),
                'BOLGE': np.random.choice(['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya'], 1000),
                'MIKTAR_BIRIM': np.random.choice(['Adet', 'Kutu', 'Şişe'], 1000)
            })
            
            st.session_state.df = sample_data
            st.session_state.data_profile = generate_data_profile(sample_data)
            st.rerun()
    
    with col2:
        if st.button("🏥 Hasta Verileri", use_container_width=True):
            np.random.seed(42)
            sample_data = pd.DataFrame({
                'HASTA_ID': range(1, 1001),
                'YAS': np.random.randint(18, 80, 1000),
                'CINSIYET': np.random.choice(['Erkek', 'Kadın'], 1000),
                'TESHIS': np.random.choice(['Grip', 'Hipertansiyon', 'Diyabet', 'Alerji', 'Migren'], 1000),
                'ILAC_ADI': np.random.choice(['Parol', 'Augmentin', 'Ventolin', 'Insulin', 'Aspirin'], 1000),
                'DOZ': np.random.choice(['1x1', '2x1', '3x1', 'Günde 1'], 1000),
                'TEDAVI_SURESI': np.random.randint(1, 30, 1000),
                'MUAYENE_UCRETI': np.random.uniform(50, 500, 1000)
            })
            
            st.session_state.df = sample_data
            st.session_state.data_profile = generate_data_profile(sample_data)
            st.rerun()
    
    with col3:
        if st.button("📦 Stok Verileri", use_container_width=True):
            np.random.seed(42)
            sample_data = pd.DataFrame({
                'URUN_KODU': [f'PRD{str(i).zfill(5)}' for i in range(1, 1001)],
                'URUN_ADI': [f'İlaç {i}' for i in range(1, 1001)],
                'KATEGORI': np.random.choice(['Ağrı Kesici', 'Vitamin', 'Antibiyotik', 'Krem', 'Şurup'], 1000),
                'STOK_MIKTAR': np.random.randint(0, 1000, 1000),
                'MIN_STOK': np.random.randint(10, 100, 1000),
                'MAX_STOK': np.random.randint(500, 2000, 1000),
                'SON_GIRIS_TARIHI': pd.date_range('2023-01-01', periods=1000, freq='D'),
                'BIRIM_FIYAT': np.random.uniform(5, 200, 1000),
                'TOPLAM_DEGER': np.random.uniform(1000, 50000, 1000)
            })
            
            st.session_state.df = sample_data
            st.session_state.data_profile = generate_data_profile(sample_data)
            st.rerun()
    
    # Kullanım kılavuzu
    st.markdown("---")
    st.markdown("## 📖 Kullanım Kılavuzu")
    
    with st.expander("Nasıl Kullanılır?", expanded=False):
        st.markdown("""
        1. **Veri Yükleme**: Sol taraftaki panelden Excel/CSV dosyanızı yükleyin
        2. **Sütun Seçimi**: Analiz etmek istediğiniz sütunları seçin
        3. **Analiz Ayarları**: Outlier tespiti, kümeleme gibi ayarları yapılandırın
        4. **Görselleştirme**: İnteraktif grafiklerle verinizi keşfedin
        5. **Raporlama**: Analiz sonuçlarını HTML veya Excel formatında indirin
        
        ### Desteklenen Özellikler:
        - **Veri Temizleme**: Eksik veri doldurma, outlier temizleme
        - **İstatistiksel Analiz**: Korelasyon, regresyon, hipotez testleri
        - **Makine Öğrenmesi**: Kümeleme, sınıflandırma, regresyon
        - **Görselleştirme**: İnteraktif grafikler, heatmap'ler, dağılım grafikleri
        - **Raporlama**: HTML, Excel, PDF raporları
        
        ### İpuçları:
        - Büyük veri setleri için satır sınırlaması kullanın
        - Otomatik öngörüleri dikkate alın
        - Farklı görselleştirme türlerini deneyin
        """)

def render_data_analysis():
    """Veri analiz ekranını render et"""
    df = st.session_state.df
    profile = st.session_state.data_profile
    
    # Üst bilgi
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f'<h2 class="sub-header">📊 {st.session_state.uploaded_file.name}</h2>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Satır Sayısı", f"{df.shape[0]:,}")
    
    with col3:
        st.metric("Sütun Sayısı", df.shape[1])
    
    # Tab navigasyonu
    tabs = st.tabs(["📋 Genel Bakış", "📈 Görselleştirme", "🔍 Detaylı Analiz", "📊 İstatistikler", "📋 Rapor"])
    
    # Tab 1: Genel Bakış
    with tabs[0]:
        render_overview_tab(df, profile)
    
    # Tab 2: Görselleştirme
    with tabs[1]:
        render_visualization_tab(df)
    
    # Tab 3: Detaylı Analiz
    with tabs[2]:
        render_detailed_analysis_tab(df)
    
    # Tab 4: İstatistikler
    with tabs[3]:
        render_statistics_tab(df, profile)
    
    # Tab 5: Rapor
    with tabs[4]:
        render_report_tab(df, profile)

def render_overview_tab(df, profile):
    """Genel bakış tab'ını render et"""
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Veri önizleme
        st.markdown("### 📋 Veri Önizleme")
        
        # Filtreleme seçenekleri
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            show_columns = st.multiselect(
                "Gösterilecek Sütunlar",
                options=list(df.columns),
                default=st.session_state.selected_columns[:10] if st.session_state.selected_columns else list(df.columns)[:10]
            )
        
        with filter_col2:
            sort_column = st.selectbox(
                "Sıralama Sütunu",
                options=show_columns if show_columns else list(df.columns),
                index=0
            )
            sort_ascending = st.checkbox("Artan Sırala", value=True)
        
        # Veriyi filtrele ve sırala
        display_df = df[show_columns] if show_columns else df
        
        if sort_column in display_df.columns:
            display_df = display_df.sort_values(by=sort_column, ascending=sort_ascending)
        
        # Dataframe'i göster
        st.dataframe(
            display_df.head(st.session_state.get('row_limit', 1000)),
            use_container_width=True,
            height=400
        )
        
        # Veri bilgileri
        st.markdown("### ℹ️ Veri Bilgileri")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Toplam Hücre", f"{profile['shape'][0] * profile['shape'][1]:,}")
        
        with info_col2:
            st.metric("Bellek Kullanımı", f"{profile['memory_usage']:.2f} MB")
        
        with info_col3:
            missing_pct = profile['missing_values']['missing_percentage']
            st.metric("Eksik Veri", f"{missing_pct:.2f}%")
    
    with col2:
        # Hızlı analiz
        st.markdown("### ⚡ Hızlı Analiz")
        
        # Veri tipleri
        st.markdown("#### 📊 Veri Tipleri")
        dtype_counts = df.dtypes.value_counts()
        
        for dtype, count in dtype_counts.items():
            st.progress(count / len(df.columns), text=f"{dtype}: {count}")
        
        # Top sütunlar
        st.markdown("#### 🏆 En Önemli Sütunlar")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # En yüksek varyans
            variances = {}
            for col in numeric_cols[:5]:  # İlk 5 sayısal sütun
                try:
                    variances[col] = df[col].var()
                except:
                    pass
            
            if variances:
                max_var_col = max(variances, key=variances.get)
                st.info(f"**En değişken sütun:** {max_var_col} (Varyans: {variances[max_var_col]:.2f})")
        
        # Öneriler
        st.markdown("#### 💡 Öneriler")
        
        if profile['missing_values']['missing_percentage'] > 5:
            st.warning(f"Veri setinde %{profile['missing_values']['missing_percentage']:.1f} eksik veri var. Temizleme önerilir.")
        
        if 'outliers' in profile['data_quality']:
            total_outliers = sum(info['count'] for info in profile['data_quality']['outliers'].values())
            if total_outliers > 0:
                st.warning(f"{total_outliers} outlier tespit edildi. İncelenmesi önerilir.")
        
        # Hızlı işlemler
        st.markdown("#### ⚙️ Hızlı İşlemler")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Eksik Verileri Temizle", use_container_width=True):
                with st.spinner("Eksik veriler temizleniyor..."):
                    cleaned_df = df.copy()
                    for col in cleaned_df.columns:
                        if cleaned_df[col].dtype == 'object':
                            cleaned_df[col] = cleaned_df[col].fillna('BİLİNMİYOR')
                        elif pd.api.types.is_numeric_dtype(cleaned_df[col]):
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                    
                    st.session_state.df = cleaned_df
                    st.session_state.data_profile = generate_data_profile(cleaned_df)
                    st.success("Eksik veriler temizlendi!")
                    st.rerun()
        
        with col2:
            if st.button("📥 CSV Olarak İndir", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="İndirmek için tıklayın",
                    data=csv,
                    file_name="veri_analizi.csv",
                    mime="text/csv"
                )

def render_visualization_tab(df):
    """Görselleştirme tab'ını render et"""
    st.markdown("### 📈 Veri Görselleştirme")
    
    # Görselleştirme türü seçimi
    viz_type = st.selectbox(
        "Görselleştirme Türü Seçin:",
        ["Korelasyon Heatmap", "Dağılım Grafiği", "Zaman Serisi", "Box Plot", "Scatter Plot", "Bar Chart", "Pie Chart"]
    )
    
    if viz_type == "Korelasyon Heatmap":
        fig = create_correlation_heatmap(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Korelasyon heatmap oluşturmak için yeterli sayısal veri yok.")
    
    elif viz_type == "Dağılım Grafiği":
        col1, col2 = st.columns(2)
        
        with col1:
            column = st.selectbox("Dağılımı görüntülenecek sütun:", options=list(df.columns))
        
        with col2:
            if pd.api.types.is_numeric_dtype(df[column]):
                bins = st.slider("Histogram bin sayısı:", min_value=10, max_value=100, value=30)
        
        fig = create_distribution_plot(df, column)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # İstatistikler
            if pd.api.types.is_numeric_dtype(df[column]):
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric("Ortalama", f"{df[column].mean():.2f}")
                
                with stats_col2:
                    st.metric("Medyan", f"{df[column].median():.2f}")
                
                with stats_col3:
                    st.metric("Std Sapma", f"{df[column].std():.2f}")
                
                with stats_col4:
                    st.metric("Çarpıklık", f"{df[column].skew():.2f}")
    
    elif viz_type == "Zaman Serisi":
        col1, col2 = st.columns(2)
        
        with col1:
            date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            if not date_columns:
                # Tarih formatındaki string kolonları deneyelim
                date_columns = [col for col in df.columns if 'TARIH' in col or 'DATE' in col]
            
            date_column = st.selectbox("Tarih sütunu:", options=date_columns if date_columns else list(df.columns))
        
        with col2:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            value_column = st.selectbox("Değer sütunu:", options=list(numeric_columns))
        
        fig = create_time_series_plot(df, date_column, value_column)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        col1, col2 = st.columns(2)
        
        with col1:
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            category_column = st.selectbox("Kategori sütunu:", options=list(categorical_columns))
        
        with col2:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            value_column = st.selectbox("Değer sütunu:", options=list(numeric_columns))
        
        fig = create_box_plot(df, category_column, value_column)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plot":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            x_column = st.selectbox("X ekseni:", options=list(numeric_columns))
        
        with col2:
            y_column = st.selectbox("Y ekseni:", options=list(numeric_columns))
        
        with col3:
            all_columns = list(df.columns)
            color_column = st.selectbox("Renk sütunu (opsiyonel):", options=['None'] + list(all_columns))
            color_column = None if color_column == 'None' else color_column
        
        fig = create_scatter_plot(df, x_column, y_column, color_column)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Korelasyon katsayısı
            if x_column != y_column:
                correlation = df[[x_column, y_column]].corr().iloc[0, 1]
                st.info(f"**Korelasyon katsayısı:** {correlation:.3f}")
    
    elif viz_type == "Bar Chart":
        col1, col2 = st.columns(2)
        
        with col1:
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            category_column = st.selectbox("Kategori sütunu:", options=list(categorical_columns))
        
        with col2:
            aggregation = st.selectbox("Toplama yöntemi:", options=['Sayı', 'Toplam', 'Ortalama', 'Medyan'])
        
        if category_column:
            if aggregation == 'Sayı':
                bar_data = df[category_column].value_counts().head(20)
            else:
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                value_column = st.selectbox("Değer sütunu:", options=list(numeric_columns))
                
                if aggregation == 'Toplam':
                    bar_data = df.groupby(category_column)[value_column].sum().head(20)
                elif aggregation == 'Ortalama':
                    bar_data = df.groupby(category_column)[value_column].mean().head(20)
                elif aggregation == 'Medyan':
                    bar_data = df.groupby(category_column)[value_column].median().head(20)
            
            fig = go.Figure(data=[go.Bar(
                x=bar_data.index,
                y=bar_data.values,
                marker_color='#667eea'
            )])
            
            fig.update_layout(
                title=f'{category_column} - {aggregation}',
                xaxis_title=category_column,
                yaxis_title=aggregation,
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Pie Chart":
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        category_column = st.selectbox("Kategori sütunu:", options=list(categorical_columns))
        
        if category_column:
            value_counts = df[category_column].value_counts().head(10)
            
            fig = go.Figure(data=[go.Pie(
                labels=value_counts.index,
                values=value_counts.values,
                hole=.3,
                marker_colors=px.colors.sequential.Viridis
            )])
            
            fig.update_layout(
                title=f'{category_column} Dağılımı',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_detailed_analysis_tab(df):
    """Detaylı analiz tab'ını render et"""
    st.markdown("### 🔍 Detaylı Analiz")
    
    analysis_type = st.selectbox(
        "Analiz Türü Seçin:",
        ["Kümeleme Analizi", "Outlier Tespiti", "Korelasyon Analizi", "Trend Analizi", "Anomali Tespiti"]
    )
    
    if analysis_type == "Kümeleme Analizi":
        st.markdown("#### 🎯 Kümeleme Analizi")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            st.warning("Kümeleme analizi için en az 2 sayısal sütun gereklidir.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_columns = st.multiselect(
                    "Kümeleme için sütunları seçin:",
                    options=list(numeric_df.columns),
                    default=list(numeric_df.columns)[:4]
                )
            
            with col2:
                n_clusters = st.slider("Küme sayısı:", min_value=2, max_value=10, value=4)
            
            if selected_columns and len(selected_columns) >= 2:
                analysis_df = df[selected_columns].dropna()
                
                if len(analysis_df) > 0:
                    with st.spinner("Kümeleme analizi yapılıyor..."):
                        # Standardizasyon
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(analysis_df)
                        
                        # K-Means
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(scaled_data)
                        
                        # PCA
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(scaled_data)
                        
                        # Sonuçları göster
                        result_df = analysis_df.copy()
                        result_df['Küme'] = clusters
                        result_df['PCA1'] = pca_result[:, 0]
                        result_df['PCA2'] = pca_result[:, 1]
                        
                        # Scatter plot
                        fig = px.scatter(
                            result_df,
                            x='PCA1',
                            y='PCA2',
                            color='Küme',
                            title='Kümeleme Sonuçları (PCA ile 2D)',
                            color_continuous_scale='Viridis'
                        )
                        
                        fig.update_layout(
                            template='plotly_white',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Küme istatistikleri
                        st.markdown("#### 📊 Küme İstatistikleri")
                        
                        cluster_stats = result_df.groupby('Küme').agg(['mean', 'count']).round(2)
                        st.dataframe(cluster_stats, use_container_width=True)
                        
                        # PCA açıklama oranı
                        st.info(f"**PCA Açıklama Oranı:** %{(pca.explained_variance_ratio_.sum() * 100):.1f}")
    
    elif analysis_type == "Outlier Tespiti":
        st.markdown("#### ⚠️ Outlier Tespiti")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.warning("Outlier tespiti için sayısal sütun bulunamadı.")
        else:
            method = st.selectbox("Tespit yöntemi:", ["IQR (Çeyrekler Arası Aralık)", "Z-Skor"])
            
            selected_column = st.selectbox("Analiz edilecek sütun:", options=list(numeric_df.columns))
            
            if selected_column:
                data = df[selected_column].dropna()
                
                if method == "IQR (Çeyrekler Arası Aralık)":
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[selected_column] < lower_bound) | (df[selected_column] > upper_bound)]
                
                else:  # Z-Skor
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(data))
                    outliers = df[z_scores > 3]
                
                # Sonuçları göster
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Toplam Veri", len(data))
                
                with col2:
                    st.metric("Outlier Sayısı", len(outliers))
                
                with col3:
                    outlier_percentage = (len(outliers) / len(data)) * 100
                    st.metric("Outlier Yüzdesi", f"{outlier_percentage:.2f}%")
                
                if len(outliers) > 0:
                    st.markdown("##### 📋 Outlier Detayları")
                    st.dataframe(outliers[[selected_column]].head(20), use_container_width=True)
                    
                    # Box plot ile görselleştirme
                    fig = go.Figure()
                    
                    fig.add_trace(go.Box(
                        y=data,
                        name=selected_column,
                        marker_color='#667eea',
                        boxmean='sd'
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_column} - Outlier Analizi',
                        yaxis_title=selected_column,
                        template='plotly_white',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Outlier'ları temizle butonu
                    if st.button("🧹 Outlier'ları Temizle"):
                        if method == "IQR (Çeyrekler Arası Aralık)":
                            cleaned_df = df[(df[selected_column] >= lower_bound) & (df[selected_column] <= upper_bound)].copy()
                        else:
                            cleaned_df = df[z_scores <= 3].copy()
                        
                        st.session_state.df = cleaned_df
                        st.session_state.data_profile = generate_data_profile(cleaned_df)
                        st.success(f"{len(outliers)} outlier temizlendi!")
                        st.rerun()
    
    elif analysis_type == "Korelasyon Analizi":
        st.markdown("#### 🔗 Korelasyon Analizi")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            st.warning("Korelasyon analizi için en az 2 sayısal sütun gereklidir.")
        else:
            # Korelasyon matrisi
            corr_matrix = numeric_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Korelasyon"),
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Korelasyon Matrisi',
                height=600,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # En yüksek korelasyonlar
            st.markdown("##### 🏆 En Yüksek Korelasyonlar")
            
            high_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if abs(corr_value) > 0.5:
                        high_correlations.append({
                            'Değişken 1': col1,
                            'Değişken 2': col2,
                            'Korelasyon': corr_value,
                            'Tip': 'Güçlü Pozitif' if corr_value > 0.8 else 
                                  'Orta Pozitif' if corr_value > 0.5 else
                                  'Güçlü Negatif' if corr_value < -0.8 else
                                  'Orta Negatif'
                        })
            
            if high_correlations:
                high_correlations_df = pd.DataFrame(high_correlations)
                high_correlations_df = high_correlations_df.sort_values('Korelasyon', key=abs, ascending=False)
                st.dataframe(high_correlations_df.head(10), use_container_width=True)
            else:
                st.info("0.5'ten yüksek korelasyon bulunamadı.")
    
    elif analysis_type == "Trend Analizi":
        st.markdown("#### 📈 Trend Analizi")
        
        # Tarih sütunu bul
        date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if not date_columns:
            date_columns = [col for col in df.columns if 'TARIH' in col or 'DATE' in col]
        
        if not date_columns:
            st.warning("Tarih sütunu bulunamadı.")
        else:
            date_column = st.selectbox("Tarih sütunu:", options=date_columns)
            
            # Sayısal sütunlar
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                st.warning("Sayısal sütun bulunamadı.")
            else:
                value_column = st.selectbox("Analiz edilecek sütun:", options=list(numeric_columns))
                
                # Tarihe göre grupla
                try:
                    df[date_column] = pd.to_datetime(df[date_column])
                    time_series = df.groupby(date_column)[value_column].sum().reset_index()
                    
                    # Trend analizi
                    from scipy import stats
                    
                    # Zaman indeksi
                    time_series['time_index'] = range(len(time_series))
                    
                    # Lineer regresyon
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        time_series['time_index'], 
                        time_series[value_column]
                    )
                    
                    # Sonuçları göster
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Trend Eğimi", f"{slope:.4f}")
                    
                    with col2:
                        trend_direction = "Artış" if slope > 0 else "Azalış" if slope < 0 else "Sabit"
                        st.metric("Trend Yönü", trend_direction)
                    
                    with col3:
                        st.metric("R² Değeri", f"{r_value**2:.4f}")
                    
                    with col4:
                        significance = "Anlamlı" if p_value < 0.05 else "Anlamlı Değil"
                        st.metric("İstatistiksel Anlamlılık", significance)
                    
                    # Trend grafiği
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=time_series[date_column],
                        y=time_series[value_column],
                        mode='lines+markers',
                        name='Gerçek Değerler',
                        line=dict(color='#667eea', width=2)
                    ))
                    
                    # Trend çizgisi
                    trend_line = intercept + slope * time_series['time_index']
                    
                    fig.add_trace(go.Scatter(
                        x=time_series[date_column],
                        y=trend_line,
                        mode='lines',
                        name='Trend Çizgisi',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f'{value_column} Trend Analizi',
                        xaxis_title='Tarih',
                        yaxis_title=value_column,
                        template='plotly_white',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Trend analizi sırasında hata oluştu: {str(e)}")

def render_statistics_tab(df, profile):
    """İstatistikler tab'ını render et"""
    st.markdown("### 📊 İstatistiksel Analiz")
    
    # İstatistiksel test seçimi
    test_type = st.selectbox(
        "İstatistiksel Test Seçin:",
        ["Temel İstatistikler", "Normallik Testi", "Varyans Analizi", "Ki-Kare Testi", "Regresyon Analizi"]
    )
    
    if test_type == "Temel İstatistikler":
        st.markdown("#### 📈 Temel İstatistikler")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.warning("Sayısal sütun bulunamadı.")
        else:
            selected_column = st.selectbox("Analiz edilecek sütun:", options=list(numeric_df.columns))
            
            if selected_column:
                data = df[selected_column].dropna()
                
                # İstatistikleri hesapla
                stats_data = {
                    'Ortalama': data.mean(),
                    'Medyan': data.median(),
                    'Mod': data.mode().iloc[0] if not data.mode().empty else 'N/A',
                    'Standart Sapma': data.std(),
                    'Varyans': data.var(),
                    'Minimum': data.min(),
                    'Maksimum': data.max(),
                    'Çeyrekler Arası Aralık (IQR)': data.quantile(0.75) - data.quantile(0.25),
                    'Çarpıklık (Skewness)': data.skew(),
                    'Basıklık (Kurtosis)': data.kurtosis(),
                    'Değişim Katsayısı': (data.std() / data.mean() * 100) if data.mean() != 0 else 0
                }
                
                # İstatistikleri göster
                col1, col2 = st.columns(2)
                
                with col1:
                    for stat_name, stat_value in list(stats_data.items())[:6]:
                        if isinstance(stat_value, float):
                            st.metric(stat_name, f"{stat_value:.4f}")
                        else:
                            st.metric(stat_name, str(stat_value))
                
                with col2:
                    for stat_name, stat_value in list(stats_data.items())[6:]:
                        if isinstance(stat_value, float):
                            st.metric(stat_name, f"{stat_value:.4f}")
                        else:
                            st.metric(stat_name, str(stat_value))
                
                # Dağılım grafiği
                fig = make_subplots(
                    rows=1, 
                    cols=2,
                    subplot_titles=('Histogram', 'Box Plot')
                )
                
                # Histogram
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        nbinsx=30,
                        name='Histogram',
                        marker_color='#667eea'
                    ),
                    row=1, col=1
                )
                
                # Box plot
                fig.add_trace(
                    go.Box(
                        y=data,
                        name='Box Plot',
                        marker_color='#764ba2'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif test_type == "Normallik Testi":
        st.markdown("#### 📏 Normallik Testi")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.warning("Sayısal sütun bulunamadı.")
        else:
            selected_column = st.selectbox("Test edilecek sütun:", options=list(numeric_df.columns))
            
            if selected_column:
                data = df[selected_column].dropna()
                
                from scipy import stats
                
                # Shapiro-Wilk testi
                shapiro_stat, shapiro_p = stats.shapiro(data)
                
                # Kolmogorov-Smirnov testi
                ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                
                # Sonuçları göster
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Shapiro-Wilk İstatistiği", f"{shapiro_stat:.4f}")
                    st.metric("Shapiro-Wilk p-değeri", f"{shapiro_p:.4f}")
                
                with col2:
                    st.metric("Kolmogorov-Smirnov İstatistiği", f"{ks_stat:.4f}")
                    st.metric("Kolmogorov-Smirnov p-değeri", f"{ks_p:.4f}")
                
                # Normallik değerlendirmesi
                alpha = 0.05
                is_normal_shapiro = shapiro_p > alpha
                is_normal_ks = ks_p > alpha
                
                if is_normal_shapiro and is_normal_ks:
                    st.success("✅ Veri normal dağılıma uyuyor (p > 0.05)")
                else:
                    st.warning("⚠️ Veri normal dağılıma uymuyor (p ≤ 0.05)")
                
                # Q-Q plot
                fig = make_subplots(
                    rows=1, 
                    cols=2,
                    subplot_titles=('Q-Q Plot', 'Dağılım Karşılaştırması')
                )
                
                # Q-Q plot
                from scipy import stats
                qq = stats.probplot(data, dist="norm")
                
                fig.add_trace(
                    go.Scatter(
                        x=qq[0][0],
                        y=qq[0][1],
                        mode='markers',
                        name='Gözlenen',
                        marker=dict(color='#667eea')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=qq[0][0],
                        y=qq[0][0] * qq[1][0] + qq[1][1],
                        mode='lines',
                        name='Teorik',
                        line=dict(color='red', dash='dash')
                    ),
                    row=1, col=1
                )
                
                fig.update_xaxes(title_text="Teorik Değerler", row=1, col=1)
                fig.update_yaxes(title_text="Gözlenen Değerler", row=1, col=1)
                
                # Histogram ve normal dağılım
                hist_data, bin_edges = np.histogram(data, bins=30, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Normal dağılım PDF
                pdf = stats.norm.pdf(bin_centers, data.mean(), data.std())
                
                fig.add_trace(
                    go.Bar(
                        x=bin_centers,
                        y=hist_data,
                        name='Histogram',
                        marker_color='#667eea',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=bin_centers,
                        y=pdf,
                        mode='lines',
                        name='Normal Dağılım',
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text=selected_column, row=1, col=2)
                fig.update_yaxes(title_text="Yoğunluk", row=1, col=2)
                
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)

def render_report_tab(df, profile):
    """Rapor tab'ını render et"""
    st.markdown("### 📋 Analiz Raporu")
    
    # Rapor özelleştirme
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_stats = st.checkbox("İstatistikler", value=True)
    
    with col2:
        include_visuals = st.checkbox("Görselleştirmeler", value=True)
    
    with col3:
        include_insights = st.checkbox("Öngörüler", value=True)
    
    # Rapor oluşturma butonu
    if st.button("📄 Rapor Oluştur", use_container_width=True):
        with st.spinner("Rapor oluşturuluyor..."):
            # HTML raporu oluştur
            html_report = generate_html_report(df, profile, {})
            
            # Raporu göster
            st.markdown("---")
            st.markdown("### 📊 Rapor Önizleme")
            
            # Raporu iframe içinde göster
            import base64
            from io import StringIO
            
            # HTML'i base64'e çevir
            b64 = base64.b64encode(html_report.encode()).decode()
            
            # Iframe ile göster
            st.markdown(
                f'<iframe src="data:text/html;base64,{b64}" width="100%" height="600" '
                'style="border: 1px solid #e2e8f0; border-radius: 10px;"></iframe>',
                unsafe_allow_html=True
            )
            
            # İndirme butonları
            st.markdown("---")
            st.markdown("### 📥 Raporu İndir")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # HTML olarak indir
                st.download_button(
                    label="📄 HTML Olarak İndir",
                    data=html_report,
                    file_name="pharma_analiz_raporu.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col2:
                # Excel olarak indir
                excel_data = export_to_excel(df, profile, {})
                st.download_button(
                    label="📊 Excel Olarak İndir",
                    data=excel_data,
                    file_name="pharma_analiz_raporu.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col3:
                # CSV olarak indir
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📁 CSV Olarak İndir",
                    data=csv_data,
                    file_name="pharma_verileri.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Hızlı özet
    st.markdown("---")
    st.markdown("### ⚡ Hızlı Özet")
    
    # Özet metrikler
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Veri Kalitesi", 
                 f"{100 - profile['missing_values']['missing_percentage']:.1f}%",
                 delta=f"%{profile['missing_values']['missing_percentage']:.1f} eksik veri")
    
    with col2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            avg_correlation = df[numeric_cols].corr().abs().mean().mean()
            st.metric("Ortalama Korelasyon", f"{avg_correlation:.3f}")
        else:
            st.metric("Ortalama Korelasyon", "N/A")
    
    with col3:
        total_outliers = 0
        if 'outliers' in profile['data_quality']:
            total_outliers = sum(info['count'] for info in profile['data_quality']['outliers'].values())
        st.metric("Outlier Sayısı", f"{total_outliers:,}")
    
    with col4:
        unique_values = sum(df[col].nunique() for col in df.columns)
        st.metric("Benzersiz Değer", f"{unique_values:,}")
    
    # Anahtar öngörüler
    st.markdown("#### 💡 Anahtar Öngörüler")
    
    insights = []
    
    # Eksik veri öngörüsü
    if profile['missing_values']['missing_percentage'] > 5:
        insights.append(f"⚠️ **Eksik Veri Uyarısı**: Veri setinde %{profile['missing_values']['missing_percentage']:.1f} eksik veri bulunuyor. Temizleme önerilir.")
    
    # Outlier öngörüsü
    if 'outliers' in profile['data_quality']:
        outlier_cols = [col for col, info in profile['data_quality']['outliers'].items() if info['count'] > 0]
        if outlier_cols:
            insights.append(f"⚠️ **Outlier Tespiti**: {len(outlier_cols)} sütunda outlier bulundu. İncelenmesi önerilir.")
    
    # Yüksek korelasyon öngörüsü
    if 'high_correlations' in profile:
        strong_correlations = [c for c in profile['high_correlations'] if c['type'] == 'strong']
        if strong_correlations:
            insights.append(f"🔗 **Güçlü Korelasyon**: {len(strong_correlations)} çift değişken arasında güçlü korelasyon bulundu.")
    
    # Veri dağılımı öngörüsü
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewed_cols = []
    for col in numeric_cols[:5]:
        try:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                skewed_cols.append((col, skewness))
        except:
            pass
    
    if skewed_cols:
        most_skewed = max(skewed_cols, key=lambda x: abs(x[1]))
        insights.append(f"📊 **Çarpık Dağılım**: '{most_skewed[0]}' sütunu yüksek çarpıklığa sahip ({most_skewed[1]:.2f}).")
    
    # Öngörüleri göster
    if insights:
        for insight in insights:
            st.info(insight)
    else:
        st.success("✅ Veri seti genel olarak iyi durumda. Önemli bir sorun tespit edilmedi.")

# Ana uygulama
def main():
    """Ana uygulama"""
    # CSS enjeksiyonu
    inject_custom_css()
    
    # Session state başlatma
    initialize_session_state()
    
    # Layout
    render_sidebar()
    render_main_content()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #718096; padding: 2rem;'>
            <p>Pharma Market Analytics Platform v3.0 © 2026 | Tüm hakları saklıdır.</p>
            <p style='font-size: 0.9rem;'>Profesyonel veri analiz platformu</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Uygulamayı başlat
if __name__ == "__main__":
    main()


