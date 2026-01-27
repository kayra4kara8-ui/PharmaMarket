# app.py - DataInsight Pro Analytics Dashboard
"""
DataInsight Pro Analytics Dashboard
Tek Dosyalık Streamlit Uygulaması - Tüm özellikler bu dosyada
"""

# ============================================================================
# BÖLÜM 1: IMPORTS & CONFIGURATION
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io
import base64
import json
import time
import datetime
import math
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit sayfa konfigürasyonu
st.set_page_config(
    page_title="DataInsight Pro Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/streamlit',
        'Report a bug': "https://github.com/streamlit",
        'About': "# DataInsight Pro v1.0\nProfesyonel Veri Analiz Platformu"
    }
)

# CSS stilleri
def inject_custom_css():
    """Özel CSS stilleri enjekte et"""
    st.markdown("""
    <style>
    /* Ana stil */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563EB 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4B5563;
        margin-bottom: 1rem;
    }
    
    /* Kart stilleri */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 4px solid #2563EB;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1F2937;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Buton stilleri */
    .stButton > button {
        border-radius: 8px;
        border: none;
        background: linear-gradient(90deg, #2563EB 0%, #3B82F6 100%);
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2563EB 0%, #7C3AED 100%);
    }
    
    /* Sekme stilleri */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #F3F4F6;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    /* Veri tablosu stilleri */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Responsive düzen */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .metric-value {
            font-size: 1.5rem;
        }
    }
    
    /* Dark mode desteği */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: #1F2937;
            color: #F9FAFB;
        }
        .metric-value {
            color: #F9FAFB;
        }
        .metric-label {
            color: #D1D5DB;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# BÖLÜM 2: HELPER FUNCTIONS - VERİ İŞLEME
# ============================================================================

def detect_encoding(file_content: bytes) -> str:
    """Dosya encoding'ini otomatik tespit et"""
    encodings = ['utf-8', 'ISO-8859-1', 'windows-1254', 'cp1252', 'latin-1']
    
    for encoding in encodings:
        try:
            file_content.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return 'utf-8'

def load_excel_file(uploaded_file) -> pd.DataFrame:
    """Excel dosyasını yükle ve optimize et"""
    try:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Dosya yükleniyor...")
        progress_bar.progress(10)
        
        # Dosyayı yükle
        if uploaded_file.name.endswith('.csv'):
            # CSV için
            encoding = detect_encoding(uploaded_file.read())
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
        else:
            # Excel için
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        status_text.text("Veri işleniyor...")
        progress_bar.progress(30)
        
        # Kolon isimlerini temizle
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        # Veri tiplerini optimize et
        df = optimize_data_types(df)
        
        status_text.text("Analiz hazırlanıyor...")
        progress_bar.progress(70)
        
        progress_bar.progress(100)
        status_text.text("✓ Dosya başarıyla yüklendi!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        return df
        
    except Exception as e:
        st.error(f"Dosya yükleme hatası: {str(e)}")
        return pd.DataFrame()

def optimize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Veri tiplerini optimize et - memory kullanımını azalt"""
    if df.empty:
        return df
    
    for col in df.columns:
        try:
            col_type = str(df[col].dtype)
            
            if col_type == 'object':
                # String kolonları optimize et
                try:
                    if df[col].nunique() / len(df) < 0.5:  # Eğer unique değerler azsa
                        df[col] = df[col].astype('category')
                except:
                    pass
                    
            elif 'int' in col_type:
                # Integer kolonları optimize et
                try:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                except:
                    pass
                    
            elif 'float' in col_type:
                # Float kolonları optimize et
                try:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                except:
                    pass
        except:
            continue
    
    return df

def analyze_missing_values(df: pd.DataFrame) -> Dict:
    """Eksik veri analizi"""
    if df.empty:
        return {
            'total_cells': 0,
            'total_missing': 0,
            'missing_percentage': 0,
            'columns': {}
        }
    
    try:
        total_cells = df.size  # np.prod yerine df.size kullan
        total_missing = df.isnull().sum().sum()
        
        missing_stats = {
            'total_cells': int(total_cells),
            'total_missing': int(total_missing),
            'missing_percentage': (total_missing / total_cells * 100) if total_cells > 0 else 0,
            'columns': {}
        }
        
        # Kolon bazlı eksik veriler
        for col in df.columns:
            try:
                missing_count = df[col].isnull().sum()
                missing_percentage = (missing_count / len(df)) * 100 if len(df) > 0 else 0
                
                missing_stats['columns'][col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_percentage),
                    'dtype': str(df[col].dtype)
                }
            except:
                missing_stats['columns'][col] = {
                    'count': 0,
                    'percentage': 0.0,
                    'dtype': 'unknown'
                }
        
        return missing_stats
    except Exception as e:
        return {
            'total_cells': 0,
            'total_missing': 0,
            'missing_percentage': 0,
            'columns': {}
        }

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> Dict:
    """IQR yöntemiyle outlier tespiti"""
    if df.empty or column not in df.columns:
        return {}
    
    try:
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {}
        
        # NaN değerleri temizle
        col_data = df[column].dropna()
        if len(col_data) < 4:  # IQR için en az 4 veri gerekli
            return {}
        
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:  # Eğer IQR 0 ise outlier yok
            return {
                'column': column,
                'outlier_count': 0,
                'outlier_percentage': 0,
                'lower_bound': float(Q1),
                'upper_bound': float(Q3),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'mean': float(col_data.mean()),
                'median': float(col_data.median())
            }
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        
        return {
            'column': column,
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(col_data)) * 100,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'mean': float(col_data.mean()),
            'median': float(col_data.median())
        }
    except:
        return {}

def calculate_basic_statistics(df: pd.DataFrame) -> Dict:
    """Temel istatistikleri hesapla"""
    stats_dict = {}
    
    if df.empty:
        return stats_dict
    
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Sayısal kolonlar için
                col_data = df[col].dropna()
                
                if len(col_data) > 0:
                    stats_dict[col] = {
                        'count': int(len(col_data)),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()) if len(col_data) > 1 else 0.0,
                        'min': float(col_data.min()),
                        '25%': float(col_data.quantile(0.25)),
                        '50%': float(col_data.quantile(0.5)),
                        '75%': float(col_data.quantile(0.75)),
                        'max': float(col_data.max()),
                        'skewness': float(col_data.skew()) if len(col_data) > 2 else 0.0,
                        'kurtosis': float(col_data.kurtosis()) if len(col_data) > 3 else 0.0,
                        'unique': int(col_data.nunique())
                    }
                else:
                    stats_dict[col] = {
                        'count': 0,
                        'mean': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        '25%': 0.0,
                        '50%': 0.0,
                        '75%': 0.0,
                        'max': 0.0,
                        'skewness': 0.0,
                        'kurtosis': 0.0,
                        'unique': 0
                    }
            else:
                # Kategorik kolonlar için
                col_data = df[col].dropna()
                value_counts = col_data.value_counts()
                
                top_value = None
                top_frequency = 0
                
                if len(value_counts) > 0:
                    try:
                        top_value = str(value_counts.index[0])
                        top_frequency = int(value_counts.iloc[0])
                    except:
                        top_value = None
                        top_frequency = 0
                
                stats_dict[col] = {
                    'count': int(len(col_data)),
                    'unique': int(col_data.nunique()),
                    'top_value': top_value,
                    'top_frequency': top_frequency
                }
        except Exception as e:
            # Hata durumunda basit istatistikler
            stats_dict[col] = {
                'count': int(df[col].count()),
                'unique': int(df[col].nunique()),
                'error': str(e)
            }
    
    return stats_dict

def generate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Korelasyon matrisi oluştur"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return pd.DataFrame()
        
        # NaN değerleri temizle
        numeric_df = numeric_df.dropna()
        
        if len(numeric_df) < 2:
            return pd.DataFrame()
        
        correlation_matrix = numeric_df.corr()
        return correlation_matrix
    except:
        return pd.DataFrame()

def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """Veri kalitesi analizi"""
    quality_report = {
        'overall_score': 0,
        'quality_grade': 'F',
        'dimensions': {}
    }
    
    if df.empty:
        return quality_report
    
    try:
        # Completeness (Tamlık)
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_score = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
        
        quality_report['dimensions']['completeness'] = {
            'score': completeness_score,
            'missing_cells': int(missing_cells),
            'total_cells': int(total_cells)
        }
        
        # Uniqueness (Benzersizlik)
        duplicate_rows = df.duplicated().sum()
        uniqueness_score = ((len(df) - duplicate_rows) / len(df)) * 100 if len(df) > 0 else 0
        
        quality_report['dimensions']['uniqueness'] = {
            'score': uniqueness_score,
            'duplicate_rows': int(duplicate_rows),
            'total_rows': len(df)
        }
        
        # Validity (Geçerlilik) - basit kontrol
        validity_score = 100
        
        quality_report['dimensions']['validity'] = {
            'score': validity_score,
            'issues': []
        }
        
        # Consistency (Tutarlılık)
        consistency_issues = []
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if (df[col] < 0).any() and col.upper() not in ['CHANGE', 'DIFFERENCE', 'VARIANCE']:
                        consistency_issues.append(f"{col}: Negatif değerler var")
            except:
                continue
        
        consistency_score = max(0, 100 - len(consistency_issues) * 10)
        quality_report['dimensions']['consistency'] = {
            'score': consistency_score,
            'issues': consistency_issues
        }
        
        # Overall score
        weights = {'completeness': 0.3, 'uniqueness': 0.3, 'validity': 0.2, 'consistency': 0.2}
        overall_score = 0
        weight_sum = 0
        
        for dim, weight in weights.items():
            if dim in quality_report['dimensions']:
                overall_score += quality_report['dimensions'][dim]['score'] * weight
                weight_sum += weight
        
        if weight_sum > 0:
            overall_score = overall_score / weight_sum
        else:
            overall_score = 0
        
        quality_report['overall_score'] = overall_score
        quality_report['quality_grade'] = get_quality_grade(overall_score)
        
    except Exception as e:
        quality_report['error'] = str(e)
    
    return quality_report

def get_quality_grade(score: float) -> str:
    """Kalite skoruna göre harf notu ver"""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"

def generate_data_profile(df: pd.DataFrame) -> Dict:
    """Kapsamlı veri profili oluştur"""
    profile = {}
    
    if df.empty:
        return {
            'overview': {
                'num_rows': 0,
                'num_columns': 0,
                'memory_usage_mb': 0.0,
                'data_types': {}
            },
            'statistics': {},
            'missing_values': analyze_missing_values(df),
            'data_quality': analyze_data_quality(df)
        }
    
    try:
        # Dataset overview
        dtypes = df.dtypes.apply(lambda x: str(x)).value_counts().to_dict()
        
        profile['overview'] = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'data_types': dtypes
        }
        
        # Statistics
        profile['statistics'] = calculate_basic_statistics(df)
        
        # Missing values
        profile['missing_values'] = analyze_missing_values(df)
        
        # Data quality
        profile['data_quality'] = analyze_data_quality(df)
        
    except Exception as e:
        profile['error'] = str(e)
    
    return profile

# ============================================================================
# BÖLÜM 3: GÖRSELLEŞTİRME FONKSİYONLARI
# ============================================================================

def create_interactive_datagrid(df: pd.DataFrame, height: int = 400):
    """İnteraktif veri ızgarası oluştur"""
    try:
        st.dataframe(
            df,
            height=height,
            use_container_width=True,
            hide_index=False
        )
    except:
        st.dataframe(df, use_container_width=True)

def create_metric_cards(profile: Dict):
    """Metrik kartları oluştur"""
    if not profile:
        return
    
    try:
        overview = profile.get('overview', {})
        quality = profile.get('data_quality', {})
        
        cols = st.columns(4)
        
        with cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{overview.get('num_rows', 0):,}</div>
                <div class="metric-label">SATIR SAYISI</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{overview.get('num_columns', 0)}</div>
                <div class="metric-label">KOLON SAYISI</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{overview.get('memory_usage_mb', 0)} MB</div>
                <div class="metric-label">BELLEK KULLANIMI</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[3]:
            grade = quality.get('quality_grade', 'N/A')
            score = quality.get('overall_score', 0)
            color = "#10B981" if grade in ["A+", "A"] else "#F59E0B" if grade in ["B", "C"] else "#EF4444"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {color};">{grade} ({round(score, 1)}%)</div>
                <div class="metric-label">VERİ KALİTESİ</div>
            </div>
            """, unsafe_allow_html=True)
    except:
        st.warning("Metrik kartları oluşturulamadı.")

def plot_missing_values_heatmap(df: pd.DataFrame):
    """Eksik veri heatmap'i oluştur"""
    if df.empty:
        st.info("Veri seti boş.")
        return
    
    try:
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            st.info("✅ Veri setinde eksik değer bulunmuyor.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Eksik veri matrisi
        missing_matrix = df.isnull()
        
        # Heatmap
        sns.heatmap(missing_matrix, cbar=False, cmap=['#10B981', '#EF4444'], 
                    yticklabels=False, ax=ax)
        
        ax.set_title('Eksik Veri Dağılımı', fontsize=14, fontweight='bold')
        ax.set_xlabel('Kolonlar', fontsize=12)
        
        # Kolon isimlerini ayarla
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close(fig)
        
    except Exception as e:
        st.error(f"Eksik veri grafiği oluşturulamadı: {str(e)}")

def plot_correlation_heatmap(df: pd.DataFrame):
    """Korelasyon heatmap'i oluştur"""
    try:
        corr_matrix = generate_correlation_matrix(df)
        
        if corr_matrix.empty:
            st.warning("Korelasyon analizi için yeterli sayısal veri bulunamadı.")
            return
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(
                title="Korelasyon",
                titleside="right"
            )
        ))
        
        fig.update_layout(
            title='Korelasyon Matrisi',
            title_font_size=16,
            width=800,
            height=600,
            xaxis_title="Kolonlar",
            yaxis_title="Kolonlar",
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Korelasyon grafiği oluşturulamadı: {str(e)}")

def plot_distribution_chart(df: pd.DataFrame, column: str):
    """Dağılım grafiği oluştur"""
    if df.empty or column not in df.columns:
        st.warning(f"'{column}' kolonu bulunamadı.")
        return
    
    try:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Sayısal kolon için histogram
            plot_data = df[column].dropna()
            
            if len(plot_data) == 0:
                st.warning(f"'{column}' kolonunda görselleştirme için veri bulunamadı.")
                return
            
            fig = px.histogram(
                df, 
                x=column,
                nbins=min(50, len(plot_data)),
                title=f'{column} Dağılımı',
                marginal='box',
                opacity=0.7
            )
            
            # Ortalama ve medyan çizgileri
            try:
                mean_val = plot_data.mean()
                median_val = plot_data.median()
                
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                             annotation_text=f"Ortalama: {mean_val:.2f}")
                fig.add_vline(x=median_val, line_dash="dot", line_color="green",
                             annotation_text=f"Medyan: {median_val:.2f}")
            except:
                pass
            
        else:
            # Kategorik kolon için bar chart
            value_counts = df[column].value_counts().head(20)
            
            if len(value_counts) == 0:
                st.warning(f"'{column}' kolonunda görselleştirme için veri bulunamadı.")
                return
            
            fig = px.bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                title=f'{column} - En Sık Değerler (Top 20)',
                labels={'x': column, 'y': 'Frekans'}
            )
        
        fig.update_layout(
            showlegend=False,
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Dağılım grafiği oluşturulamadı: {str(e)}")

def plot_comparative_bar_chart(df: pd.DataFrame, category_col: str, value_col: str):
    """Karşılaştırmalı bar chart"""
    if df.empty or category_col not in df.columns or value_col not in df.columns:
        st.warning("Gerekli kolonlar bulunamadı.")
        return
    
    try:
        # Gruplama ve toplama
        grouped_data = df.groupby(category_col)[value_col].sum().reset_index()
        grouped_data = grouped_data.sort_values(value_col, ascending=False).head(15)
        
        if grouped_data.empty:
            st.info("Görselleştirme için yeterli veri bulunamadı.")
            return
        
        fig = px.bar(
            grouped_data,
            x=category_col,
            y=value_col,
            title=f'{category_col} Bazında {value_col} Toplamları',
            color=value_col,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title=category_col,
            yaxis_title=f'Toplam {value_col}',
            template='plotly_white',
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Karşılaştırmalı bar chart oluşturulamadı: {str(e)}")

def plot_scatter_with_regression(df: pd.DataFrame, x_col: str, y_col: str):
    """Regresyon çizgili scatter plot"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        st.warning("Gerekli kolonlar bulunamadı.")
        return
    
    try:
        if not (pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])):
            st.warning("Lütfen sayısal kolonlar seçin.")
            return
        
        # NaN değerleri temizle
        plot_df = df[[x_col, y_col]].dropna()
        
        if len(plot_df) < 2:
            st.warning("Görselleştirme için yeterli veri bulunamadı.")
            return
        
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            title=f'{x_col} vs {y_col} - Korelasyon Analizi',
            trendline='ols',
            trendline_color_override='red',
            opacity=0.6
        )
        
        # Korelasyon katsayısı
        try:
            correlation = plot_df[x_col].corr(plot_df[y_col])
            
            fig.add_annotation(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"Korelasyon: {correlation:.3f}",
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                opacity=0.8
            )
        except:
            pass
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Scatter plot oluşturulamadı: {str(e)}")

def plot_pie_chart(df: pd.DataFrame, column: str, limit: int = 10):
    """Pasta grafiği oluştur"""
    if df.empty or column not in df.columns:
        st.warning(f"'{column}' kolonu bulunamadı.")
        return
    
    try:
        value_counts = df[column].value_counts().head(limit)
        
        if len(value_counts) == 0:
            st.info(f"'{column}' kolonunda veri bulunamadı.")
            return
        
        fig = px.pie(
            names=value_counts.index.astype(str),
            values=value_counts.values,
            title=f'{column} Dağılımı (Top {limit})',
            hole=0.3
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            pull=[0.1 if i == 0 else 0 for i in range(len(value_counts))]
        )
        
        fig.update_layout(
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Pasta grafiği oluşturulamadı: {str(e)}")

# ============================================================================
# BÖLÜM 4: GELİŞMİŞ ANALİZ FONKSİYONLARI
# ============================================================================

def perform_segmentation_analysis(df: pd.DataFrame, n_clusters: int = 3):
    """Segmentasyon analizi (K-means clustering)"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return None, "Segmentasyon için en az 2 sayısal kolon gereklidir."
        
        # NaN değerleri temizle
        analysis_df = df[numeric_cols].dropna()
        
        if len(analysis_df) < n_clusters:
            return None, f"Segmentasyon için en az {n_clusters} satır gereklidir."
        
        if len(analysis_df) < 10:  # Çok az veri için uyarı
            return None, "Segmentasyon için daha fazla veri gereklidir."
        
        # Veriyi ölçekle
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(analysis_df)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        clusters = kmeans.fit_predict(scaled_data)
        
        # PCA ile boyut indirgeme (görselleştirme için)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Sonuçları dataframe'e ekle
        result_df = analysis_df.copy()
        result_df['Cluster'] = clusters
        result_df['PC1'] = pca_result[:, 0]
        result_df['PC2'] = pca_result[:, 1]
        
        # Cluster istatistikleri
        cluster_stats = result_df.groupby('Cluster').agg(['mean', 'std', 'count']).round(2)
        
        return {
            'result_df': result_df,
            'cluster_stats': cluster_stats,
            'pca_result': pca_result,
            'clusters': clusters,
            'inertia': kmeans.inertia_
        }, None
        
    except Exception as e:
        return None, f"Segmentasyon analizi hatası: {str(e)}"

def calculate_trend_analysis(df: pd.DataFrame, date_col: str, value_col: str):
    """Trend analizi"""
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        return None
    
    try:
        # Tarih kolonunu datetime'a çevir
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # NaN tarihleri temizle
        trend_df = df.dropna(subset=[date_col, value_col]).copy()
        
        if len(trend_df) < 2:
            return None
        
        # Tarihe göre sırala
        trend_df = trend_df.sort_values(date_col)
        
        # Lineer regresyon için sayısal değerler
        trend_df['Days'] = (trend_df[date_col] - trend_df[date_col].min()).dt.days
        
        # Lineer regresyon
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            trend_df['Days'], trend_df[value_col]
        )
        
        # Trend çizgisi
        trend_df['Trend_Line'] = intercept + slope * trend_df['Days']
        
        # Büyüme oranları
        first_value = trend_df[value_col].iloc[0]
        last_value = trend_df[value_col].iloc[-1]
        
        if first_value != 0:
            total_growth = ((last_value - first_value) / first_value * 100)
        else:
            total_growth = 0
        
        # Aylık büyüme (ortalama)
        monthly_growth = (slope * 30)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'total_growth_pct': total_growth,
            'monthly_growth': monthly_growth,
            'trend_df': trend_df,
            'direction': 'ARTIŞ' if slope > 0 else 'AZALIŞ' if slope < 0 else 'STABİL'
        }
        
    except Exception as e:
        return None

def perform_statistical_tests(df: pd.DataFrame, col1: str, col2: str, test_type: str = 'ttest'):
    """İstatistiksel testler"""
    if df.empty or col1 not in df.columns or col2 not in df.columns:
        return None
    
    try:
        if not (pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2])):
            return "Lütfen sayısal kolonlar seçin."
        
        # NaN değerleri temizle
        data1 = df[col1].dropna()
        data2 = df[col2].dropna()
        
        if len(data1) < 2 or len(data2) < 2:
            return "Test için her grupta en az 2 gözlem gereklidir."
        
        if test_type == 'ttest':
            # T-test (bağımsız örneklem)
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
            test_name = "Bağımsız Örneklem T-Testi"
            
        elif test_type == 'mannwhitney':
            # Mann-Whitney U testi
            u_stat, p_value = stats.mannwhitneyu(data1, data2, nan_policy='omit')
            test_name = "Mann-Whitney U Testi"
            
        elif test_type == 'pearson':
            # Pearson korelasyon testi
            if len(data1) == len(data2):
                corr, p_value = stats.pearsonr(data1, data2)
                t_stat = corr
                test_name = "Pearson Korelasyon Testi"
            else:
                return "Korelasyon testi için eşit uzunlukta veri gereklidir."
        else:
            return "Geçersiz test türü."
        
        # Sonuç yorumu
        if p_value < 0.01:
            significance = "ÇOK ANLAMLI (p < 0.01)"
        elif p_value < 0.05:
            significance = "ANLAMLI (p < 0.05)"
        else:
            significance = "ANLAMSIZ (p ≥ 0.05)"
        
        return {
            'test_name': test_name,
            'test_statistic': float(t_stat) if test_type in ['ttest', 'pearson'] else float(u_stat),
            'p_value': float(p_value),
            'significance': significance,
            'sample_size1': len(data1),
            'sample_size2': len(data2),
            'mean1': float(data1.mean()) if len(data1) > 0 else 0,
            'mean2': float(data2.mean()) if len(data2) > 0 else 0,
            'std1': float(data1.std()) if len(data1) > 1 else 0,
            'std2': float(data2.std()) if len(data2) > 1 else 0
        }
        
    except Exception as e:
        return f"Test çalıştırılırken hata oluştu: {str(e)}"

def generate_automatic_insights(df: pd.DataFrame, profile: Dict) -> List[str]:
    """Otomatik insight üretimi"""
    insights = []
    
    if df.empty:
        insights.append("⚠️ **Boş Dataset**: Yüklenen dataset boş.")
        return insights
    
    try:
        # Dataset boyutu
        num_rows = len(df)
        num_cols = len(df.columns)
        
        insights.append(f"📊 **Dataset Boyutu**: {num_rows:,} satır ve {num_cols} kolon")
        
        # Eksik veri insights
        missing_stats = profile.get('missing_values', {})
        missing_pct = missing_stats.get('missing_percentage', 0)
        
        if missing_pct > 20:
            insights.append(f"⚠️ **Yüksek Eksik Veri**: Dataset'in %{missing_pct:.1f}'i eksik değer içeriyor")
        elif missing_pct > 0:
            insights.append(f"ℹ️ **Eksik Veri**: Dataset'in %{missing_pct:.1f}'i eksik değer içeriyor")
        else:
            insights.append("✅ **Tam Veri**: Dataset'te eksik değer bulunmuyor")
        
        # Sayısal kolonlar için insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # En yüksek varyanslı kolon
            variances = {}
            for col in numeric_cols[:5]:  # İlk 5 sayısal kolonu kontrol et
                try:
                    col_data = df[col].dropna()
                    if len(col_data) > 1:
                        variances[col] = col_data.var()
                except:
                    continue
            
            if variances:
                max_var_col = max(variances, key=variances.get)
                insights.append(f"📈 **En Değişken Kolon**: '{max_var_col}' en yüksek varyansa sahip")
            
            # Potansiyel outlier insights
            for col in numeric_cols[:3]:
                try:
                    outlier_info = detect_outliers_iqr(df, col)
                    if outlier_info.get('outlier_percentage', 0) > 5:
                        insights.append(f"🔍 **Outlier Uyarısı**: '{col}' kolonunda %{outlier_info['outlier_percentage']:.1f} outlier var")
                except:
                    continue
        
        # Kategorik kolonlar için insights
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols[:2]:
            try:
                unique_count = df[col].nunique()
                if unique_count == 1:
                    insights.append(f"ℹ️ **Sabit Değer**: '{col}' kolonunda tüm değerler aynı")
                elif unique_count < 10 and unique_count > 1:
                    insights.append(f"🏷️ **Sınırlı Kategori**: '{col}' kolonunda {unique_count} farklı kategori var")
            except:
                continue
        
        # Korelasyon insights
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = generate_correlation_matrix(df)
                if not corr_matrix.empty:
                    # En güçlü korelasyonu bul
                    strongest_corr = 0
                    strongest_pair = ()
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = abs(corr_matrix.iloc[i, j])
                            if not math.isnan(corr_val) and corr_val > strongest_corr:
                                strongest_corr = corr_val
                                strongest_pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                    
                    if strongest_corr > 0.7:
                        insights.append(f"🔗 **Güçlü Korelasyon**: '{strongest_pair[0]}' ve '{strongest_pair[1]}' arasında {strongest_corr:.2f} korelasyon var")
                    elif strongest_corr > 0.3:
                        insights.append(f"↔️ **Orta Korelasyon**: '{strongest_pair[0]}' ve '{strongest_pair[1]}' arasında {strongest_corr:.2f} korelasyon var")
            except:
                pass
        
    except Exception as e:
        insights.append(f"⚠️ **Insight Üretim Hatası**: {str(e)}")
    
    return insights

# ============================================================================
# BÖLÜM 5: UI COMPONENTS & DASHBOARD
# ============================================================================

def render_sidebar():
    """Sidebar bileşenlerini oluştur"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 1.8rem; margin-bottom: 0.5rem;">📊 DataInsight Pro</h1>
            <p style="color: #6B7280; font-size: 0.9rem;">Profesyonel Veri Analiz Platformu</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dosya yükleme bölümü
        st.markdown("### 📁 Veri Yükleme")
        
        uploaded_file = st.file_uploader(
            "Excel veya CSV dosyası yükleyin",
            type=['xlsx', 'xls', 'csv'],
            help="Excel veya CSV dosyası yükleyin"
        )
        
        if uploaded_file is not None:
            if 'current_df' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
                try:
                    with st.spinner("Dosya yükleniyor..."):
                        # Dosyayı yükle
                        df = load_excel_file(uploaded_file)
                        
                        if not df.empty:
                            st.session_state['original_df'] = df.copy()
                            st.session_state['current_df'] = df.copy()
                            st.session_state['uploaded_file_name'] = uploaded_file.name
                            st.session_state['data_profile'] = generate_data_profile(df)
                            st.success(f"✓ {uploaded_file.name} yüklendi!")
                        else:
                            st.error("Dosya yüklenemedi veya boş.")
                except Exception as e:
                    st.error(f"Dosya yükleme hatası: {str(e)}")
        
        # Dataset bilgileri
        if 'current_df' in st.session_state and st.session_state['current_df'] is not None:
            st.markdown("---")
            st.markdown("### 📋 Dataset Bilgileri")
            
            df = st.session_state['current_df']
            if not df.empty:
                profile = st.session_state.get('data_profile', {})
                overview = profile.get('overview', {})
                
                st.info(f"""
                **Dosya**: {st.session_state.get('uploaded_file_name', 'Bilinmiyor')}
                
                **Satır**: {overview.get('num_rows', 0):,}
                **Kolon**: {overview.get('num_columns', 0)}
                **Bellek**: {overview.get('memory_usage_mb', 0)} MB
                """)
                
                # Quick filters
                st.markdown("---")
                st.markdown("### 🔍 Hızlı Filtreler")
                
                # Kolon seçimi için multiselect
                available_columns = df.columns.tolist()
                if available_columns:
                    selected_columns = st.multiselect(
                        "Görüntülenecek Kolonlar",
                        options=available_columns,
                        default=available_columns[:min(8, len(available_columns))],
                        help="Analiz için kolonları seçin"
                    )
                    
                    if selected_columns:
                        st.session_state['selected_columns'] = selected_columns
                
                # Satır sayısı sınırı
                row_limit = st.slider(
                    "Görüntülenecek Satır Sayısı",
                    min_value=10,
                    max_value=min(10000, len(df)),
                    value=min(1000, len(df)),
                    step=10
                )
                st.session_state['row_limit'] = row_limit
                
                # Temizleme seçenekleri
                st.markdown("---")
                st.markdown("### 🧹 Veri Temizleme")
                
                if st.button("Eksik Değerleri Temizle", use_container_width=True):
                    if 'current_df' in st.session_state:
                        df_clean = st.session_state['current_df'].copy()
                        before = df_clean.isnull().sum().sum()
                        df_clean = df_clean.dropna()
                        after = df_clean.isnull().sum().sum()
                        st.session_state['current_df'] = df_clean
                        st.session_state['data_profile'] = generate_data_profile(df_clean)
                        st.success(f"✓ {before - after} eksik değer temizlendi!")
                        st.rerun()
                
                if st.button("Duplicate Satırları Temizle", use_container_width=True):
                    if 'current_df' in st.session_state:
                        df_clean = st.session_state['current_df'].copy()
                        before = len(df_clean)
                        df_clean = df_clean.drop_duplicates()
                        after = len(df_clean)
                        st.session_state['current_df'] = df_clean
                        st.session_state['data_profile'] = generate_data_profile(df_clean)
                        st.success(f"✓ {before - after} duplicate satır temizlendi!")
                        st.rerun()
                
                # Reset butonu
                if st.button("Orijinal Veriye Dön", use_container_width=True, type="secondary"):
                    if 'original_df' in st.session_state:
                        st.session_state['current_df'] = st.session_state['original_df'].copy()
                        st.session_state['data_profile'] = generate_data_profile(st.session_state['current_df'])
                        st.success("✓ Orijinal veriye dönüldü!")
                        st.rerun()
            
            else:
                st.warning("Dataset boş.")
        
        # Export seçenekleri
        st.markdown("---")
        st.markdown("### 📤 Export")
        
        if 'current_df' in st.session_state and st.session_state['current_df'] is not None:
            df = st.session_state['current_df']
            if not df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="CSV İndir",
                        data=csv,
                        file_name="data_export.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Data')
                    
                    st.download_button(
                        label="Excel İndir",
                        data=buffer.getvalue(),
                        file_name="data_export.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        
        # Ayarlar
        st.markdown("---")
        st.markdown("### ⚙️ Ayarlar")
        
        theme = st.selectbox(
            "Tema",
            ["Light", "Dark", "High Contrast", "Pastel"],
            index=0
        )
        
        st.session_state['theme'] = theme

def render_data_explorer():
    """Veri keşif bölümünü oluştur"""
    if 'current_df' not in st.session_state or st.session_state['current_df'] is None:
        st.info("👈 Lütfen sol taraftan bir veri dosyası yükleyin")
        return
    
    df = st.session_state['current_df']
    if df.empty:
        st.warning("Yüklenen dataset boş. Lütfen başka bir dosya yükleyin.")
        return
    
    profile = st.session_state.get('data_profile', {})
    
    # Sekmeler
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Veri Keşfi", 
        "📈 Görselleştirme", 
        "🤖 Analizler",
        "📋 Rapor"
    ])
    
    with tab1:
        # Overview sekmesi
        st.markdown('<div class="main-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        # Metrik kartları
        create_metric_cards(profile)
        
        # Dataset önizleme
        st.markdown("### 📋 Dataset Önizleme")
        
        # Kolon filtreleme
        selected_columns = st.session_state.get('selected_columns', df.columns.tolist()[:min(8, len(df.columns))])
        row_limit = st.session_state.get('row_limit', min(1000, len(df)))
        
        if selected_columns:
            display_df = df[selected_columns].head(row_limit)
        else:
            display_df = df.head(row_limit)
        
        create_interactive_datagrid(display_df, height=400)
        
        st.caption(f"Toplam {len(df):,} satırdan {len(display_df):,} satır gösteriliyor")
        
        # Veri profili detayları
        st.markdown("### 📊 Veri Profili")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Veri Tipleri")
            if 'overview' in profile and 'data_types' in profile['overview']:
                for dtype, count in profile['overview']['data_types'].items():
                    st.metric(label=dtype, value=count)
        
        with col2:
            st.markdown("##### Eksik Değerler")
            missing_stats = profile.get('missing_values', {})
            st.metric(
                label="Eksik Hücre", 
                value=f"{missing_stats.get('missing_percentage', 0):.1f}%"
            )
            
            # Eksik değer detayları
            if missing_stats.get('columns'):
                missing_cols = {k: v for k, v in missing_stats['columns'].items() 
                              if v['percentage'] > 0}
                if missing_cols:
                    st.markdown("**Eksik değerli kolonlar:**")
                    for col, stats in list(missing_cols.items())[:5]:
                        st.caption(f"{col}: %{stats['percentage']:.1f} eksik")
    
    with tab2:
        # Veri keşfi sekmesi
        st.markdown('<div class="sub-header">Veri Keşfi ve İstatistikler</div>', unsafe_allow_html=True)
        
        # Kolon analizi
        st.markdown("### 🔬 Kolon Analizi")
        
        selected_col = st.selectbox(
            "Analiz Edilecek Kolon Seçin",
            options=df.columns.tolist(),
            index=0
        )
        
        if selected_col:
            col1, col2 = st.columns(2)
            
            with col1:
                # Temel istatistikler
                st.markdown("##### 📈 Temel İstatistikler")
                
                col_data = df[selected_col]
                
                if pd.api.types.is_numeric_dtype(col_data):
                    col_stats = profile.get('statistics', {}).get(selected_col, {})
                    
                    stats_cols = st.columns(2)
                    
                    with stats_cols[0]:
                        st.metric("Ortalama", f"{col_stats.get('mean', 0):.2f}")
                        st.metric("Minimum", f"{col_stats.get('min', 0):.2f}")
                        st.metric("Medyan", f"{col_stats.get('50%', 0):.2f}")
                    
                    with stats_cols[1]:
                        st.metric("Standart Sapma", f"{col_stats.get('std', 0):.2f}")
                        st.metric("Maksimum", f"{col_stats.get('max', 0):.2f}")
                        st.metric("Benzersiz Değer", col_stats.get('unique', 0))
                
                else:
                    value_counts = col_data.value_counts()
                    unique_count = col_data.nunique()
                    
                    # En sık değeri güvenli bir şekilde al
                    most_frequent_value = "N/A"
                    most_frequent_count = 0
                    
                    if len(value_counts) > 0:
                        try:
                            most_frequent_value = str(value_counts.index[0])
                            most_frequent_count = int(value_counts.iloc[0])
                        except:
                            pass
                    
                    st.metric("Benzersiz Değer", unique_count)
                    st.metric("En Sık Değer", most_frequent_value)
                    st.metric("En Sık Değer Frekansı", most_frequent_count)
            
            with col2:
                # Dağılım grafiği
                plot_distribution_chart(df, selected_col)
            
            # Outlier analizi
            if pd.api.types.is_numeric_dtype(df[selected_col]):
                st.markdown("### 📊 Outlier Analizi")
                
                outlier_info = detect_outliers_iqr(df, selected_col)
                
                if outlier_info:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Outlier Sayısı", outlier_info.get('outlier_count', 0))
                    
                    with col2:
                        st.metric("Outlier Yüzdesi", f"{outlier_info.get('outlier_percentage', 0):.1f}%")
                    
                    with col3:
                        st.metric("Alt Sınır", f"{outlier_info.get('lower_bound', 0):.2f}")
                    
                    with col4:
                        st.metric("Üst Sınır", f"{outlier_info.get('upper_bound', 0):.2f}")
        
        # Eksik veri analizi
        st.markdown("### 🔍 Eksik Veri Analizi")
        plot_missing_values_heatmap(df)
    
    with tab3:
        # Görselleştirme sekmesi
        st.markdown('<div class="sub-header">Veri Görselleştirme</div>', unsafe_allow_html=True)
        
        # Grafik türü seçimi
        chart_type = st.selectbox(
            "Grafik Türü Seçin",
            [
                "Korelasyon Heatmap",
                "Karşılaştırmalı Bar Chart",
                "Scatter Plot (Regresyonlu)",
                "Pasta Grafiği",
                "Dağılım Grafiği"
            ]
        )
        
        if chart_type == "Korelasyon Heatmap":
            plot_correlation_heatmap(df)
            
        elif chart_type == "Karşılaştırmalı Bar Chart":
            col1, col2 = st.columns(2)
            
            with col1:
                # Sadece kategorik kolonları göster
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if categorical_cols:
                    category_col = st.selectbox(
                        "Kategori Kolonu",
                        options=categorical_cols,
                        key="bar_category"
                    )
                else:
                    st.warning("Kategorik kolon bulunamadı.")
                    category_col = None
            
            with col2:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    value_col = st.selectbox(
                        "Değer Kolonu",
                        options=numeric_cols,
                        key="bar_value"
                    )
                else:
                    st.warning("Sayısal kolon bulunamadı.")
                    value_col = None
            
            if category_col and value_col:
                plot_comparative_bar_chart(df, category_col, value_col)
        
        elif chart_type == "Scatter Plot (Regresyonlu)":
            col1, col2 = st.columns(2)
            
            with col1:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    x_col = st.selectbox(
                        "X Ekseni",
                        options=numeric_cols,
                        key="scatter_x"
                    )
                else:
                    st.warning("Sayısal kolon bulunamadı.")
                    x_col = None
            
            with col2:
                if numeric_cols and x_col:
                    # X kolonu hariç diğer sayısal kolonlar
                    other_numeric = [col for col in numeric_cols if col != x_col]
                    if other_numeric:
                        y_col = st.selectbox(
                            "Y Ekseni",
                            options=other_numeric,
                            key="scatter_y"
                        )
                    else:
                        st.warning("İkinci sayısal kolon bulunamadı.")
                        y_col = None
                else:
                    y_col = None
            
            if x_col and y_col:
                plot_scatter_with_regression(df, x_col, y_col)
        
        elif chart_type == "Pasta Grafiği":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                category_col = st.selectbox(
                    "Kategori Kolonu",
                    options=categorical_cols
                )
                
                if category_col:
                    limit = st.slider("Gösterilecek Kategori Sayısı", 5, 20, 10)
                    plot_pie_chart(df, category_col, limit)
            else:
                st.warning("Kategorik kolon bulunamadı.")
        
        elif chart_type == "Dağılım Grafiği":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                numeric_col = st.selectbox(
                    "Sayısal Kolon",
                    options=numeric_cols
                )
                
                if numeric_col:
                    plot_distribution_chart(df, numeric_col)
            else:
                st.warning("Sayısal kolon bulunamadı.")
    
    with tab4:
        # Analizler sekmesi
        st.markdown('<div class="sub-header">Gelişmiş Analizler</div>', unsafe_allow_html=True)
        
        analysis_type = st.selectbox(
            "Analiz Türü Seçin",
            [
                "Otomatik Insights",
                "Segmentasyon Analizi",
                "Trend Analizi",
                "İstatistiksel Testler"
            ]
        )
        
        if analysis_type == "Otomatik Insights":
            st.markdown("### 🤖 Otomatik Insights")
            
            insights = generate_automatic_insights(df, profile)
            
            for insight in insights:
                st.info(insight)
            
            # Veri kalitesi detayları
            st.markdown("### 📊 Veri Kalitesi Raporu")
            
            quality_report = profile.get('data_quality', {})
            
            if quality_report:
                overall_score = quality_report.get('overall_score', 0)
                grade = quality_report.get('quality_grade', 'N/A')
                
                # Kalite skoru gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=overall_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Veri Kalite Skoru"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2563EB"},
                        'steps': [
                            {'range': [0, 60], 'color': "#EF4444"},
                            {'range': [60, 80], 'color': "#F59E0B"},
                            {'range': [80, 100], 'color': "#10B981"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': overall_score
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Kalite boyutları
                dimensions = quality_report.get('dimensions', {})
                
                cols = st.columns(4)
                dimension_names = list(dimensions.keys())[:4]  # En fazla 4 boyut göster
                
                for i, (col, dim_name) in enumerate(zip(cols, dimension_names)):
                    with col:
                        dim_score = dimensions[dim_name].get('score', 0)
                        st.metric(
                            label=dim_name.capitalize(),
                            value=f"{dim_score:.1f}%"
                        )
        
        elif analysis_type == "Segmentasyon Analizi":
            st.markdown("### 🎯 Segmentasyon Analizi (K-means Clustering)")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Segmentasyon için en az 2 sayısal kolon gereklidir.")
            else:
                n_clusters = st.slider("Segment Sayısı", 2, 10, 3)
                
                if st.button("Segmentasyon Analizi Yap", type="primary"):
                    with st.spinner("Segmentasyon analizi yapılıyor..."):
                        result, error = perform_segmentation_analysis(df, n_clusters)
                        
                        if error:
                            st.error(error)
                        elif result:
                            # Cluster görselleştirme
                            fig = px.scatter(
                                result['result_df'],
                                x='PC1',
                                y='PC2',
                                color='Cluster',
                                title='Segmentasyon Sonuçları (PCA Görünümü)',
                                hover_data=numeric_cols[:3]
                            )
                            
                            fig.update_layout(
                                template='plotly_white',
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Cluster istatistikleri
                            st.markdown("##### 📊 Segment İstatistikleri")
                            st.dataframe(result['cluster_stats'], use_container_width=True)
                            
                            st.success(f"✓ Segmentasyon başarıyla tamamlandı. Inertia: {result['inertia']:.2f}")
        
        elif analysis_type == "Trend Analizi":
            st.markdown("### 📈 Trend Analizi")
            
            # Kolon seçimi
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'tarih' in col.lower()]
            all_cols = df.columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if date_cols:
                    date_col = st.selectbox(
                        "Tarih Kolonu",
                        options=date_cols,
                        key="trend_date"
                    )
                else:
                    st.info("Tarih kolonu bulunamadı. Diğer kolonları deneyin.")
                    date_col = st.selectbox(
                        "Tarih Kolonu",
                        options=all_cols,
                        key="trend_date"
                    )
            
            with col2:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    value_col = st.selectbox(
                        "Analiz Edilecek Kolon",
                        options=numeric_cols,
                        key="trend_value"
                    )
                else:
                    st.warning("Sayısal kolon bulunamadı.")
                    value_col = None
            
            if date_col and value_col:
                if st.button("Trend Analizi Yap", type="primary"):
                    trend_result = calculate_trend_analysis(df, date_col, value_col)
                    
                    if trend_result:
                        # Trend sonuçları
                        st.markdown("##### 📊 Trend Sonuçları")
                        
                        cols = st.columns(4)
                        
                        with cols[0]:
                            st.metric(
                                label="Trend Yönü",
                                value=trend_result['direction'],
                                delta=f"{trend_result['slope']:.4f}"
                            )
                        
                        with cols[1]:
                            st.metric(
                                label="R² Değeri",
                                value=f"{trend_result['r_squared']:.3f}"
                            )
                        
                        with cols[2]:
                            st.metric(
                                label="Toplam Büyüme",
                                value=f"%{trend_result['total_growth_pct']:.1f}"
                            )
                        
                        with cols[3]:
                            p_val = trend_result['p_value']
                            significance = "Anlamlı" if p_val < 0.05 else "Anlamsız"
                            st.metric(
                                label="İstatistiksel Anlamlılık",
                                value=significance,
                                delta=f"p={p_val:.4f}"
                            )
                        
                        # Trend grafiği
                        try:
                            fig = px.line(
                                trend_result['trend_df'],
                                x=date_col,
                                y=[value_col, 'Trend_Line'],
                                title=f'{value_col} Trend Analizi',
                                labels={'value': value_col, 'variable': 'Seriler'}
                            )
                            
                            fig.update_layout(
                                template='plotly_white',
                                height=500,
                                legend_title_text='',
                                hovermode='x unified'
                            )
                            
                            fig.data[1].line.dash = 'dash'
                            fig.data[1].name = 'Trend Çizgisi'
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.error("Trend grafiği oluşturulamadı.")
                    else:
                        st.warning("Trend analizi yapılamadı. Lütfen verileri kontrol edin.")
        
        elif analysis_type == "İstatistiksel Testler":
            st.markdown("### 📊 İstatistiksel Testler")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("İstatistiksel test için en az 2 sayısal kolon gereklidir.")
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    col1_test = st.selectbox(
                        "Birinci Kolon",
                        options=numeric_cols,
                        key="test_col1"
                    )
                
                with col2:
                    col2_options = [col for col in numeric_cols if col != col1_test]
                    if col2_options:
                        col2_test = st.selectbox(
                            "İkinci Kolon",
                            options=col2_options,
                            key="test_col2"
                        )
                    else:
                        st.warning("İkinci kolon seçilemedi.")
                        col2_test = None
                
                with col3:
                    test_type = st.selectbox(
                        "Test Türü",
                        options=['ttest', 'mannwhitney', 'pearson'],
                        format_func=lambda x: {
                            'ttest': 'T-Testi',
                            'mannwhitney': 'Mann-Whitney U',
                            'pearson': 'Pearson Korelasyon'
                        }[x]
                    )
                
                if col1_test and col2_test:
                    if st.button("Testi Çalıştır", type="primary"):
                        test_result = perform_statistical_tests(df, col1_test, col2_test, test_type)
                        
                        if isinstance(test_result, dict):
                            # Test sonuçlarını göster
                            st.markdown("##### 📈 Test Sonuçları")
                            
                            result_cols = st.columns(2)
                            
                            with result_cols[0]:
                                st.metric(
                                    label="Test İstatistiği",
                                    value=f"{test_result['test_statistic']:.4f}"
                                )
                                st.metric(
                                    label="P Değeri",
                                    value=f"{test_result['p_value']:.4f}"
                                )
                            
                            with result_cols[1]:
                                st.metric(
                                    label="Anlamlılık",
                                    value=test_result['significance']
                                )
                                st.metric(
                                    label="Örneklem Büyüklüğü",
                                    value=f"{test_result['sample_size1']} vs {test_result['sample_size2']}"
                                )
                            
                            # Yorum
                            st.markdown("##### 💡 Yorum")
                            
                            if test_type == 'pearson':
                                corr = test_result['test_statistic']
                                if abs(corr) > 0.7:
                                    strength = "çok güçlü"
                                elif abs(corr) > 0.5:
                                    strength = "güçlü"
                                elif abs(corr) > 0.3:
                                    strength = "orta"
                                else:
                                    strength = "zayıf"
                                
                                direction = "pozitif" if corr > 0 else "negatif"
                                st.info(f"İki değişken arasında {strength} {direction} korelasyon bulunmaktadır.")
                            
                            elif test_type in ['ttest', 'mannwhitney']:
                                if test_result['p_value'] < 0.05:
                                    st.success("İki grup arasında istatistiksel olarak anlamlı fark vardır (p < 0.05).")
                                else:
                                    st.warning("İki grup arasında istatistiksel olarak anlamlı fark yoktur (p ≥ 0.05).")
                            
                            # Detaylı istatistikler
                            with st.expander("Detaylı İstatistikler"):
                                st.write(f"**{col1_test}:**")
                                st.write(f"- Ortalama: {test_result['mean1']:.2f}")
                                st.write(f"- Standart Sapma: {test_result['std1']:.2f}")
                                
                                st.write(f"**{col2_test}:**")
                                st.write(f"- Ortalama: {test_result['mean2']:.2f}")
                                st.write(f"- Standart Sapma: {test_result['std2']:.2f}")
                        elif isinstance(test_result, str):
                            st.error(test_result)
    
    with tab5:
        # Rapor sekmesi
        st.markdown('<div class="sub-header">Analiz Raporu</div>', unsafe_allow_html=True)
        
        # Rapor oluşturma
        report_title = st.text_input("Rapor Başlığı", "Veri Analiz Raporu")
        
        if st.button("📄 Rapor Oluştur", type="primary", use_container_width=True):
            with st.spinner("Rapor oluşturuluyor..."):
                try:
                    # Rapor içeriği
                    report_content = f"""
# {report_title}

**Oluşturulma Tarihi:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Dataset:** {st.session_state.get('uploaded_file_name', 'Bilinmiyor')}

---

## 1. Dataset Özeti

- **Toplam Satır:** {profile.get('overview', {}).get('num_rows', 0):,}
- **Toplam Kolon:** {profile.get('overview', {}).get('num_columns', 0)}
- **Bellek Kullanımı:** {profile.get('overview', {}).get('memory_usage_mb', 0)} MB
- **Veri Kalite Skoru:** {profile.get('data_quality', {}).get('overall_score', 0):.1f}% ({profile.get('data_quality', {}).get('quality_grade', 'N/A')})

---

## 2. Veri Kalitesi

**Eksik Veri:** %{profile.get('missing_values', {}).get('missing_percentage', 0):.1f}

**Veri Tipleri Dağılımı:**
"""
                    
                    # Veri tipleri
                    if 'overview' in profile and 'data_types' in profile['overview']:
                        for dtype, count in profile['overview']['data_types'].items():
                            report_content += f"- {dtype}: {count} kolon\n"
                    
                    report_content += "\n---\n"
                    
                    # Otomatik insights
                    report_content += "## 3. Önemli Bulgular\n\n"
                    
                    insights = generate_automatic_insights(df, profile)
                    for insight in insights:
                        # Markdown formatına çevir
                        insight_md = insight.replace("**", "**").replace("✅", "✓").replace("⚠️", "⚠")
                        report_content += f"- {insight_md}\n"
                    
                    report_content += "\n---\n"
                    
                    # Korelasyon özeti
                    report_content += "## 4. Korelasyon Analizi\n\n"
                    
                    corr_matrix = generate_correlation_matrix(df)
                    if not corr_matrix.empty:
                        # En güçlü 3 korelasyonu göster
                        correlations = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_value = corr_matrix.iloc[i, j]
                                if not math.isnan(corr_value):
                                    correlations.append({
                                        'var1': corr_matrix.columns[i],
                                        'var2': corr_matrix.columns[j],
                                        'value': corr_value
                                    })
                        
                        correlations.sort(key=lambda x: abs(x['value']), reverse=True)
                        
                        if correlations:
                            report_content += "**En Güçlü Korelasyonlar:**\n\n"
                            for corr in correlations[:3]:
                                strength = "Çok Güçlü" if abs(corr['value']) > 0.7 else "Güçlü" if abs(corr['value']) > 0.5 else "Orta" if abs(corr['value']) > 0.3 else "Zayıf"
                                report_content += f"- **{corr['var1']}** ↔ **{corr['var2']}**: {corr['value']:.3f} ({strength})\n"
                    
                    # Raporu göster
                    st.markdown(report_content)
                    
                    # İndirme butonu
                    st.download_button(
                        label="📥 Raporu İndir (Markdown)",
                        data=report_content,
                        file_name=f"data_analysis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Rapor oluşturulurken hata oluştu: {str(e)}")

def render_main_content():
    """Ana içeriği oluştur"""
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-header">📊 DataInsight Pro Analytics Dashboard</h1>
        <p style="color: #6B7280; font-size: 1.1rem;">
            Profesyonel veri analizi için tek dosyalık kapsamlı çözüm
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo verisi butonu
    if 'current_df' not in st.session_state or st.session_state['current_df'] is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     border-radius: 15px; color: white; margin-bottom: 2rem;">
                <h2 style="color: white; margin-bottom: 1rem;">🚀 Hemen Başlayın</h2>
                <p style="margin-bottom: 1.5rem;">Sol taraftan veri dosyanızı yükleyin veya demo verisi ile deneyin</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🎮 Demo Verisi ile Deneyin", use_container_width=True, type="primary"):
                try:
                    # Demo verisi oluştur
                    np.random.seed(42)
                    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
                    
                    demo_data = {
                        'Tarih': np.random.choice(dates, 1000),
                        'Ürün_Kategorisi': np.random.choice(['Elektronik', 'Giyim', 'Ev', 'Spor', 'Kitap'], 1000),
                        'Bölge': np.random.choice(['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya'], 1000),
                        'Satış_Miktarı': np.random.randint(10, 1000, 1000),
                        'Gelir': np.random.uniform(100, 10000, 1000).round(2),
                        'Maliyet': np.random.uniform(50, 5000, 1000).round(2),
                        'Müşteri_Puanı': np.random.uniform(1, 5, 1000).round(1),
                        'Promosyon_Kullanımı': np.random.choice(['Evet', 'Hayır'], 1000, p=[0.3, 0.7])
                    }
                    
                    demo_df = pd.DataFrame(demo_data)
                    demo_df['Kar'] = demo_df['Gelir'] - demo_df['Maliyet']
                    demo_df['Kar_Marjı'] = (demo_df['Kar'] / demo_df['Gelir'] * 100).round(2)
                    
                    st.session_state['original_df'] = demo_df.copy()
                    st.session_state['current_df'] = demo_df.copy()
                    st.session_state['uploaded_file_name'] = "demo_dataset.csv"
                    st.session_state['data_profile'] = generate_data_profile(demo_df)
                    
                    st.success("✅ Demo verisi yüklendi! Analiz bölümlerini keşfedin.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Demo verisi oluşturulurken hata: {str(e)}")
    
    # Ana içerik
    if 'current_df' in st.session_state and st.session_state['current_df'] is not None:
        render_data_explorer()

# ============================================================================
# BÖLÜM 6: ANA UYGULAMA
# ============================================================================

def main():
    """Ana uygulama fonksiyonu"""
    
    # Session state initialization
    if 'current_df' not in st.session_state:
        st.session_state['current_df'] = None
    if 'original_df' not in st.session_state:
        st.session_state['original_df'] = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state['uploaded_file_name'] = None
    if 'data_profile' not in st.session_state:
        st.session_state['data_profile'] = {}
    if 'selected_columns' not in st.session_state:
        st.session_state['selected_columns'] = []
    if 'row_limit' not in st.session_state:
        st.session_state['row_limit'] = 1000
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'Light'
    
    # CSS enjekte et
    inject_custom_css()
    
    # Layout
    try:
        render_sidebar()
        render_main_content()
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        st.info("Lütfen sayfayı yenileyin veya başka bir dosya yükleyin.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.9rem; padding: 1rem;">
        <p>DataInsight Pro v1.0 | Profesyonel Veri Analiz Platformu</p>
        <p>© 2024 Tüm hakları saklıdır | Geliştirici: DataInsight Team</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# UYGULAMAYI BAŞLAT
# ============================================================================

if __name__ == "__main__":
    main()
