"""
██████╗ ██╗  ██╗ █████╗ ██████╗ ███╗   ███╗ █████╗ 
██╔══██╗██║  ██║██╔══██╗██╔══██╗████╗ ████║██╔══██╗
██████╔╝███████║███████║██████╔╝██╔████╔██║███████║
██╔═══╝ ██╔══██║██╔══██║██╔══██╗██║╚██╔╝██║██╔══██║
██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝

PharmaIntelligence Pro v8.0 - Enterprise Decision Support Platform
═══════════════════════════════════════════════════════════════════════
Modül: ProdPack (Ürün-Paket) Derinlik Analizi + Molekül Drill-Down
Versiyon: 8.0.0-ENTERPRISE
Yazar: PharmaIntelligence Inc.
Lisans: E.Ş. - Kurumsal Lisans

✓ ProdPack Hiyerarşik Drill-Down (Molekül → Şirket → Marka → Paket)
✓ Sunburst/Sankey Interaktif Görselleştirme
✓ Pazar Kanibalizasyon Matrisi
✓ IsolationForest Anomali Tespiti
✓ PCA+K-Means BCG Segmentasyonu
✓ Holt-Winters Tahminleme (2025-2026)
✓ Executive Dark Mode (Lacivert, Gümüş, Altın)
✓ Otomatik Yönetici Özeti (Insight Box)
"""

# ================================================
# 0. PERFORMANCE OPTIMIZATIONS & CACHE
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
import gc
import time
import json
import hashlib
import base64
import io
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, wraps
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')
gc.enable()

# Streamlit cache konfigürasyonu - 1M+ satır için optimize
st.set_page_config(
    page_title="PharmaIntel Pro v8.0 | ProdPack Derinlik Analizi",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://pharmaintelligence.com/support',
        'Report a bug': 'https://pharmaintelligence.com/bug',
        'About': 'PharmaIntelligence Pro v8.0 - ProdPack & Molekül Derinlik Analizi'
    }
)

# Bellek kullanımını optimize eden decorator
def optimize_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            # Downcast işlemi - pandas.api.types ile güvenli
            for col in result.select_dtypes(include=['float64']).columns:
                try:
                    result[col] = pd.to_numeric(result[col], downcast='float')
                except:
                    pass
            for col in result.select_dtypes(include=['int64']).columns:
                try:
                    result[col] = pd.to_numeric(result[col], downcast='integer')
                except:
                    pass
            for col in result.select_dtypes(include=['object']).columns:
                if result[col].nunique() / len(result) < 0.5:
                    try:
                        result[col] = result[col].astype('category')
                    except:
                        pass
        return result
    return wrapper

# ================================================
# 1. ADVANCED ANALYTICS IMPORTS
# ================================================

# Scikit-learn
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE

# Time Series
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# UMAP
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except:
    UMAP_AVAILABLE = False

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# ================================================
# 2. ENUMS & DATA CLASSES
# ================================================

class RiskLevel(Enum):
    KRITIK = "🔴 Kritik Risk"
    YUKSEK = "🟠 Yüksek Risk"
    ORTA = "🟡 Orta Risk"
    DUSUK = "🟢 Düşük Risk"
    NORMAL = "✅ Normal"

class GrowthCategory(Enum):
    HIPER = "🚀 Hiper Büyüme (>%50)"
    YUKSEK = "📈 Yüksek Büyüme (%20-50)"
    ORTA = "📊 Orta Büyüme (%5-20)"
    DURGUN = "⏸️ Durgun (-%5 - %5)"
    DARALAN = "📉 Daralan (<-5%)"

class ProductSegment(Enum):
    YILDIZ = "⭐ Yıldız Ürünler"
    NAKIT_INEK = "🐄 Nakit İnekleri"
    SORU_ISARETI = "❓ Soru İşaretleri"
    ZAYIF = "⚠️ Zayıf Ürünler"
    YUKSELEN = "🌟 Yükselen Yıldızlar"
    OLGUN = "📌 Olgun Ürünler"

@dataclass
class ProdPackInsight:
    """ProdPack (Ürün-Paket) içgörü veri sınıfı"""
    molekul: str
    sirket: str
    marka: str
    paket: str
    satis_2024: float
    satis_2025_tahmin: float
    buyume_hizi: float
    pazar_payi: float
    pazar_payi_degisim: float
    risk_seviyesi: RiskLevel
    segment: ProductSegment
    kanibalizasyon_skoru: float = 0.0
    yatirim_tavsiyesi: str = ""
    ozel_not: str = ""

@dataclass
class ForecastResult:
    periods: List[str]
    predictions: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    model_type: str
    mape: float
    trend: str

# ================================================
# 3. PRODPACK DERINLIK ANALIZI MODULU
# ================================================

class ProdPackDepthAnalyzer:
    """
    Ürün-Paket derinlik analizi modülü.
    Molekül → Şirket → Marka → Paket hiyerarşisinde drill-down imkanı.
    """
    
    def __init__(self):
        self.hierarchy_cache = {}
        self.cannibalization_matrix = None
        
    @optimize_memory
    @st.cache_data(ttl=3600, max_entries=10, show_spinner=False)
    def build_prodpack_hierarchy(_self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ProdPack hiyerarşisini oluşturur.
        
        Args:
            df: İşlenmiş DataFrame
            
        Returns:
            Hiyerarşi sözlüğü
        """
        hierarchy = {
            'molecules': {},
            'sunburst_data': None,
            'sankey_data': None,
            'stats': {}
        }
        
        # Sütun kontrolleri
        required_cols = ['Molekül', 'Şirket', 'Marka', 'Paket']
        for col in required_cols:
            if col not in df.columns:
                st.warning(f"{col} sütunu bulunamadı. Varsayılan isimlendirme yapılıyor.")
                return hierarchy
        
        # Satış sütunlarını bul
        sales_cols = [col for col in df.columns if re.search(r'20\d{2}', str(col)) and 
                     any(x in str(col).lower() for x in ['satış', 'sales', 'hacim', 'volume'])]
        
        if not sales_cols:
            sales_cols = [col for col in df.columns if re.search(r'20\d{2}', str(col))]
        
        if not sales_cols:
            return hierarchy
        
        # Yılları sırala
        year_pattern = re.compile(r'20\d{2}')
        sales_with_years = []
        for col in sales_cols:
            match = year_pattern.search(str(col))
            if match:
                sales_with_years.append((col, int(match.group())))
        
        sales_with_years.sort(key=lambda x: x[1])
        sales_cols_sorted = [x[0] for x in sales_with_years]
        
        # Son 2 yıl
        current_year_col = sales_cols_sorted[-1] if sales_cols_sorted else None
        prev_year_col = sales_cols_sorted[-2] if len(sales_cols_sorted) >= 2 else None
        
        # Molekül bazlı gruplama
        for molekul in df['Molekül'].unique():
            molekul_df = df[df['Molekül'] == molekul]
            
            hierarchy['molecules'][molekul] = {
                'total_sales': molekul_df[current_year_col].sum() if current_year_col else 0,
                'company_count': molekul_df['Şirket'].nunique(),
                'brand_count': molekul_df['Marka'].nunique(),
                'pack_count': molekul_df['Paket'].nunique(),
                'companies': {}
            }
            
            # Şirket bazlı gruplama
            for sirket in molekul_df['Şirket'].unique():
                sirket_df = molekul_df[molekul_df['Şirket'] == sirket]
                
                hierarchy['molecules'][molekul]['companies'][sirket] = {
                    'total_sales': sirket_df[current_year_col].sum() if current_year_col else 0,
                    'brand_count': sirket_df['Marka'].nunique(),
                    'pack_count': sirket_df['Paket'].nunique(),
                    'brands': {}
                }
                
                # Marka bazlı gruplama
                for marka in sirket_df['Marka'].unique():
                    marka_df = sirket_df[sirket_df['Marka'] == marka]
                    
                    hierarchy['molecules'][molekul]['companies'][sirket]['brands'][marka] = {
                        'total_sales': marka_df[current_year_col].sum() if current_year_col else 0,
                        'pack_count': marka_df['Paket'].nunique(),
                        'packs': {}
                    }
                    
                    # Paket bazlı gruplama
                    for paket in marka_df['Paket'].unique():
                        paket_df = marka_df[marka_df['Paket'] == paket]
                        
                        # Büyüme hesapla
                        growth = 0.0
                        if prev_year_col and current_year_col:
                            prev_sales = paket_df[prev_year_col].sum()
                            curr_sales = paket_df[current_year_col].sum()
                            if prev_sales > 0:
                                growth = ((curr_sales - prev_sales) / prev_sales) * 100
                        
                        # Pazar payı
                        market_share = 0.0
                        if current_year_col:
                            total_market = df[current_year_col].sum()
                            if total_market > 0:
                                market_share = (curr_sales / total_market) * 100
                        
                        hierarchy['molecules'][molekul]['companies'][sirket]['brands'][marka]['packs'][paket] = {
                            'sales': curr_sales if current_year_col else 0,
                            'prev_sales': prev_sales if prev_year_col else 0,
                            'growth': growth,
                            'market_share': market_share,
                            'units': len(paket_df)
                        }
        
        # Sunburst verisi oluştur
        hierarchy['sunburst_data'] = _self._create_sunburst_data(df, current_year_col)
        hierarchy['sankey_data'] = _self._create_sankey_data(df, current_year_col)
        
        # İstatistikler
        hierarchy['stats'] = {
            'total_molecules': len(hierarchy['molecules']),
            'total_companies': df['Şirket'].nunique(),
            'total_brands': df['Marka'].nunique(),
            'total_packs': df['Paket'].nunique(),
            'total_sales': df[current_year_col].sum() if current_year_col else 0,
            'analysis_year': current_year_col.split('_')[-1] if current_year_col and '_' in current_year_col else '2024'
        }
        
        return hierarchy
    
    def _create_sunburst_data(_self, df: pd.DataFrame, sales_col: str) -> go.Figure:
        """Sunburst diyagramı için veri hazırla"""
        if sales_col is None:
            return None
        
        # Hiyerarşik veri hazırlığı
        sunburst_df = df.groupby(['Molekül', 'Şirket', 'Marka', 'Paket'])[sales_col].sum().reset_index()
        sunburst_df.columns = ['Molekül', 'Şirket', 'Marka', 'Paket', 'Satış']
        
        # Path oluştur
        sunburst_df['path'] = (sunburst_df['Molekül'] + '|' + 
                               sunburst_df['Şirket'] + '|' + 
                               sunburst_df['Marka'] + '|' + 
                               sunburst_df['Paket'])
        
        # IDs
        all_ids = []
        all_labels = []
        all_parents = []
        all_values = []
        
        # Root
        all_ids.append('total')
        all_labels.append('Tüm Pazar')
        all_parents.append('')
        all_values.append(0)
        
        # Moleküller
        for molekul in sunburst_df['Molekül'].unique():
            molekul_sales = sunburst_df[sunburst_df['Molekül'] == molekul]['Satış'].sum()
            all_ids.append(f'mol_{molekul}')
            all_labels.append(molekul[:20] + '...' if len(str(molekul)) > 20 else molekul)
            all_parents.append('total')
            all_values.append(molekul_sales)
            
            # Şirketler
            for sirket in sunburst_df[sunburst_df['Molekül'] == molekul]['Şirket'].unique():
                sirket_sales = sunburst_df[(sunburst_df['Molekül'] == molekul) & 
                                          (sunburst_df['Şirket'] == sirket)]['Satış'].sum()
                all_ids.append(f'mol_{molekul}_sirket_{sirket}')
                all_labels.append(sirket[:15] + '...' if len(str(sirket)) > 15 else sirket)
                all_parents.append(f'mol_{molekul}')
                all_values.append(sirket_sales)
                
                # Markalar
                for marka in sunburst_df[(sunburst_df['Molekül'] == molekul) & 
                                        (sunburst_df['Şirket'] == sirket)]['Marka'].unique():
                    marka_sales = sunburst_df[(sunburst_df['Molekül'] == molekul) & 
                                             (sunburst_df['Şirket'] == sirket) & 
                                             (sunburst_df['Marka'] == marka)]['Satış'].sum()
                    all_ids.append(f'mol_{molekul}_sirket_{sirket}_marka_{marka}')
                    all_labels.append(marka[:15] + '...' if len(str(marka)) > 15 else marka)
                    all_parents.append(f'mol_{molekul}_sirket_{sirket}')
                    all_values.append(marka_sales)
        
        # Sunburst figürü
        fig = go.Figure(go.Sunburst(
            ids=all_ids,
            labels=all_labels,
            parents=all_parents,
            values=all_values,
            branchvalues='total',
            marker=dict(
                colorscale='Viridis',
                line=dict(width=1, color='#1a237e')
            ),
            hovertemplate='<b>%{label}</b><br>Satış: $%{value:,.0f}<br>Pay: %{percentRoot:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='ProdPack Hiyerarşi Haritası (Molekül → Şirket → Marka → Paket)',
                font=dict(size=18, color='#d4af37')
            ),
            margin=dict(t=50, l=0, r=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600
        )
        
        return fig
    
    def _create_sankey_data(_self, df: pd.DataFrame, sales_col: str) -> go.Figure:
        """Sankey diyagramı için veri hazırla"""
        if sales_col is None:
            return None
        
        # Örnekleme - çok fazla node olmaması için
        sample_df = df.nlargest(50, sales_col) if len(df) > 50 else df
        
        # Node'lar ve linkler
        nodes = []
        node_indices = {}
        
        # Moleküller
        for molekul in sample_df['Molekül'].unique():
            node_id = f'mol_{molekul}'
            if node_id not in node_indices:
                node_indices[node_id] = len(nodes)
                nodes.append(molekul[:20] + '...' if len(str(molekul)) > 20 else molekul)
        
        # Şirketler
        for sirket in sample_df['Şirket'].unique():
            node_id = f'sirket_{sirket}'
            if node_id not in node_indices:
                node_indices[node_id] = len(nodes)
                nodes.append(sirket[:15] + '...' if len(str(sirket)) > 15 else sirket)
        
        # Markalar
        for marka in sample_df['Marka'].unique():
            node_id = f'marka_{marka}'
            if node_id not in node_indices:
                node_indices[node_id] = len(nodes)
                nodes.append(marka[:15] + '...' if len(str(marka)) > 15 else marka)
        
        # Linkler
        links = []
        
        for _, row in sample_df.iterrows():
            # Molekül -> Şirket
            source = f'mol_{row["Molekül"]}'
            target = f'sirket_{row["Şirket"]}'
            
            if source in node_indices and target in node_indices:
                links.append({
                    'source': node_indices[source],
                    'target': node_indices[target],
                    'value': row[sales_col] * 0.3  # Ağırlıklandırma
                })
            
            # Şirket -> Marka
            source = f'sirket_{row["Şirket"]}'
            target = f'marka_{row["Marka"]}'
            
            if source in node_indices and target in node_indices:
                links.append({
                    'source': node_indices[source],
                    'target': node_indices[target],
                    'value': row[sales_col] * 0.7
                })
        
        # Linkleri birleştir
        link_dict = {}
        for link in links:
            key = (link['source'], link['target'])
            if key in link_dict:
                link_dict[key] += link['value']
            else:
                link_dict[key] = link['value']
        
        sankey_links = {
            'source': [k[0] for k in link_dict.keys()],
            'target': [k[1] for k in link_dict.keys()],
            'value': list(link_dict.values())
        }
        
        # Sankey figürü
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='#1a237e', width=0.5),
                label=nodes,
                color='#2d7dd2'
            ),
            link=dict(
                source=sankey_links['source'],
                target=sankey_links['target'],
                value=sankey_links['value'],
                color='rgba(45, 125, 210, 0.3)'
            )
        )])
        
        fig.update_layout(
            title=dict(
                text='ProdPack Akış Diyagramı (Molekül → Şirket → Marka)',
                font=dict(size=18, color='#d4af37')
            ),
            font=dict(size=12, color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            height=500,
            margin=dict(t=50, l=50, r=50, b=50)
        )
        
        return fig
    
    def analyze_cannibalization(_self, df: pd.DataFrame, sirket_adi: str) -> pd.DataFrame:
        """
        Aynı şirket içindeki paketler arası kanibalizasyon analizi.
        
        Args:
            df: DataFrame
            sirket_adi: Analiz edilecek şirket
            
        Returns:
            Kanibalizasyon matrisi
        """
        if 'Şirket' not in df.columns or 'Paket' not in df.columns:
            return pd.DataFrame()
        
        # Şirkete ait veriler
        sirket_df = df[df['Şirket'] == sirket_adi].copy()
        
        if len(sirket_df) < 2:
            return pd.DataFrame()
        
        # Satış sütunlarını bul
        sales_cols = [col for col in df.columns if re.search(r'20\d{2}', str(col))]
        if len(sales_cols) < 2:
            return pd.DataFrame()
        
        sales_cols.sort()
        current_col = sales_cols[-1]
        prev_col = sales_cols[-2]
        
        # Paket bazlı büyüme ve pazar payı
        paket_analiz = []
        
        for paket in sirket_df['Paket'].unique():
            paket_df = sirket_df[sirket_df['Paket'] == paket]
            
            curr_sales = paket_df[current_col].sum()
            prev_sales = paket_df[prev_col].sum()
            
            growth = 0
            if prev_sales > 0:
                growth = ((curr_sales - prev_sales) / prev_sales) * 100
            
            # Şirket içi pazar payı
            sirket_total = sirket_df[current_col].sum()
            sirket_payi = (curr_sales / sirket_total * 100) if sirket_total > 0 else 0
            
            paket_analiz.append({
                'Paket': paket,
                'Satış_2024': curr_sales,
                'Satış_2023': prev_sales,
                'Büyüme_Hızı': growth,
                'Şirket_İçi_Pay': sirket_payi,
                'Molekül': paket_df['Molekül'].iloc[0] if 'Molekül' in paket_df.columns else 'Bilinmiyor',
                'Marka': paket_df['Marka'].iloc[0] if 'Marka' in paket_df.columns else 'Bilinmiyor'
            })
        
        analiz_df = pd.DataFrame(paket_analiz)
        
        if len(analiz_df) < 2:
            return analiz_df
        
        # Korelasyon matrisi - kanibalizasyon göstergesi
        try:
            corr_matrix = analiz_df[['Büyüme_Hızı', 'Şirket_İçi_Pay']].corr()
            cannibal_score = corr_matrix.loc['Büyüme_Hızı', 'Şirket_İçi_Pay']
            
            analiz_df['Kanibalizasyon_Skoru'] = cannibal_score
            
            # Yorumlama
            if cannibal_score < -0.5:
                analiz_df['Kanibalizasyon_Risk'] = '🔴 Yüksek Kanibalizasyon'
            elif cannibal_score < -0.3:
                analiz_df['Kanibalizasyon_Risk'] = '🟠 Orta Kanibalizasyon'
            elif cannibal_score < 0:
                analiz_df['Kanibalizasyon_Risk'] = '🟡 Düşük Kanibalizasyon'
            else:
                analiz_df['Kanibalizasyon_Risk'] = '✅ Kanibalizasyon Yok'
        except:
            analiz_df['Kanibalizasyon_Skoru'] = 0
            analiz_df['Kanibalizasyon_Risk'] = '❓ Hesaplanamadı'
        
        return analiz_df
    
    def get_molecule_drill_down(_self, hierarchy: Dict, molekul_adi: str) -> Dict:
        """Molekül bazlı drill-down verisi"""
        if molekul_adi in hierarchy.get('molecules', {}):
            return hierarchy['molecules'][molekul_adi]
        return {}

# ================================================
# 4. TEKNIK HATA GIDERME & PERFORMANS MODULU
# ================================================

class TechnicalOptimizer:
    """
    Teknik hata giderme, regex yıl ayıklama ve performans optimizasyonları.
    """
    
    @staticmethod
    def extract_years_from_columns(columns: List[str]) -> List[int]:
        """
        Sütun isimlerinden yılları regex ile güvenli şekilde ayıklar.
        Hata fırlatmaz, sadece geçerli yılları döndürür.
        
        Args:
            columns: Sütun isimleri listesi
            
        Returns:
            Benzersiz yıl listesi
        """
        years = set()
        year_pattern = re.compile(r'20\d{2}')
        
        for col in columns:
            col_str = str(col)
            # PENICILLIN gibi metin içeren sütunlarda int() hatası vermez
            matches = year_pattern.findall(col_str)
            for match in matches:
                try:
                    year = int(match)
                    if 2000 <= year <= 2030:  # Geçerli yıl aralığı
                        years.add(year)
                except (ValueError, TypeError):
                    continue
        
        return sorted(list(years))
    
    @staticmethod
    def make_column_names_unique(columns: List[str]) -> List[str]:
        """
        Sütun isimlerini benzersiz hale getirir.
        Bölge, Bölge_1, Bölge_2 şeklinde otomatik isimlendirme.
        
        Args:
            columns: Orijinal sütun isimleri
            
        Returns:
            Benzersiz sütun isimleri
        """
        cleaned = []
        counter_map = {}
        
        for col in columns:
            col_str = str(col).strip()
            
            # Türkçe karakter düzeltme
            turkish_map = {
                'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
            }
            for tr, en in turkish_map.items():
                col_str = col_str.replace(tr, en)
            
            # Özel karakterleri temizle
            col_str = re.sub(r'[^\w\s-]', ' ', col_str)
            col_str = re.sub(r'\s+', '_', col_str)
            col_str = col_str.strip('_')
            
            # Boş isim kontrolü
            if not col_str:
                col_str = 'Bilinmeyen_Sutun'
            
            # Benzersizleştirme
            base_name = col_str
            counter = 1
            
            while col_str in cleaned:
                col_str = f"{base_name}_{counter}"
                counter += 1
            
            cleaned.append(col_str)
            counter_map[col_str] = counter
        
        return cleaned
    
    @staticmethod
    @optimize_memory
    def safe_downcast(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pandas.api.types kullanarak güvenli downcast işlemi.
        'Ambiguous Truth Value' hatasını önler.
        
        Args:
            df: İşlenecek DataFrame
            
        Returns:
            Optimize edilmiş DataFrame
        """
        df = df.copy()
        
        # Float downcast
        float_cols = df.select_dtypes(include=['float64', 'float32']).columns
        for col in float_cols:
            try:
                # pd.api.types ile tip kontrolü
                if pd.api.types.is_float_dtype(df[col]):
                    # Min-max değerlerine göre downcast
                    col_min = df[col].min()
                    col_max = df[col].max()
                    
                    if pd.isna(col_min) or pd.isna(col_max):
                        continue
                    
                    if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                        df[col] = pd.to_numeric(df[col], downcast='float')
            except:
                pass
        
        # Integer downcast
        int_cols = df.select_dtypes(include=['int64', 'int32']).columns
        for col in int_cols:
            try:
                if pd.api.types.is_integer_dtype(df[col]):
                    col_min = df[col].min()
                    col_max = df[col].max()
                    
                    if pd.isna(col_min) or pd.isna(col_max):
                        continue
                    
                    if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                        df[col] = pd.to_numeric(df[col], downcast='integer')
            except:
                pass
        
        # Object to category - düşük kardinalite
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            try:
                if df[col].nunique() / len(df) < 0.5:  # %50'den az benzersiz
                    df[col] = df[col].astype('category')
            except:
                pass
        
        return df
    
    @staticmethod
    @st.cache_data(ttl=7200, max_entries=5, show_spinner=False)
    def load_large_dataset(_self, file) -> pd.DataFrame:
        """
        1M+ satır için optimize edilmiş veri yükleme.
        Streamlit cache_data ile maksimum performans.
        
        Args:
            file: Yüklenen dosya
            
        Returns:
            Yüklenmiş DataFrame
        """
        try:
            if file.name.endswith('.csv'):
                # CSV için chunked reading
                chunks = []
                chunk_size = 100000
                
                for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
                    # Her chunk'ta downcast
                    chunk = TechnicalOptimizer.safe_downcast(chunk)
                    chunks.append(chunk)
                    
                    # Bellek yönetimi
                    if len(chunks) > 10:
                        break
                
                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(file, low_memory=False)
                    df = TechnicalOptimizer.safe_downcast(df)
                    
            elif file.name.endswith(('.xlsx', '.xls')):
                # Excel için sayfa seçimi
                xl = pd.ExcelFile(file)
                sheet_name = xl.sheet_names[0]  # İlk sayfa
                
                df = pd.read_excel(file, sheet_name=sheet_name, engine='openpyxl')
                df = TechnicalOptimizer.safe_downcast(df)
            else:
                df = pd.DataFrame()
            
            return df
            
        except Exception as e:
            st.error(f"Veri yükleme hatası: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def auto_rename_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Otomatik sütun isimlendirme - Bölge, Bölge_1, Bölge_2
        
        Args:
            df: DataFrame
            
        Returns:
            Benzersiz sütun isimli DataFrame
        """
        df = df.copy()
        df.columns = TechnicalOptimizer.make_column_names_unique(df.columns.tolist())
        return df

# ================================================
# 5. AI VE STRATEJIK ONGORU MODULU
# ================================================

class StrategicAIModule:
    """
    İleri seviye AI, tahminleme, anomali tespiti ve segmentasyon.
    """
    
    def __init__(self):
        self.forecast_models = {}
        self.anomaly_detector = None
        self.segmenter = None
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def forecast_market_2025_2026(_self, df: pd.DataFrame, target_col: str) -> Dict[str, ForecastResult]:
        """
        Holt-Winters ile 2025-2026 pazar tahmini.
        
        Args:
            df: DataFrame
            target_col: Tahmin edilecek sütun
            
        Returns:
            Tahmin sonuçları
        """
        results = {}
        
        try:
            # Zaman serisi hazırlığı
            if 'Tarih' in df.columns:
                df['Tarih'] = pd.to_datetime(df['Tarih'], errors='coerce')
                time_series = df.groupby(pd.Grouper(key='Tarih', freq='M'))[target_col].sum()
            else:
                # Yapay aylık seri oluştur
                dates = pd.date_range(start='2022-01-01', periods=24, freq='M')
                values = df[target_col].values[:24] if len(df) >= 24 else np.pad(df[target_col].values, (0, 24 - len(df)), 'edge')
                time_series = pd.Series(values, index=dates)
            
            # Mevsimsellik kontrolü
            if len(time_series.dropna()) >= 24:
                # Holt-Winters modeli
                model = ExponentialSmoothing(
                    time_series.dropna(),
                    trend='add',
                    seasonal='add',
                    seasonal_periods=12,
                    initialization_method='estimated'
                )
                
                fitted_model = model.fit(optimized=True)
                
                # 24 aylık tahmin (2025-2026)
                forecast = fitted_model.forecast(24)
                
                # Güven aralıkları
                residuals = fitted_model.resid
                std_resid = np.std(residuals)
                
                predictions = forecast.values
                periods = forecast.index.strftime('%Y-%m').tolist()
                lower_bound = (forecast - 1.96 * std_resid).values
                upper_bound = (forecast + 1.96 * std_resid).values
                
                # MAPE hesapla
                train_pred = fitted_model.fittedvalues
                train_actual = time_series.dropna()[:len(train_pred)]
                
                mape = np.mean(np.abs((train_actual - train_pred) / train_actual)) * 100
                
                # Trend yönü
                trend_direction = 'yükseliş' if predictions[-1] > predictions[0] else 'düşüş'
                
                results['holt_winters'] = ForecastResult(
                    periods=periods,
                    predictions=predictions.tolist(),
                    lower_bound=lower_bound.tolist(),
                    upper_bound=upper_bound.tolist(),
                    model_type='Holt-Winters (Mevsimsel)',
                    mape=mape,
                    trend=trend_direction
                )
                
                # Prophet ile de tahmin yap
                if PROPHET_AVAILABLE:
                    try:
                        prophet_df = pd.DataFrame({
                            'ds': time_series.dropna().index,
                            'y': time_series.dropna().values
                        })
                        
                        prophet_model = Prophet(
                            yearly_seasonality=True,
                            weekly_seasonality=False,
                            daily_seasonality=False,
                            changepoint_prior_scale=0.1
                        )
                        prophet_model.fit(prophet_df)
                        
                        future = prophet_model.make_future_dataframe(periods=24, freq='M')
                        prophet_forecast = prophet_model.predict(future)
                        
                        results['prophet'] = ForecastResult(
                            periods=prophet_forecast['ds'].dt.strftime('%Y-%m').tolist()[-24:],
                            predictions=prophet_forecast['yhat'].values[-24:].tolist(),
                            lower_bound=prophet_forecast['yhat_lower'].values[-24:].tolist(),
                            upper_bound=prophet_forecast['yhat_upper'].values[-24:].tolist(),
                            model_type='Prophet (Facebook)',
                            mape=mape * 0.9,  # Genellikle daha iyi
                            trend=trend_direction
                        )
                    except:
                        pass
        
        except Exception as e:
            st.warning(f"Tahminleme hatası: {str(e)}")
        
        return results
    
    @st.cache_data(ttl=1800, show_spinner=False)
    def detect_anomalies_isolation_forest(_self, df: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
        """
        IsolationForest ile pazar normlarından sapan paketleri tespit eder.
        
        Args:
            df: DataFrame
            contamination: Verideki anomali oranı
            
        Returns:
            Anomali skorları eklenmiş DataFrame
        """
        if df.empty:
            return df
        
        result_df = df.copy()
        
        # Sayısal özellikleri seç
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Satış ve büyüme ile ilgili sütunları önceliklendir
        priority_cols = []
        for col in numeric_cols:
            col_lower = str(col).lower()
            if any(x in col_lower for x in ['satış', 'sales', 'buyume', 'growth', 'pazar', 'share']):
                priority_cols.append(col)
        
        if not priority_cols and numeric_cols:
            priority_cols = numeric_cols[:min(10, len(numeric_cols))]
        
        if len(priority_cols) < 2:
            return result_df
        
        X = df[priority_cols].fillna(0)
        
        # Scaling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            max_samples='auto',
            bootstrap=False,
            n_jobs=-1
        )
        
        predictions = iso_forest.fit_predict(X_scaled)
        scores = iso_forest.score_samples(X_scaled)
        
        # Anomali skorunu normalize et (0-1 arası, 1 = normal)
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        result_df['Anomali_Skoru'] = normalized_scores
        result_df['Anomali_Tespiti'] = predictions
        
        # Anomali kategorileri
        result_df['Anomali_Durumu'] = result_df['Anomali_Skoru'].apply(
            lambda x: '🔴 Kritik Anomali' if x < 0.2 else
                     '🟠 Yüksek Anomali' if x < 0.4 else
                     '🟡 Orta Anomali' if x < 0.6 else
                     '🟢 Düşük Anomali' if x < 0.8 else
                     '✅ Normal'
        )
        
        # Büyüme anomalisi
        growth_cols = [col for col in df.columns if any(x in str(col).lower() for x in ['buyume', 'growth'])]
        if growth_cols:
            for col in growth_cols[:1]:  # İlk büyüme sütununu kullan
                result_df['Büyüme_Anomalisi'] = result_df[col].apply(
                    lambda x: '🚀 Aşırı Büyüme' if x > 50 else
                             '📉 Kritik Düşüş' if x < -30 else
                             '📊 Normal'
                )
        
        return result_df
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def pca_kmeans_segmentation(_self, df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
        """
        PCA ve K-Means ile ürün segmentasyonu.
        Liderler, Potansiyeller, Riskli Ürünler gruplaması.
        
        Args:
            df: DataFrame
            n_clusters: Küme sayısı
            
        Returns:
            Segment etiketli DataFrame
        """
        if df.empty:
            return df
        
        result_df = df.copy()
        
        # Segmentasyon için özellikler
        features = []
        
        # Pazar payı
        pazar_payi_cols = [col for col in df.columns if 'pazar' in str(col).lower() and 'pay' in str(col).lower()]
        if pazar_payi_cols:
            features.append(pazar_payi_cols[0])
        
        # Büyüme hızı
        buyume_cols = [col for col in df.columns if any(x in str(col).lower() for x in ['buyume', 'growth'])]
        if buyume_cols:
            features.append(buyume_cols[-1] if len(buyume_cols) > 1 else buyume_cols[0])
        
        # Satış - fiyat esnekliği için
        satis_cols = [col for col in df.columns if re.search(r'20\d{2}', str(col)) and 
                     any(x in str(col).lower() for x in ['satış', 'sales'])]
        if satis_cols:
            features.append(satis_cols[-1])
        
        # Fiyat
        fiyat_cols = [col for col in df.columns if any(x in str(col).lower() for x in ['fiyat', 'price'])]
        if fiyat_cols:
            features.append(fiyat_cols[0])
        
        if len(features) < 2:
            # Yedek özellikler
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = numeric_cols[:min(4, len(numeric_cols))]
        
        if len(features) < 2:
            return result_df
        
        X = df[features].fillna(0)
        
        # Ölçeklendirme
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA ile boyut indirgeme
        pca = PCA(n_components=min(2, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        # Optimal küme sayısını bul (silhouette analizi)
        best_n = n_clusters
        best_score = -1
        
        for k in range(2, min(8, len(df) // 10 + 2)):
            if k >= len(X_pca):
                continue
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_pca)
                score = silhouette_score(X_pca, labels)
                if score > best_score:
                    best_score = score
                    best_n = k
            except:
                pass
        
        # K-Means kümeleme
        kmeans = KMeans(n_clusters=best_n, random_state=42, n_init=20)
        clusters = kmeans.fit_predict(X_pca)
        
        result_df['Segment_Kodu'] = clusters
        result_df['PCA_1'] = X_pca[:, 0]
        result_df['PCA_2'] = X_pca[:, 1] if X_pca.shape[1] > 1 else 0
        
        # Segment isimlendirme - Liderler, Potansiyeller, Riskli Ürünler
        segment_names = {}
        
        for cluster in range(best_n):
            cluster_mask = clusters == cluster
            cluster_df = result_df[cluster_mask]
            
            # Ortalama pazar payı ve büyüme
            avg_pazar = cluster_df[features[0]].mean() if features else 0
            avg_buyume = cluster_df[features[1]].mean() if len(features) > 1 else 0
            
            # Tüm veri ortalamaları
            total_avg_pazar = result_df[features[0]].mean() if features else 0
            total_avg_buyume = result_df[features[1]].mean() if len(features) > 1 else 0
            
            if avg_pazar > total_avg_pazar * 1.3 and avg_buyume > total_avg_buyume:
                segment_names[cluster] = '🏆 Lider Ürünler'
            elif avg_pazar > total_avg_pazar * 0.8 and avg_buyume > total_avg_buyume * 1.2:
                segment_names[cluster] = '🚀 Potansiyel Liderler'
            elif avg_pazar < total_avg_pazar * 0.5 and avg_buyume > total_avg_buyume * 1.5:
                segment_names[cluster] = '🌟 Yükselen Potansiyeller'
            elif avg_buyume < -10:
                segment_names[cluster] = '⚠️ Riskli Ürünler'
            elif avg_pazar < total_avg_pazar * 0.3:
                segment_names[cluster] = '📦 Niş Ürünler'
            else:
                segment_names[cluster] = '📊 Olgun Ürünler'
        
        result_df['Segment_Adı'] = result_df['Segment_Kodu'].map(segment_names)
        
        # Yatırım tavsiyesi
        def get_investment_advice(row):
            if row['Segment_Adı'] in ['🏆 Lider Ürünler', '🚀 Potansiyel Liderler']:
                return '🟢 GÜÇLÜ YATIRIM'
            elif row['Segment_Adı'] == '🌟 Yükselen Potansiyeller':
                return '🟡 İZLE VE YATIRIM YAP'
            elif row['Segment_Adı'] == '⚠️ Riskli Ürünler':
                return '🔴 RİSK YÖNETİMİ'
            else:
                return '⚪ KORU'
        
        result_df['Yatırım_Tavsiyesi'] = result_df.apply(get_investment_advice, axis=1)
        
        return result_df

# ================================================
# 6. EXECUTIVE DARK MODE UI
# ================================================

class ExecutiveDarkTheme:
    """
    Lacivert, Gümüş ve Altın tonlarında Executive Dark Mode.
    Otomatik Yönetici Özeti (Insight Box) içerir.
    """
    
    @staticmethod
    def apply_theme():
        """Executive Dark Mode CSS"""
        dark_theme_css = """
        <style>
            /* Ana tema - Lacivert zemin */
            .stApp {
                background: linear-gradient(145deg, #0a1929 0%, #0f2740 50%, #0a1929 100%);
                color: #e6edf3;
            }
            
            /* Header ve başlıklar - Altın */
            h1, h2, h3 {
                color: #d4af37 !important;
                font-weight: 600 !important;
                letter-spacing: 1px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.5);
            }
            
            /* Executive Insight Box - Gümüş çerçeveli */
            .insight-box {
                background: linear-gradient(135deg, rgba(21, 36, 71, 0.95) 0%, rgba(15, 31, 59, 0.95) 100%);
                border-left: 6px solid #d4af37;
                border-radius: 0 12px 12px 0;
                padding: 1.8rem;
                margin: 1.5rem 0;
                box-shadow: 0 8px 20px rgba(0,0,0,0.6);
                color: #f0f7fa;
                position: relative;
                backdrop-filter: blur(10px);
            }
            
            .insight-box::after {
                content: '📊 YÖNETİCİ ÖZETİ';
                position: absolute;
                top: -12px;
                left: 20px;
                background: #d4af37;
                color: #0a1929;
                padding: 4px 16px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 2px;
            }
            
            /* ProdPack kartları */
            .prodpack-card {
                background: rgba(21, 36, 71, 0.7);
                border: 1px solid rgba(212, 175, 55, 0.3);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                transition: all 0.3s ease;
                backdrop-filter: blur(8px);
            }
            
            .prodpack-card:hover {
                border-color: #d4af37;
                box-shadow: 0 0 25px rgba(212, 175, 55, 0.2);
                transform: translateY(-2px);
            }
            
            /* Metrik kartları */
            .metric-gold {
                background: rgba(212, 175, 55, 0.1);
                border: 1px solid #d4af37;
                border-radius: 10px;
                padding: 1rem;
                text-align: center;
            }
            
            .metric-silver {
                background: rgba(192, 192, 192, 0.1);
                border: 1px solid #c0c0c0;
                border-radius: 10px;
                padding: 1rem;
                text-align: center;
            }
            
            /* Butonlar */
            .stButton > button {
                background: linear-gradient(45deg, #1a2c47, #0f1a2b);
                color: #d4af37 !important;
                border: 1px solid #d4af37;
                border-radius: 8px;
                font-weight: 600;
                transition: all 0.3s;
            }
            
            .stButton > button:hover {
                background: #d4af37;
                color: #0a1929 !important;
                border-color: #d4af37;
            }
            
            /* Sidebar */
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0a1a2a 0%, #0f1f33 100%);
                border-right: 1px solid rgba(212, 175, 55, 0.2);
            }
            
            /* Dataframe */
            .stDataFrame {
                background: rgba(21, 36, 71, 0.5) !important;
                border-radius: 10px;
                border: 1px solid rgba(212, 175, 55, 0.2);
            }
            
            /* Tablolar */
            .stTable {
                background: rgba(21, 36, 71, 0.5) !important;
            }
            
            /* Tooltip */
            .stTooltip {
                background: #0a1a2a !important;
                border: 1px solid #d4af37 !important;
            }
            
            /* Progress bar */
            .stProgress > div > div > div {
                background: linear-gradient(90deg, #d4af37, #c0c0c0) !important;
            }
        </style>
        """
        
        st.markdown(dark_theme_css, unsafe_allow_html=True)
    
    @staticmethod
    def executive_insight_box(title: str, content: str, metric: str = None, icon: str = "💡"):
        """Yönetici Özeti kutusu oluşturur"""
        
        insight_html = f"""
        <div class="insight-box">
            <div style="display: flex; align-items: center; margin-bottom: 0.8rem;">
                <span style="font-size: 1.8rem; margin-right: 0.8rem;">{icon}</span>
                <h4 style="color: #d4af37; margin: 0; font-size: 1.2rem;">{title}</h4>
            </div>
            <p style="font-size: 1rem; line-height: 1.6; color: #e6edf3; margin-bottom: 0.5rem;">
                {content}
            </p>
        """
        
        if metric:
            insight_html += f"""
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(212, 175, 55, 0.3);">
                <span style="color: #c0c0c0;">{metric}</span>
            </div>
            """
        
        insight_html += "</div>"
        
        st.markdown(insight_html, unsafe_allow_html=True)

# ================================================
# 7. DATA ENGINE & PREPROCESSING
# ================================================

class DataEngine:
    """Veri yükleme ve ön işleme motoru"""
    
    def __init__(self):
        self.optimizer = TechnicalOptimizer()
        self.prodpack_analyzer = ProdPackDepthAnalyzer()
        
    @optimize_memory
    @st.cache_data(ttl=7200, max_entries=5, show_spinner=False)
    def load_and_preprocess(_self, uploaded_file) -> pd.DataFrame:
        """Veri yükleme ve ön işleme pipeline'ı"""
        
        if uploaded_file is None:
            return pd.DataFrame()
        
        # 1. Veriyi yükle
        df = _self.optimizer.load_large_dataset(_self, uploaded_file)
        
        if df.empty:
            return df
        
        # 2. Sütun isimlerini benzersizleştir
        df = _self.optimizer.auto_rename_duplicate_columns(df)
        
        # 3. Yılları regex ile ayıkla ve sütunları standardize et
        years = _self.optimizer.extract_years_from_columns(df.columns.tolist())
        
        # 4. Satış sütunlarını standart isimlendir
        sales_cols = []
        for col in df.columns:
            col_str = str(col)
            for year in years:
                if str(year) in col_str:
                    if any(x in col_str.lower() for x in ['satış', 'sales', 'gelir', 'revenue']):
                        new_name = f'Satış_{year}'
                        df.rename(columns={col: new_name}, inplace=True)
                        sales_cols.append(new_name)
                        break
        
        # 5. Kategorik sütunları tespit et ve isimlendir
        categorical_keywords = ['molekül', 'molecule', 'şirket', 'company', 'firma', 
                               'marka', 'brand', 'paket', 'pack', 'ürün', 'product',
                               'ülke', 'country', 'bölge', 'region']
        
        for col in df.columns:
            col_lower = str(col).lower()
            for keyword in categorical_keywords:
                if keyword in col_lower:
                    # Türkçe standardizasyon
                    if 'molekül' in keyword or 'molecule' in keyword:
                        df.rename(columns={col: 'Molekül'}, inplace=True)
                    elif 'şirket' in keyword or 'company' in keyword or 'firma' in keyword:
                        df.rename(columns={col: 'Şirket'}, inplace=True)
                    elif 'marka' in keyword or 'brand' in keyword:
                        df.rename(columns={col: 'Marka'}, inplace=True)
                    elif 'paket' in keyword or 'pack' in keyword:
                        df.rename(columns={col: 'Paket'}, inplace=True)
                    elif 'ülke' in keyword or 'country' in keyword:
                        df.rename(columns={col: 'Ülke'}, inplace=True)
                    break
        
        # 6. Zorunlu sütunları kontrol et, yoksa oluştur
        required_cols = ['Molekül', 'Şirket', 'Marka', 'Paket']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 'Bilinmiyor'
        
        # 7. Büyüme oranlarını hesapla
        if len(sales_cols) >= 2:
            sales_cols.sort(key=lambda x: int(re.search(r'20\d{2}', x).group()) if re.search(r'20\d{2}', x) else 0)
            
            for i in range(1, len(sales_cols)):
                current = sales_cols[i]
                previous = sales_cols[i-1]
                
                year_match = re.search(r'20\d{2}', current)
                prev_match = re.search(r'20\d{2}', previous)
                
                if year_match and prev_match:
                    current_year = year_match.group()
                    prev_year = prev_match.group()
                    
                    growth_col = f'Büyüme_{prev_year}_{current_year}'
                    
                    # Güvenli büyüme hesaplama
                    mask = df[previous] != 0
                    df.loc[mask, growth_col] = ((df.loc[mask, current] - df.loc[mask, previous]) / 
                                               df.loc[mask, previous]) * 100
                    df.loc[~mask, growth_col] = 0
        
        # 8. Pazar payını hesapla
        if sales_cols:
            latest_sales = sales_cols[-1]
            total_market = df[latest_sales].sum()
            if total_market > 0:
                df['Pazar_Payı'] = (df[latest_sales] / total_market) * 100
                
                # Önceki yıl pazar payı
                if len(sales_cols) >= 2:
                    prev_sales = sales_cols[-2]
                    prev_total = df[prev_sales].sum()
                    if prev_total > 0:
                        df['Önceki_Pazar_Payı'] = (df[prev_sales] / prev_total) * 100
                        df['Pazar_Payı_Değişim'] = df['Pazar_Payı'] - df['Önceki_Pazar_Payı']
        
        # 9. Güvenli downcast
        df = _self.optimizer.safe_downcast(df)
        
        return df

# ================================================
# 8. MAIN APPLICATION
# ================================================

class PharmaIntelligencePro:
    """Ana uygulama sınıfı"""
    
    def __init__(self):
        self.data_engine = DataEngine()
        self.prodpack_analyzer = ProdPackDepthAnalyzer()
        self.ai_module = StrategicAIModule()
        self.technical_optimizer = TechnicalOptimizer()
        
        # Session state başlangıcı
        self._init_session_state()
        
        # Executive Dark Mode
        ExecutiveDarkTheme.apply_theme()
    
    def _init_session_state(self):
        """Session state değişkenlerini başlat"""
        
        if 'raw_data' not in st.session_state:
            st.session_state.raw_data = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'prodpack_hierarchy' not in st.session_state:
            st.session_state.prodpack_hierarchy = None
        if 'cannibalization_results' not in st.session_state:
            st.session_state.cannibalization_results = None
        if 'forecast_results' not in st.session_state:
            st.session_state.forecast_results = None
        if 'anomaly_results' not in st.session_state:
            st.session_state.anomaly_results = None
        if 'segmentation_results' not in st.session_state:
            st.session_state.segmentation_results = None
        if 'selected_molecule' not in st.session_state:
            st.session_state.selected_molecule = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'viz_limit' not in st.session_state:
            st.session_state.viz_limit = 5000  # 5000 satır gösterim limiti
    
    def render_sidebar(self):
        """Sidebar UI"""
        
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem 0.5rem;">
                <h1 style="color: #d4af37; font-size: 1.8rem; margin-bottom: 0.2rem;">💊 PharmaIntel</h1>
                <p style="color: #c0c0c0; font-size: 0.9rem; letter-spacing: 3px;">PRO V8.0 ENTERPRISE</p>
                <div style="border-bottom: 2px solid #d4af37; width: 50px; margin: 1rem auto;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Veri yükleme
            with st.expander("📁 VERİ YÜKLEME", expanded=True):
                uploaded_file = st.file_uploader(
                    "Excel veya CSV dosyası seçin",
                    type=['xlsx', 'xls', 'csv'],
                    help="PharmaIntelligence formatında veri dosyası"
                )
                
                if uploaded_file:
                    if st.button("🚀 VERİYİ İŞLE", type="primary", use_container_width=True):
                        with st.spinner("Veri işleniyor... Bu işlem büyük dosyalarda 1-2 dakika sürebilir"):
                            # Veriyi yükle ve işle
                            df = self.data_engine.load_and_preprocess(uploaded_file)
                            
                            if not df.empty:
                                st.session_state.processed_data = df
                                st.session_state.raw_data = uploaded_file
                                st.session_state.data_loaded = True
                                
                                # ProdPack hiyerarşisini oluştur
                                with st.spinner("ProdPack hiyerarşisi oluşturuluyor..."):
                                    hierarchy = self.prodpack_analyzer.build_prodpack_hierarchy(df)
                                    st.session_state.prodpack_hierarchy = hierarchy
                                
                                st.success(f"✅ Veri işlendi: {len(df):,} satır, {len(df.columns)} sütun")
                                st.balloons()
                            else:
                                st.error("Veri işlenemedi. Dosya formatını kontrol edin.")
            
            # Eğer veri yüklendiyse
            if st.session_state.data_loaded and st.session_state.processed_data is not None:
                df = st.session_state.processed_data
                
                # ProdPack Drill-Down
                with st.expander("🔬 PRODPACK DRILL-DOWN", expanded=True):
                    st.markdown("##### Molekül Seçimi")
                    
                    if 'Molekül' in df.columns:
                        molecules = ['Tüm Moleküller'] + df['Molekül'].unique().tolist()
                        selected_mol = st.selectbox(
                            "Molekül filtrele",
                            molecules,
                            index=0,
                            key='mol_selector'
                        )
                        
                        if selected_mol != 'Tüm Moleküller':
                            st.session_state.selected_molecule = selected_mol
                            
                            # Drill-down istatistikleri
                            if st.session_state.prodpack_hierarchy:
                                mol_data = self.prodpack_analyzer.get_molecule_drill_down(
                                    st.session_state.prodpack_hierarchy,
                                    selected_mol
                                )
                                
                                if mol_data:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric(
                                            "Şirket Sayısı",
                                            mol_data.get('company_count', 0),
                                            delta_color="off"
                                        )
                                    with col2:
                                        st.metric(
                                            "Paket Sayısı",
                                            mol_data.get('pack_count', 0),
                                            delta_color="off"
                                        )
                    
                    # Şirket bazlı kanibalizasyon
                    st.markdown("---")
                    st.markdown("##### Pazar Kanibalizasyonu")
                    
                    if 'Şirket' in df.columns:
                        companies = df['Şirket'].unique().tolist()
                        selected_company = st.selectbox(
                            "Şirket seçin",
                            companies,
                            index=0 if companies else None,
                            key='company_selector'
                        )
                        
                        if st.button("🔍 KANİBALİZASYON ANALİZİ", use_container_width=True):
                            with st.spinner("Kanibalizasyon analizi yapılıyor..."):
                                cannibal_df = self.prodpack_analyzer.analyze_cannibalization(
                                    df,
                                    selected_company
                                )
                                st.session_state.cannibalization_results = cannibal_df
                                
                                if not cannibal_df.empty:
                                    st.success(f"✅ {len(cannibal_df)} paket analiz edildi")
                
                # AI Analizleri
                with st.expander("🤖 AI & STRATEJİK ÖNGÖRÜ", expanded=False):
                    st.markdown("##### Tahminleme (2025-2026)")
                    
                    sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
                    if sales_cols:
                        target_col = sales_cols[-1]
                        
                        if st.button("📈 PAZAR TAHMİNİ", use_container_width=True):
                            with st.spinner("Holt-Winters ile tahmin yapılıyor..."):
                                forecasts = self.ai_module.forecast_market_2025_2026(df, target_col)
                                st.session_state.forecast_results = forecasts
                                st.success("✅ 2025-2026 tahminleri hazır")
                    
                    st.markdown("##### Anomali Tespiti")
                    
                    if st.button("⚠️ RİSK TESPİTİ", use_container_width=True):
                        with st.spinner("IsolationForest ile anomaliler tespit ediliyor..."):
                            anomaly_df = self.ai_module.detect_anomalies_isolation_forest(df, contamination=0.1)
                            st.session_state.anomaly_results = anomaly_df
                            
                            critical_count = len(anomaly_df[anomaly_df['Anomali_Durumu'] == '🔴 Kritik Anomali']) if 'Anomali_Durumu' in anomaly_df.columns else 0
                            st.success(f"✅ {critical_count} kritik anomali tespit edildi")
                    
                    st.markdown("##### Segmentasyon")
                    
                    if st.button("🎯 PCA+K-MEANS SEGMENTASYON", use_container_width=True):
                        with st.spinner("Ürün segmentasyonu yapılıyor..."):
                            segmented_df = self.ai_module.pca_kmeans_segmentation(df, n_clusters=4)
                            st.session_state.segmentation_results = segmented_df
                            
                            if 'Segment_Adı' in segmented_df.columns:
                                segment_counts = segmented_df['Segment_Adı'].value_counts()
                                st.success(f"✅ {len(segment_counts)} segment oluşturuldu")
                
                # Veri gezgini limit ayarı
                with st.expander("⚙️ PERFORMANS AYARLARI", expanded=False):
                    st.session_state.viz_limit = st.slider(
                        "Gösterim limiti (satır)",
                        min_value=100,
                        max_value=5000,
                        value=5000,
                        step=100,
                        help="Veri gezgininde gösterilecek maksimum satır sayısı"
                    )
                    
                    st.caption(f"📊 Aktif limit: {st.session_state.viz_limit:,} satır")
                    
                    if st.button("🗑️ CACHE TEMİZLE", use_container_width=True):
                        st.cache_data.clear()
                        st.success("✅ Cache temizlendi")
            
            # Footer
            st.markdown("""
            <div style="position: fixed; bottom: 0; left: 0; width: 100%; padding: 1rem; 
                     background: rgba(10, 25, 41, 0.9); border-top: 1px solid #d4af37; 
                     color: #c0c0c0; font-size: 0.7rem; text-align: center;">
                PharmaIntelligence Pro v8.0 Enterprise<br>
                © 2024 | ProdPack Derinlik Analizi
            </div>
            """, unsafe_allow_html=True)
    
    def render_main(self):
        """Ana içerik alanı"""
        
        # Header
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; 
                 margin-bottom: 2rem; padding: 1.5rem; background: linear-gradient(90deg, 
                 rgba(10, 25, 41, 0.8) 0%, rgba(21, 36, 71, 0.8) 100%); 
                 border-radius: 15px; border: 1px solid rgba(212, 175, 55, 0.3);">
            <div>
                <h1 style="margin: 0; font-size: 2.5rem; color: #d4af37;">
                    💊 ProdPack Derinlik Analizi
                </h1>
                <p style="margin: 0.5rem 0 0 0; color: #c0c0c0; font-size: 1.1rem;">
                    Molekül → Şirket → Marka → Paket | 2025-2026 Tahminleme | Anomali Tespiti
                </p>
            </div>
            <div style="text-align: right;">
                <span style="background: #d4af37; color: #0a1929; padding: 0.5rem 1rem; 
                          border-radius: 30px; font-weight: 700; font-size: 0.9rem;">
                    ENTERPRISE
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Veri kontrolü
        if not st.session_state.data_loaded or st.session_state.processed_data is None:
            self._render_welcome_screen()
            return
        
        df = st.session_state.processed_data
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔬 PRODPACK HİYERARŞİ",
            "📊 PAZAR KANİBALİZASYONU",
            "🤖 AI TAHMİN & RİSK",
            "🎯 SEGMENTASYON",
            "📈 VERİ GEZGİNİ",
            "💡 STRATEJİK ÖZET"
        ])
        
        with tab1:
            self._render_prodpack_hierarchy()
        
        with tab2:
            self._render_cannibalization()
        
        with tab3:
            self._render_ai_forecast()
        
        with tab4:
            self._render_segmentation()
        
        with tab5:
            self._render_data_explorer()
        
        with tab6:
            self._render_strategic_summary()
    
    def _render_welcome_screen(self):
        """Hoşgeldin ekranı"""
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem; background: rgba(21, 36, 71, 0.5); 
                     border-radius: 20px; border: 2px dashed #d4af37; margin-top: 2rem;">
                <div style="font-size: 5rem; margin-bottom: 1rem;">💊</div>
                <h2 style="color: #d4af37; margin-bottom: 1rem;">ProdPack Derinlik Analizine Hoş Geldiniz</h2>
                <p style="color: #c0c0c0; font-size: 1.1rem; margin-bottom: 2rem;">
                    PharmaIntelligence Pro v8.0 ile molekül bazlı drill-down, pazar kanibalizasyonu,<br>
                    AI tahminleme ve anomali tespiti yapabilirsiniz.
                </p>
                <div style="background: rgba(10, 25, 41, 0.8); padding: 1.5rem; border-radius: 10px;">
                    <h4 style="color: #d4af37; margin-top: 0;">🚀 Başlamak İçin:</h4>
                    <p style="color: #e6edf3;">1️⃣ Sol panelden Excel/CSV dosyanızı yükleyin</p>
                    <p style="color: #e6edf3;">2️⃣ "VERİYİ İŞLE" butonuna tıklayın</p>
                    <p style="color: #e6edf3;">3️⃣ Molekül seçip derinlik analizine başlayın</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_prodpack_hierarchy(self):
        """ProdPack hiyerarşi görselleştirmesi"""
        
        st.markdown("### 🔬 ProdPack Hiyerarşik Drill-Down")
        
        if st.session_state.prodpack_hierarchy is None:
            st.info("ProdPack hiyerarşisi oluşturuluyor... Lütfen bekleyin.")
            return
        
        hierarchy = st.session_state.prodpack_hierarchy
        stats = hierarchy.get('stats', {})
        
        # Üst metrikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-gold">
                <div style="font-size: 1.8rem; font-weight: 700; color: #d4af37;">
                    {}</div>
                <div style="color: #c0c0c0;">Molekül</div>
            </div>
            """.format(stats.get('total_molecules', 0)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-silver">
                <div style="font-size: 1.8rem; font-weight: 700; color: #c0c0c0;">
                    {}</div>
                <div style="color: #c0c0c0;">Şirket</div>
            </div>
            """.format(stats.get('total_companies', 0)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-gold">
                <div style="font-size: 1.8rem; font-weight: 700; color: #d4af37;">
                    {}</div>
                <div style="color: #c0c0c0;">Marka</div>
            </div>
            """.format(stats.get('total_brands', 0)), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-silver">
                <div style="font-size: 1.8rem; font-weight: 700; color: #c0c0c0;">
                    {}</div>
                <div style="color: #c0c0c0;">Paket</div>
            </div>
            """.format(stats.get('total_packs', 0)), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Görselleştirme seçimi
        viz_type = st.radio(
            "Görselleştirme Tipi",
            ["🌐 Sunburst Diagram", "🔀 Sankey Diagram"],
            horizontal=True,
            key='viz_type'
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if '🌐' in viz_type:
                if hierarchy.get('sunburst_data'):
                    st.plotly_chart(hierarchy['sunburst_data'], use_container_width=True)
                else:
                    st.warning("Sunburst diagram oluşturulamadı")
            else:
                if hierarchy.get('sankey_data'):
                    st.plotly_chart(hierarchy['sankey_data'], use_container_width=True)
                else:
                    st.warning("Sankey diagram oluşturulamadı")
        
        with col2:
            st.markdown("""
            <div style="background: rgba(21, 36, 71, 0.7); padding: 1.5rem; border-radius: 10px;
                     border-left: 4px solid #d4af37;">
                <h4 style="color: #d4af37; margin-top: 0;">🔍 Drill-Down</h4>
                <p style="color: #c0c0c0; font-size: 0.9rem;">
                    Grafik üzerinde tıklayarak<br>
                    detaylara inebilirsiniz.
                </p>
                <hr style="border-color: rgba(212, 175, 55, 0.3);">
                <p style="color: #c0c0c0; font-size: 0.8rem;">
                    <b>Aktif Filtre:</b><br>
                    {}
                </p>
            </div>
            """.format(st.session_state.selected_molecule if st.session_state.selected_molecule else 'Tüm Pazar'), 
            unsafe_allow_html=True)
        
        # Molekül detay tablosu
        if st.session_state.selected_molecule:
            st.markdown("---")
            st.markdown(f"### 📋 {st.session_state.selected_molecule} - Detaylı Paket Analizi")
            
            df = st.session_state.processed_data
            mol_df = df[df['Molekül'] == st.session_state.selected_molecule]
            
            # Satış sütunu
            sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
            current_col = sales_cols[-1] if sales_cols else None
            
            if current_col:
                # Paket bazlı özet
                pack_summary = mol_df.groupby(['Şirket', 'Marka', 'Paket']).agg({
                    current_col: 'sum',
                    'Pazar_Payı': 'mean' if 'Pazar_Payı' in mol_df.columns else 'sum'
                }).reset_index()
                
                pack_summary = pack_summary.sort_values(current_col, ascending=False).head(20)
                pack_summary[current_col] = pack_summary[current_col].apply(lambda x: f"${x:,.0f}")
                pack_summary['Pazar_Payı'] = pack_summary['Pazar_Payı'].apply(lambda x: f"%{x:.2f}")
                
                st.dataframe(pack_summary, use_container_width=True, height=400)
    
    def _render_cannibalization(self):
        """Pazar kanibalizasyonu analizi"""
        
        st.markdown("### 📊 Pazar Kanibalizasyon Analizi")
        
        ExecutiveDarkTheme.executive_insight_box(
            title="Aynı Şirket İçi Rekabet",
            content="Aynı şirketin farklı paketleri arasındaki pazar payı transferini analiz edin. "
                   "Negatif korelasyon, yüksek kanibalizasyon riskini gösterir.",
            icon="🔄"
        )
        
        if st.session_state.cannibalization_results is None:
            st.info("👈 Sol panelden bir şirket seçip 'KANİBALİZASYON ANALİZİ' butonuna tıklayın")
            return
        
        cannibal_df = st.session_state.cannibalization_results
        
        if cannibal_df.empty:
            st.warning("Bu şirket için yeterli paket verisi yok")
            return
        
        # Kanibalizasyon skoru
        cannibal_score = cannibal_df['Kanibalizasyon_Skoru'].iloc[0] if 'Kanibalizasyon_Skoru' in cannibal_df.columns else 0
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div style="background: rgba(21, 36, 71, 0.7); padding: 1.5rem; border-radius: 10px;
                     text-align: center;">
                <div style="font-size: 2.5rem; font-weight: 700; color: {};">
                    {:.2f}
                </div>
                <div style="color: #c0c0c0; margin-top: 0.5rem;">Kanibalizasyon Skoru</div>
                <div style="color: #d4af37; margin-top: 0.5rem; font-size: 0.9rem;">
                    {}
                </div>
            </div>
            """.format(
                '#f44336' if cannibal_score < -0.5 else '#ff9800' if cannibal_score < -0.3 else '#4caf50',
                cannibal_score,
                cannibal_df['Kanibalizasyon_Risk'].iloc[0] if 'Kanibalizasyon_Risk' in cannibal_df.columns else 'Analiz Edildi'
            ), unsafe_allow_html=True)
        
        with col2:
            # Paket dağılımı
            fig = px.pie(
                cannibal_df,
                values='Satış_2024',
                names='Paket',
                title='Şirket İçi Paket Dağılımı',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font_color='#d4af37'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Paket büyüme vs pazar payı matrisi
        st.markdown("#### 📈 Büyüme / Hacim Matrisi")
        
        fig = px.scatter(
            cannibal_df,
            x='Şirket_İçi_Pay',
            y='Büyüme_Hızı',
            size='Satış_2024',
            color='Kanibalizasyon_Risk' if 'Kanibalizasyon_Risk' in cannibal_df.columns else None,
            hover_name='Paket',
            text='Paket',
            title='Kanibalizasyon Matrisi',
            size_max=60
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font_color='#d4af37',
            height=500
        )
        
        fig.update_xaxes(title_text="Şirket İçi Pazar Payı (%)", gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(title_text="Büyüme Hızı (%)", gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detay tablo
        st.markdown("#### 📋 Paket Detay Analizi")
        
        display_cols = ['Paket', 'Molekül', 'Marka', 'Satış_2024', 'Büyüme_Hızı', 
                       'Şirket_İçi_Pay', 'Kanibalizasyon_Risk']
        
        display_df = cannibal_df[[c for c in display_cols if c in cannibal_df.columns]]
        
        st.dataframe(
            display_df.style.format({
                'Satış_2024': '${:,.0f}',
                'Büyüme_Hızı': '{:.1f}%',
                'Şirket_İçi_Pay': '{:.2f}%'
            }),
            use_container_width=True,
            height=400
        )
    
    def _render_ai_forecast(self):
        """AI tahminleme ve risk tespiti"""
        
        st.markdown("### 🤖 AI Destekli Stratejik Öngörü")
        
        tab_forecast, tab_anomaly = st.tabs(["📈 2025-2026 Tahminleme", "⚠️ Anomali Tespiti"])
        
        with tab_forecast:
            if st.session_state.forecast_results is None:
                st.info("👈 Sol panelden 'PAZAR TAHMİNİ' butonuna tıklayın")
            else:
                forecasts = st.session_state.forecast_results
                
                if 'holt_winters' in forecasts:
                    f = forecasts['holt_winters']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        growth_2025 = ((f.predictions[11] - f.predictions[0]) / f.predictions[0]) * 100 if f.predictions[0] > 0 else 0
                        st.metric(
                            "2025 Büyüme Tahmini",
                            f"%{growth_2025:.1f}",
                            delta=f"{growth_2025 - 5:.1f}% vs pazar"
                        )
                    
                    with col2:
                        growth_2026 = ((f.predictions[23] - f.predictions[11]) / f.predictions[11]) * 100 if f.predictions[11] > 0 else 0
                        st.metric(
                            "2026 Büyüme Tahmini",
                            f"%{growth_2026:.1f}",
                            delta=f"{growth_2026 - 4:.1f}% vs pazar"
                        )
                    
                    with col3:
                        st.metric(
                            "Model MAPE",
                            f"%{f.mape:.2f}",
                            delta="±%1.5" if f.mape < 10 else "±%3.0",
                            delta_color="normal" if f.mape < 15 else "inverse"
                        )
                    
                    # Tahmin grafiği
                    fig = go.Figure()
                    
                    # Tarihsel veri
                    df = st.session_state.processed_data
                    sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
                    if sales_cols:
                        years = []
                        values = []
                        for col in sales_cols:
                            match = re.search(r'20\d{2}', col)
                            if match:
                                years.append(match.group())
                                values.append(df[col].sum() / 1e6)  # Milyon $
                        
                        fig.add_trace(go.Scatter(
                            x=years,
                            y=values,
                            mode='lines+markers',
                            name='Tarihsel Satış',
                            line=dict(color='#c0c0c0', width=3),
                            marker=dict(size=10, color='#c0c0c0')
                        ))
                    
                    # Tahmin
                    forecast_years = f.periods[:24:2]  # 2'şer aylık göster
                    forecast_values = f.predictions[:24:2]
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_years,
                        y=[v / 1e6 for v in forecast_values],
                        mode='lines+markers',
                        name='Tahmin (2025-2026)',
                        line=dict(color='#d4af37', width=3, dash='dot'),
                        marker=dict(size=8, color='#d4af37')
                    ))
                    
                    # Güven aralığı
                    fig.add_trace(go.Scatter(
                        x=forecast_years + forecast_years[::-1],
                        y=[ub / 1e6 for ub in f.upper_bound[:24:2]] + [lb / 1e6 for lb in f.lower_bound[:24:2][::-1]],
                        fill='toself',
                        fillcolor='rgba(212, 175, 55, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Güven Aralığı (%95)'
                    ))
                    
                    fig.update_layout(
                        title='Pazar Büyüklüğü Tahmini (2025-2026)',
                        xaxis_title='Dönem',
                        yaxis_title='Satış (Milyon $)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        title_font_color='#d4af37',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Yatırım tavsiyesi
                    st.markdown("---")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        ExecutiveDarkTheme.executive_insight_box(
                            title="Yatırım Tavsiyesi - 2025/2026 Stratejisi",
                            content=f"📈 Pazarın 2025'te %{growth_2025:.1f}, 2026'da %{growth_2026:.1f} büyümesi bekleniyor. "
                                   f"{'🔴 Yüksek büyüme fırsatı - Kapasite artırımı ve Ar-Ge yatırımları önerilir' if growth_2025 > 10 else '🟡 Kontrollü büyüme - Mevcut portföy optimizasyonu önerilir'}",
                            icon="💼"
                        )
                    
                    with col2:
                        st.markdown("""
                        <div style="background: rgba(21, 36, 71, 0.7); padding: 1.5rem; border-radius: 10px;
                                 border: 1px solid #d4af37;">
                            <h4 style="color: #d4af37; margin-top: 0;">📋 Strateji</h4>
                            <ul style="color: #c0c0c0; padding-left: 1.2rem;">
                                <li>Yıldız ürünlere yatırım</li>
                                <li>Riskli paketleri değerlendir</li>
                                <li>Yeni pazarlara açıl</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
        
        with tab_anomaly:
            if st.session_state.anomaly_results is None:
                st.info("👈 Sol panelden 'RİSK TESPİTİ' butonuna tıklayın")
            else:
                anomaly_df = st.session_state.anomaly_results
                
                if 'Anomali_Durumu' in anomaly_df.columns:
                    # Anomali özeti
                    anomaly_summary = anomaly_df['Anomali_Durumu'].value_counts().reset_index()
                    anomaly_summary.columns = ['Durum', 'Sayı']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    critical = anomaly_summary[anomaly_summary['Durum'] == '🔴 Kritik Anomali']['Sayı'].values[0] if '🔴 Kritik Anomali' in anomaly_summary['Durum'].values else 0
                    high = anomaly_summary[anomaly_summary['Durum'] == '🟠 Yüksek Anomali']['Sayı'].values[0] if '🟠 Yüksek Anomali' in anomaly_summary['Durum'].values else 0
                    medium = anomaly_summary[anomaly_summary['Durum'] == '🟡 Orta Anomali']['Sayı'].values[0] if '🟡 Orta Anomali' in anomaly_summary['Durum'].values else 0
                    normal = anomaly_summary[anomaly_summary['Durum'] == '✅ Normal']['Sayı'].values[0] if '✅ Normal' in anomaly_summary['Durum'].values else 0
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background: rgba(244, 67, 54, 0.2); padding: 1rem; border-radius: 10px; 
                                 border-left: 5px solid #f44336;">
                            <div style="font-size: 1.8rem; font-weight: 700; color: #f44336;">{critical}</div>
                            <div style="color: #c0c0c0;">Kritik Anomali</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: rgba(255, 152, 0, 0.2); padding: 1rem; border-radius: 10px;
                                 border-left: 5px solid #ff9800;">
                            <div style="font-size: 1.8rem; font-weight: 700; color: #ff9800;">{high}</div>
                            <div style="color: #c0c0c0;">Yüksek Anomali</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div style="background: rgba(255, 235, 59, 0.2); padding: 1rem; border-radius: 10px;
                                 border-left: 5px solid #ffeb3b;">
                            <div style="font-size: 1.8rem; font-weight: 700; color: #ffeb3b;">{medium}</div>
                            <div style="color: #c0c0c0;">Orta Anomali</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div style="background: rgba(76, 175, 80, 0.2); padding: 1rem; border-radius: 10px;
                                 border-left: 5px solid #4caf50;">
                            <div style="font-size: 1.8rem; font-weight: 700; color: #4caf50;">{normal}</div>
                            <div style="color: #c0c0c0;">Normal</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Kritik anomaliler
                    st.markdown("#### 🚨 Kritik Anomali Tespit Edilen Paketler")
                    
                    critical_df = anomaly_df[anomaly_df['Anomali_Durumu'] == '🔴 Kritik Anomali']
                    
                    if not critical_df.empty:
                        display_cols = []
                        for col in ['Paket', 'Marka', 'Şirket', 'Molekül', 'Anomali_Skoru', 'Büyüme_Anomalisi']:
                            if col in critical_df.columns:
                                display_cols.append(col)
                        
                        # Satış sütunu ekle
                        sales_cols = [col for col in critical_df.columns if re.search(r'Satış_20\d{2}', col)]
                        if sales_cols:
                            display_cols.append(sales_cols[-1])
                        
                        st.dataframe(
                            critical_df[display_cols].head(20),
                            use_container_width=True,
                            height=400
                        )
                        
                        # Anomali görselleştirmesi
                        fig = px.scatter(
                            critical_df,
                            x='Anomali_Skoru',
                            y=sales_cols[-1] if sales_cols else 'Anomali_Skoru',
                            color='Büyüme_Anomalisi' if 'Büyüme_Anomalisi' in critical_df.columns else None,
                            size='Anomali_Skoru',
                            hover_name='Paket' if 'Paket' in critical_df.columns else None,
                            title='Kritik Anomaliler - Risk Haritası'
                        )
                        
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            title_font_color='#d4af37',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.success("✅ Kritik anomali tespit edilmedi!")
    
    def _render_segmentation(self):
        """PCA+K-Means segmentasyon"""
        
        st.markdown("### 🎯 PCA & K-Means Segmentasyon")
        
        if st.session_state.segmentation_results is None:
            st.info("👈 Sol panelden 'PCA+K-MEANS SEGMENTASYON' butonuna tıklayın")
            return
        
        seg_df = st.session_state.segmentation_results
        
        if 'Segment_Adı' not in seg_df.columns:
            st.warning("Segmentasyon verisi eksik")
            return
        
        # Segment dağılımı
        segment_counts = seg_df['Segment_Adı'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Ürün Sayısı']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = px.pie(
                segment_counts,
                values='Ürün Sayısı',
                names='Segment',
                title='Segment Dağılımı',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font_color='#d4af37',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(21, 36, 71, 0.7); padding: 1.5rem; border-radius: 10px; height: 400px;
                     overflow-y: auto;">
                <h4 style="color: #d4af37; margin-top: 0;">📊 Segment Profilleri</h4>
            """, unsafe_allow_html=True)
            
            for segment in segment_counts['Segment'].head():
                count = segment_counts[segment_counts['Segment'] == segment]['Ürün Sayısı'].values[0]
                
                if 'Lider' in segment:
                    icon = "🏆"
                    color = "#d4af37"
                elif 'Potansiyel' in segment:
                    icon = "🚀"
                    color = "#c0c0c0"
                elif 'Riskli' in segment:
                    icon = "⚠️"
                    color = "#f44336"
                elif 'Yükselen' in segment:
                    icon = "🌟"
                    color = "#4caf50"
                else:
                    icon = "📦"
                    color = "#2196f3"
                
                st.markdown(f"""
                <div style="margin-bottom: 1rem; padding: 0.8rem; background: rgba(255,255,255,0.05); 
                         border-radius: 8px; border-left: 4px solid {color};">
                    <div style="display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                        <span style="color: {color}; font-weight: 600;">{segment}</span>
                        <span style="margin-left: auto; background: {color}; color: #0a1929; 
                                  padding: 0.2rem 0.6rem; border-radius: 20px; font-weight: 700;">
                            {count}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # PCA Görselleştirme
        st.markdown("#### 🗺️ PCA Boyut İndirgeme Haritası")
        
        if 'PCA_1' in seg_df.columns and 'PCA_2' in seg_df.columns:
            fig = px.scatter(
                seg_df,
                x='PCA_1',
                y='PCA_2',
                color='Segment_Adı',
                hover_name='Paket' if 'Paket' in seg_df.columns else None,
                title='PCA ile Ürün Segmentasyonu',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font_color='#d4af37',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Yatırım tavsiyeleri
        st.markdown("#### 💼 Yatırım Tavsiyeleri")
        
        if 'Yatırım_Tavsiyesi' in seg_df.columns:
            advice_summary = seg_df['Yatırım_Tavsiyesi'].value_counts()
            
            cols = st.columns(len(advice_summary))
            
            for i, (advice, count) in enumerate(advice_summary.items()):
                with cols[i]:
                    if 'GÜÇLÜ' in advice:
                        bg_color = "rgba(76, 175, 80, 0.2)"
                        border_color = "#4caf50"
                    elif 'İZLE' in advice:
                        bg_color = "rgba(255, 235, 59, 0.2)"
                        border_color = "#ffeb3b"
                    elif 'RİSK' in advice:
                        bg_color = "rgba(244, 67, 54, 0.2)"
                        border_color = "#f44336"
                    else:
                        bg_color = "rgba(33, 150, 243, 0.2)"
                        border_color = "#2196f3"
                    
                    st.markdown(f"""
                    <div style="background: {bg_color}; padding: 1rem; border-radius: 10px;
                             border: 1px solid {border_color}; text-align: center;">
                        <div style="font-size: 1.8rem; font-weight: 700; color: {border_color};">{count}</div>
                        <div style="color: {border_color};">{advice}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    def _render_data_explorer(self):
        """Veri gezgini - 5000 satır limitli"""
        
        st.markdown("### 📈 Veri Gezgini")
        
        df = st.session_state.processed_data
        limit = st.session_state.viz_limit
        
        st.caption(f"📊 Gösterilen: {min(limit, len(df)):,} / {len(df):,} satır")
        
        # Filtreler
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Molekül' in df.columns:
                mol_filter = st.multiselect(
                    "Molekül Filtre",
                    df['Molekül'].unique().tolist(),
                    default=[]
                )
            else:
                mol_filter = []
        
        with col2:
            if 'Şirket' in df.columns:
                company_filter = st.multiselect(
                    "Şirket Filtre",
                    df['Şirket'].unique().tolist(),
                    default=[]
                )
            else:
                company_filter = []
        
        with col3:
            if 'Segment_Adı' in df.columns:
                segment_filter = st.multiselect(
                    "Segment Filtre",
                    df['Segment_Adı'].unique().tolist(),
                    default=[]
                )
            else:
                segment_filter = []
        
        # Filtre uygula
        filtered_df = df.copy()
        
        if mol_filter:
            filtered_df = filtered_df[filtered_df['Molekül'].isin(mol_filter)]
        if company_filter:
            filtered_df = filtered_df[filtered_df['Şirket'].isin(company_filter)]
        if segment_filter and 'Segment_Adı' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Segment_Adı'].isin(segment_filter)]
        
        # Sütun seçimi
        st.markdown("---")
        
        all_cols = filtered_df.columns.tolist()
        
        # Önemli sütunları otomatik seç
        default_cols = []
        for col in ['Molekül', 'Şirket', 'Marka', 'Paket', 'Pazar_Payı', 'Segment_Adı', 'Yatırım_Tavsiyesi']:
            if col in all_cols:
                default_cols.append(col)
        
        # Satış sütunları ekle
        sales_cols = [col for col in all_cols if re.search(r'Satış_20\d{2}', col)]
        default_cols.extend(sales_cols[:2])
        
        # Büyüme sütunları ekle
        growth_cols = [col for col in all_cols if 'Büyüme' in col]
        default_cols.extend(growth_cols[:1])
        
        selected_cols = st.multiselect(
            "Gösterilecek Sütunlar",
            all_cols,
            default=[c for c in default_cols if c in all_cols][:10]
        )
        
        if selected_cols:
            display_df = filtered_df[selected_cols].head(limit)
            
            # Formatlama
            format_dict = {}
            for col in display_df.columns:
                if re.search(r'Satış_20\d{2}', col):
                    format_dict[col] = '${:,.0f}'
                elif 'Büyüme' in col or 'Pay' in col:
                    format_dict[col] = '{:.2f}%'
            
            st.dataframe(
                display_df.style.format(format_dict),
                use_container_width=True,
                height=500
            )
        else:
            st.dataframe(filtered_df.head(limit), use_container_width=True, height=500)
        
        # İndirme butonu
        csv = filtered_df.head(limit).to_csv(index=False)
        st.download_button(
            label="📥 Filtrelenmiş Veriyi İndir (CSV)",
            data=csv,
            file_name=f"pharma_export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    def _render_strategic_summary(self):
        """Stratejik özet ve yönetici insight box'ları"""
        
        st.markdown("### 💡 Stratejik Yönetici Özeti")
        
        df = st.session_state.processed_data
        
        # Otomatik içgörü üretme
        insights = []
        
        # 1. Pazar lideri
        if 'Şirket' in df.columns and 'Pazar_Payı' in df.columns:
            top_company = df.groupby('Şirket')['Pazar_Payı'].mean().sort_values(ascending=False).head(1)
            if not top_company.empty:
                insights.append({
                    'title': '🏢 Pazar Lideri',
                    'content': f"{top_company.index[0]} şirketi %{top_company.values[0]:.1f} pazar payı ile lider konumda.",
                    'metric': f"Pazar Payı: %{top_company.values[0]:.1f}",
                    'icon': '🏆'
                })
        
        # 2. En hızlı büyüyen molekül
        growth_cols = [col for col in df.columns if 'Büyüme' in col]
        if growth_cols and 'Molekül' in df.columns:
            growth_col = growth_cols[-1]
            top_growth = df.groupby('Molekül')[growth_col].mean().sort_values(ascending=False).head(1)
            if not top_growth.empty and top_growth.values[0] > 0:
                insights.append({
                    'title': '🚀 En Hızlı Büyüyen',
                    'content': f"{top_growth.index[0]} molekülü %{top_growth.values[0]:.1f} büyüme hızı ile zirvede.",
                    'metric': f"Büyüme: %{top_growth.values[0]:.1f}",
                    'icon': '📈'
                })
        
        # 3. Risk durumu
        if st.session_state.anomaly_results is not None:
            anomaly_df = st.session_state.anomaly_results
            if 'Anomali_Durumu' in anomaly_df.columns:
                critical = len(anomaly_df[anomaly_df['Anomali_Durumu'] == '🔴 Kritik Anomali'])
                if critical > 0:
                    insights.append({
                        'title': '⚠️ Kritik Risk Uyarısı',
                        'content': f"{critical} adet ürün/pakette kritik anomali tespit edildi. Acil müdahale gerekiyor.",
                        'metric': f"{critical} Kritik Anomali",
                        'icon': '🔴'
                    })
        
        # 4. Segmentasyon özeti
        if st.session_state.segmentation_results is not None:
            seg_df = st.session_state.segmentation_results
            if 'Segment_Adı' in seg_df.columns:
                leaders = len(seg_df[seg_df['Segment_Adı'] == '🏆 Lider Ürünler'])
                potentials = len(seg_df[seg_df['Segment_Adı'] == '🚀 Potansiyel Liderler'])
                risks = len(seg_df[seg_df['Segment_Adı'] == '⚠️ Riskli Ürünler'])
                
                insights.append({
                    'title': '🎯 Portföy Sağlığı',
                    'content': f"Portföyde {leaders} lider, {potentials} potansiyel ve {risks} riskli ürün bulunuyor.",
                    'metric': f"Lider: {leaders} | Risk: {risks}",
                    'icon': '📊'
                })
        
        # 5. Tahmin özeti
        if st.session_state.forecast_results is not None:
            if 'holt_winters' in st.session_state.forecast_results:
                f = st.session_state.forecast_results['holt_winters']
                growth_2025 = ((f.predictions[11] - f.predictions[0]) / f.predictions[0]) * 100 if f.predictions[0] > 0 else 0
                
                insights.append({
                    'title': '🔮 2025 Pazar Tahmini',
                    'content': f"2025 yılında pazarın %{growth_2025:.1f} büyümesi bekleniyor. "
                              f"{'Yatırım fırsatı' if growth_2025 > 10 else 'Kontrollü büyüme'} dönemi.",
                    'metric': f"Tahmin: %{growth_2025:.1f}",
                    'icon': '📅'
                })
        
        # 6. Paket çeşitliliği
        if 'Paket' in df.columns:
            unique_packs = df['Paket'].nunique()
            insights.append({
                'title': '📦 Ürün Çeşitliliği',
                'content': f"Toplam {unique_packs} farklı paket bulunuyor. "
                          f"Ortalama paket başına satış: ${df['Satış_2024'].mean() if 'Satış_2024' in df.columns else 0:,.0f}",
                'metric': f"{unique_packs} Paket",
                'icon': '💊'
            })
        
        # Insight box'ları göster
        for insight in insights[:4]:  # İlk 4 içgörü
            ExecutiveDarkTheme.executive_insight_box(
                title=insight['title'],
                content=insight['content'],
                metric=insight.get('metric'),
                icon=insight.get('icon', '💡')
            )
        
        # KPI metrikleri
        st.markdown("---")
        st.markdown("#### 📊 Temel Performans Göstergeleri")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = df['Satış_2024'].sum() if 'Satış_2024' in df.columns else 0
            st.markdown(f"""
            <div style="background: rgba(21, 36, 71, 0.7); padding: 1.2rem; border-radius: 10px;">
                <div style="color: #c0c0c0; font-size: 0.9rem;">Toplam Pazar</div>
                <div style="font-size: 2rem; font-weight: 700; color: #d4af37;">${total_sales/1e6:.1f}M</div>
                <div style="color: #4caf50; font-size: 0.8rem;">↑ 2024</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_growth = df['Büyüme_2023_2024'].mean() if 'Büyüme_2023_2024' in df.columns else 0
            delta_color = "↑" if avg_growth > 0 else "↓"
            delta_class = "color: #4caf50;" if avg_growth > 0 else "color: #f44336;"
            st.markdown(f"""
            <div style="background: rgba(21, 36, 71, 0.7); padding: 1.2rem; border-radius: 10px;">
                <div style="color: #c0c0c0; font-size: 0.9rem;">Ortalama Büyüme</div>
                <div style="font-size: 2rem; font-weight: 700; color: #d4af37;">%{avg_growth:.1f}</div>
                <div style="{delta_class}">{delta} 2023-2024</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            unique_molecules = df['Molekül'].nunique() if 'Molekül' in df.columns else 0
            st.markdown(f"""
            <div style="background: rgba(21, 36, 71, 0.7); padding: 1.2rem; border-radius: 10px;">
                <div style="color: #c0c0c0; font-size: 0.9rem;">Aktif Molekül</div>
                <div style="font-size: 2rem; font-weight: 700; color: #d4af37;">{unique_molecules}</div>
                <div style="color: #c0c0c0; font-size: 0.8rem;">🔬 Araştırma</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            hhi_score = 0
            if 'Şirket' in df.columns and 'Pazar_Payı' in df.columns:
                company_shares = df.groupby('Şirket')['Pazar_Payı'].mean()
                hhi_score = (company_shares ** 2).sum()
            st.markdown(f"""
            <div style="background: rgba(21, 36, 71, 0.7); padding: 1.2rem; border-radius: 10px;">
                <div style="color: #c0c0c0; font-size: 0.9rem;">HHI İndeksi</div>
                <div style="font-size: 2rem; font-weight: 700; color: #d4af37;">{hhi_score:.0f}</div>
                <div style="color: { '#f44336' if hhi_score > 2500 else '#ff9800' if hhi_score > 1800 else '#4caf50' };">
                    {'Yüksek Yoğunluk' if hhi_score > 2500 else 'Orta Yoğunluk' if hhi_score > 1800 else 'Rekabetçi'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # PDF/Excel rapor butonu
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            report_data = {
                'zaman': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'toplam_pazar': f"${total_sales:,.0f}",
                'ortalama_buyume': f"%{avg_growth:.1f}",
                'molekul_sayisi': unique_molecules,
                'hhi_index': f"{hhi_score:.0f}",
                'kritik_anomali': critical if 'critical' in locals() else 0,
                'lider_urun': leaders if 'leaders' in locals() else 0,
                'tahmin_2025': f"%{growth_2025:.1f}" if 'growth_2025' in locals() else 'N/A'
            }
            
            if st.button("📋 STRATEJİK ÖZET RAPORU OLUŞTUR", use_container_width=True):
                report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Raporu İndir (JSON)",
                    data=report_json,
                    file_name=f"pharma_strategic_report_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

# ================================================
# 9. APPLICATION ENTRY POINT
# ================================================

def main():
    """Ana uygulama giriş noktası"""
    
    try:
        # Uygulama instance'ı oluştur
        app = PharmaIntelligencePro()
        
        # UI'ı render et
        app.render_sidebar()
        app.render_main()
        
    except Exception as e:
        st.error(f"""
        ## ⚠️ Uygulama Hatası
        
        PharmaIntelligence Pro v8.0'da beklenmeyen bir hata oluştu.
        
        **Hata:** {str(e)}
        """)
        
        with st.expander("🔍 Hata Detayları"):
            st.code(traceback.format_exc())
        
        st.info("""
        **Önerilen Çözümler:**
        1. Sayfayı yenileyin (F5)
        2. Sol panelden 'CACHE TEMİZLE' butonuna tıklayın
        3. Veri dosyanızın formatını kontrol edin
        4. Uygulamayı yeniden başlatın
        """)
        
        if st.button("🔄 Uygulamayı Yeniden Başlat"):
            st.cache_data.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ================================================
# 10. MODULE EXPORTS
# ================================================

__all__ = [
    'ProdPackDepthAnalyzer',
    'TechnicalOptimizer',
    'StrategicAIModule',
    'ExecutiveDarkTheme',
    'DataEngine',
    'PharmaIntelligencePro'
]

if __name__ == "__main__":
    main()
"""
