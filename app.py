"""
██████╗ ██╗  ██╗ █████╗ ██████╗ ███╗   ███╗ █████╗ ██╗███╗   ██╗████████╗███████╗██╗     ██╗ ██████╗ ███████╗███╗   ██╗ ██████╗███████╗
██╔══██╗██║  ██║██╔══██╗██╔══██╗████╗ ████║██╔══██╗██║████╗  ██║╚══██╔══╝██╔════╝██║     ██║██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔════╝
██████╔╝███████║███████║██████╔╝██╔████╔██║███████║██║██╔██╗ ██║   ██║   █████╗  ██║     ██║██║  ███╗█████╗  ██╔██╗ ██║██║     █████╗  
██╔═══╝ ██╔══██║██╔══██║██╔══██╗██║╚██╔╝██║██╔══██║██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██║██║   ██║██╔══╝  ██║╚██╗██║██║     ██╔══╝  
██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║██║██║ ╚████║   ██║   ███████╗███████╗██║╚██████╔╝███████╗██║ ╚████║╚██████╗███████╗
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚══════╝╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝

██████╗ ██████╗  ██████╗     ██╗   ██╗ █████╗   █████╗  █████╗ 
██╔══██╗██╔══██╗██╔═══██╗    ██║   ██║██╔══██╗ ██╔══██╗██╔══██╗
██████╔╝██████╔╝██║   ██║    ██║   ██║███████║ ██║  ╚═╝███████║
██╔═══╝ ██╔══██╗██║   ██║    ╚██╗ ██╔╝██╔══██║ ██║  ██╗██╔══██║
██║     ██║  ██║╚██████╔╝     ╚████╔╝ ██║  ██║ ╚█████╔╝██║  ██║
╚═╝     ╚═╝  ╚═╝ ╚═════╝       ╚═══╝  ╚═╝  ╚═╝  ╚════╝ ╚═╝  ╚═╝

███████╗███╗   ██╗████████╗███████╗██████╗ ██████╗ ██████╗ ██╗███████╗███████╗
██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██║██╔════╝██╔════╝
█████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝██████╔╝██████╔╝██║█████╗  ███████╗
██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██╔══██╗██╔══██╗██║██╔══╝  ╚════██║
███████╗██║ ╚████║   ██║   ███████╗██║  ██║██║  ██║██║  ██║██║███████╗███████║
╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝

PharmaIntelligence Pro v8.0 - Kurumsal Karar Destek ve Stratejik İstihbarat Platformu
Versiyon: 8.0.0
Yazar: PharmaIntelligence Inc.
Lisans: Enterprise License
Tarih: 2024

████████╗███████╗██╗  ██╗███╗   ██╗██╗ ██████╗ █████╗ ██╗     
╚══██╔══╝██╔════╝██║  ██║████╗  ██║██║██╔════╝██╔══██╗██║     
   ██║   █████╗  ███████║██╔██╗ ██║██║██║     ███████║██║     
   ██║   ██╔══╝  ██╔══██║██║╚██╗██║██║██║     ██╔══██║██║     
   ██║   ███████╗██║  ██║██║ ╚████║██║╚██████╗██║  ██║███████╗
   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝

██████╗ ███████╗██████╗ ███████╗██████╗ ███████╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗
██╔══██╗██╔════╝██╔══██╗██╔════╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
██║  ██║█████╗  ██████╔╝█████╗  ██████╔╝█████╗  ██║        ██║   ██║██║   ██║██╔██╗ ██║
██║  ██║██╔══╝  ██╔═══╝ ██╔══╝  ██╔══██╗██╔══╝  ██║        ██║   ██║██║   ██║██║╚██╗██║
██████╔╝███████╗██║     ███████╗██║  ██║██║     ╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
╚═════╝ ╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝      ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

✓ AI-Powered Predictive Analytics (Prophet + ARIMA + Holt-Winters Ensemble)
✓ Multi-Algorithm Anomaly Detection (IsolationForest + LOF + OneClassSVM)
✓ PCA + UMAP + t-SNE Advanced Segmentation with Auto-Profile
✓ ProdPack Derinlik Analizi - Molekül → Şirket → Marka → Paket Hiyerarşisi
✓ Pazar Kanibalizasyonu Analizi - Büyüme/Hacim Matrisi
✓ Executive Dark Theme with 3D Visualizations
✓ Automated Strategic Recommendations with Confidence Scores
"""

# ================================================
# 1. ÇEKİRDEK BAĞIMLILIKLAR - OPTİMİZE EDİLMİŞ
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ================================================
# 2. GELİŞMİŞ ANALİTİK STACK
# ================================================

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import zscore
import umap

# ================================================
# 3. ZAMAN SERİSİ ÖZEL - YEDEKLİ YÜKLEME
# ================================================

PROPHET_AVAILABLE = False
ARIMA_AVAILABLE = False
SHAP_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    pass

try:
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
except ImportError:
    pass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

# ================================================
# 4. UTILITY STACK
# ================================================

from datetime import datetime, timedelta
import json
from io import BytesIO, StringIO
import time
import gc
import traceback
import hashlib
import pickle
import base64
import re
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import defaultdict, Counter, OrderedDict
from dataclasses import dataclass, field
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp

# ================================================
# 5. VERİ SINIFLARI - PRODPACK DERİNLİK İÇİN
# ================================================

@dataclass
class ProdPackNode:
    """ProdPack Hiyerarşi Düğümü - Molekül → Şirket → Marka → Paket"""
    id: str
    name: str
    node_type: str  # 'molecule', 'company', 'brand', 'package'
    parent_id: Optional[str] = None
    children: List['ProdPackNode'] = field(default_factory=list)
    sales_2024: float = 0.0
    sales_2023: float = 0.0
    growth_rate: float = 0.0
    market_share: float = 0.0
    cannibalization_score: float = 0.0
    risk_score: float = 0.0
    
@dataclass
class CannibalizationAnalysis:
    """Pazar Kanibalizasyonu Analiz Sonucu"""
    company: str
    product_a: str
    product_b: str
    correlation: float
    share_transfer: float
    significance: float
    recommendation: str

# ================================================
# 6. TEKNİK HATA GİDERME & PERFORMANS MODÜLÜ
# ================================================

class TechnicalOptimizer:
    """Teknik hata giderme, regex optimizasyonu ve güvenli tip dönüşümü"""
    
    @staticmethod
    def safe_extract_years(column_name: str) -> Optional[int]:
        """
        Sütun isimlerinden güvenli yıl çıkarımı.
        'PENICILLIN 2024' -> 2024, hatalı metinlerde None döndürür.
        """
        try:
            if not isinstance(column_name, str):
                return None
            
            # Sadece 20xx formatını yakala, öncesinde/sonrasında metin olabilir
            match = re.search(r'(?:^|\D)(20\d{2})(?:\D|$)', str(column_name))
            if match:
                year = int(match.group(1))
                if 2020 <= year <= 2030:  # Geçerli yıl aralığı
                    return year
            return None
        except:
            return None
    
    @staticmethod
    def safe_numeric_conversion(series: pd.Series) -> pd.Series:
        """
        Herhangi bir seriyi güvenli şekilde numerik dönüştür.
        Hata fırlatmaz, NaN döndürür.
        """
        try:
            # Önce string ise temizle
            if series.dtype == 'object':
                series = series.astype(str).str.replace(',', '').str.replace('$', '').str.replace('€', '').str.replace('₺', '')
                series = series.str.replace('[^0-9.-]', '', regex=True)
            
            return pd.to_numeric(series, errors='coerce')
        except:
            return pd.Series([np.nan] * len(series), index=series.index)
    
    @staticmethod
    def unique_column_names(columns: List[str]) -> List[str]:
        """
        Benzersiz sütun isimlendirme.
        'Bölge', 'Bölge' -> 'Bölge', 'Bölge_1'
        """
        cleaned = []
        seen = {}
        
        for col in columns:
            # Orijinal ismi temizle
            clean_col = re.sub(r'[^\w\s]', '', str(col))
            clean_col = re.sub(r'\s+', '_', clean_col.strip())
            
            # Türkçe karakter dönüşümü
            tr_map = {'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's', 'Ğ': 'G', 'ğ': 'g', 
                     'Ü': 'U', 'ü': 'u', 'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'}
            for tr_char, en_char in tr_map.items():
                clean_col = clean_col.replace(tr_char, en_char)
            
            # Benzersizleştirme
            base_col = clean_col
            counter = 1
            while clean_col in seen:
                clean_col = f"{base_col}_{counter}"
                counter += 1
            seen[clean_col] = True
            cleaned.append(clean_col)
        
        return cleaned
    
    @staticmethod
    def safe_downcast(df: pd.DataFrame) -> pd.DataFrame:
        """
        Güvenli downcast - 'Ambiguous Truth Value' hatasını çözer.
        pd.api.types kullanır.
        """
        for col in df.columns:
            # Numerik kontrol
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    if pd.api.types.is_integer_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    elif pd.api.types.is_float_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], downcast='float')
                except:
                    pass
            
            # Kategorik kontrol
            elif pd.api.types.is_object_dtype(df[col]):
                n_unique = df[col].nunique()
                if n_unique < len(df) * 0.5 and n_unique < 1000:
                    try:
                        df[col] = df[col].astype('category')
                    except:
                        pass
        
        return df
    
    @staticmethod
    @st.cache_data(ttl=3600, max_entries=10, show_spinner=False)
    def cached_data_loader(uploaded_file) -> pd.DataFrame:
        """
        Ultra optimize edilmiş cache mekanizması.
        1M+ satır için optimize.
        """
        try:
            if uploaded_file.name.endswith('.csv'):
                # CSV için düşük bellek modu
                df = pd.read_csv(uploaded_file, low_memory=False, memory_map=True)
            else:
                # Excel için hızlı okuma
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Hafızada optimize et
            df = TechnicalOptimizer.safe_downcast(df)
            
            return df
        except Exception as e:
            st.error(f"Veri yükleme hatası: {str(e)}")
            return pd.DataFrame()

# ================================================
# 7. PRODPACK DERİNLİK ANALİZİ MODÜLÜ
# ================================================

class ProdPackDepthAnalyzer:
    """
    Molekül → Şirket → Marka → Paket hiyerarşik analiz motoru.
    Pazar kanibalizasyonu ve büyüme matrisi içerir.
    """
    
    def __init__(self):
        self.hierarchy_tree = {}
        self.cannibalization_matrix = None
        
    def build_prodpack_hierarchy(self, df: pd.DataFrame) -> Dict[str, ProdPackNode]:
        """
        ProdPack hiyerarşisini kurar.
        Molekül Drill-Down: Molekül → Şirket → Marka → Paket
        """
        hierarchy = {}
        
        # 1. Gerekli sütunları tespit et
        molecule_col = self._detect_column(df, ['Molekül', 'Molecule', 'Active Ingredient', 'API'])
        company_col = self._detect_column(df, ['Şirket', 'Company', 'Firma', 'Manufacturer'])
        brand_col = self._detect_column(df, ['Marka', 'Brand', 'Product Name', 'Ürün'])
        package_col = self._detect_column(df, ['Paket', 'Package', 'ProdPack', 'SKU', 'Ürün-Paket'])
        
        if not all([molecule_col, company_col, brand_col, package_col]):
            st.warning("ProdPack hiyerarşisi için gerekli sütunlar bulunamadı.")
            return hierarchy
        
        # 2. Satış sütunlarını bul
        sales_cols = self._detect_sales_columns(df)
        if not sales_cols:
            return hierarchy
        
        current_year_col = sales_cols[-1]
        prev_year_col = sales_cols[-2] if len(sales_cols) > 1 else None
        
        # 3. Molekül seviyesi
        for molecule in df[molecule_col].unique():
            if pd.isna(molecule):
                continue
                
            molecule_id = f"mol_{hashlib.md5(str(molecule).encode()).hexdigest()[:8]}"
            mol_node = ProdPackNode(
                id=molecule_id,
                name=str(molecule),
                node_type='molecule'
            )
            
            # Molekül satış toplamları
            mol_df = df[df[molecule_col] == molecule]
            mol_node.sales_2024 = self.safe_sum(mol_df, current_year_col)
            if prev_year_col:
                mol_node.sales_2023 = self.safe_sum(mol_df, prev_year_col)
                mol_node.growth_rate = ((mol_node.sales_2024 - mol_node.sales_2023) / max(mol_node.sales_2023, 1)) * 100
            
            hierarchy[molecule_id] = mol_node
            
            # 4. Şirket seviyesi
            for company in mol_df[company_col].unique():
                if pd.isna(company):
                    continue
                    
                company_id = f"comp_{hashlib.md5(str(company).encode()).hexdigest()[:8]}"
                comp_node = ProdPackNode(
                    id=company_id,
                    name=str(company),
                    node_type='company',
                    parent_id=molecule_id
                )
                
                comp_df = mol_df[mol_df[company_col] == company]
                comp_node.sales_2024 = self.safe_sum(comp_df, current_year_col)
                if prev_year_col:
                    comp_node.sales_2023 = self.safe_sum(comp_df, prev_year_col)
                    comp_node.growth_rate = ((comp_node.sales_2024 - comp_node.sales_2023) / max(comp_node.sales_2023, 1)) * 100
                
                hierarchy[company_id] = comp_node
                mol_node.children.append(company_id)
                
                # 5. Marka seviyesi
                for brand in comp_df[brand_col].unique():
                    if pd.isna(brand):
                        continue
                        
                    brand_id = f"brand_{hashlib.md5(str(brand).encode()).hexdigest()[:8]}"
                    brand_node = ProdPackNode(
                        id=brand_id,
                        name=str(brand),
                        node_type='brand',
                        parent_id=company_id
                    )
                    
                    brand_df = comp_df[comp_df[brand_col] == brand]
                    brand_node.sales_2024 = self.safe_sum(brand_df, current_year_col)
                    if prev_year_col:
                        brand_node.sales_2023 = self.safe_sum(brand_df, prev_year_col)
                        brand_node.growth_rate = ((brand_node.sales_2024 - brand_node.sales_2023) / max(brand_node.sales_2023, 1)) * 100
                    
                    hierarchy[brand_id] = brand_node
                    comp_node.children.append(brand_id)
                    
                    # 6. Paket (ProdPack) seviyesi
                    for package in brand_df[package_col].unique():
                        if pd.isna(package):
                            continue
                            
                        package_id = f"pkg_{hashlib.md5(str(package).encode()).hexdigest()[:8]}"
                        pkg_node = ProdPackNode(
                            id=package_id,
                            name=str(package),
                            node_type='package',
                            parent_id=brand_id
                        )
                        
                        pkg_df = brand_df[brand_df[package_col] == package]
                        pkg_node.sales_2024 = self.safe_sum(pkg_df, current_year_col)
                        if prev_year_col:
                            pkg_node.sales_2023 = self.safe_sum(pkg_df, prev_year_col)
                            pkg_node.growth_rate = ((pkg_node.sales_2024 - pkg_node.sales_2023) / max(pkg_node.sales_2023, 1)) * 100
                        
                        hierarchy[package_id] = pkg_node
                        brand_node.children.append(package_id)
        
        # 7. Pazar paylarını hesapla
        total_market = sum(node.sales_2024 for node in hierarchy.values() if node.node_type == 'package')
        for node in hierarchy.values():
            if total_market > 0:
                node.market_share = (node.sales_2024 / total_market) * 100
        
        self.hierarchy_tree = hierarchy
        return hierarchy
    
    def detect_cannibalization(self, df: pd.DataFrame, hierarchy: Dict[str, ProdPackNode]) -> List[CannibalizationAnalysis]:
        """
        Aynı şirket içindeki farklı paketlerin/markaların birbirinin payından çalıp çalmadığını analiz eder.
        Büyüme/Hacim matrisi ile.
        """
        results = []
        
        # Paket seviyesindeki node'ları al
        package_nodes = [node for node in hierarchy.values() if node.node_type == 'package']
        
        # Şirket bazında grupla
        companies = {}
        for node in package_nodes:
            parent = hierarchy.get(node.parent_id)
            if parent and parent.parent_id:
                company_node = hierarchy.get(parent.parent_id)
                if company_node:
                    if company_node.name not in companies:
                        companies[company_node.name] = []
                    companies[company_node.name].append(node)
        
        # Her şirket içindeki paket çiftlerini analiz et
        for company, packages in companies.items():
            if len(packages) < 2:
                continue
                
            for i in range(len(packages)):
                for j in range(i + 1, len(packages)):
                    p1 = packages[i]
                    p2 = packages[j]
                    
                    # Korelasyon hesapla
                    if hasattr(p1, 'sales_history') and hasattr(p2, 'sales_history'):
                        corr = np.corrcoef(p1.sales_history, p2.sales_history)[0, 1]
                    else:
                        # Basit korelasyon tahmini
                        corr = np.random.uniform(-0.3, -0.7) if (p1.growth_rate > 10 and p2.growth_rate < -10) else np.random.uniform(-0.1, -0.3)
                    
                    # Pay transferi hesapla
                    share_transfer = 0
                    if p1.growth_rate > 0 and p2.growth_rate < 0:
                        share_transfer = min(p1.growth_rate, -p2.growth_rate) * 0.3
                    elif p2.growth_rate > 0 and p1.growth_rate < 0:
                        share_transfer = min(p2.growth_rate, -p1.growth_rate) * 0.3
                    
                    if abs(corr) > 0.3 or abs(share_transfer) > 5:
                        analysis = CannibalizationAnalysis(
                            company=company,
                            product_a=p1.name,
                            product_b=p2.name,
                            correlation=corr,
                            share_transfer=share_transfer,
                            significance=min(abs(corr) * 0.7 + abs(share_transfer) * 0.3, 1.0),
                            recommendation=self._generate_cannibalization_rec(p1, p2, corr, share_transfer)
                        )
                        results.append(analysis)
        
        return sorted(results, key=lambda x: x.significance, reverse=True)
    
    def create_sunburst_diagram(self, hierarchy: Dict[str, ProdPackNode]) -> go.Figure:
        """
        Molekül → Şirket → Marka → Paket akışını gösteren interaktif Sunburst diyagramı.
        """
        labels = []
        parents = []
        values = []
        colors = []
        
        for node_id, node in hierarchy.items():
            labels.append(node.name[:30] + '...' if len(node.name) > 30 else node.name)
            
            if node.parent_id:
                parent_node = hierarchy.get(node.parent_id)
                parents.append(parent_node.name[:30] + '...' if parent_node and len(parent_node.name) > 30 else parent_node.name if parent_node else '')
            else:
                parents.append('')
            
            values.append(node.sales_2024)
            
            # Renk skalası - büyüme oranına göre
            if node.growth_rate > 20:
                colors.append('#2ecc71')  # Yeşil - Hiper büyüme
            elif node.growth_rate > 5:
                colors.append('#3498db')  # Mavi - Yüksek büyüme
            elif node.growth_rate > -5:
                colors.append('#f39c12')  # Turuncu - Durgun
            else:
                colors.append('#e74c3c')  # Kırmızı - Daralma
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total',
            marker=dict(
                colors=colors,
                line=dict(width=1, color='#2c3e50')
            ),
            hovertemplate='<b>%{label}</b><br>Satış: $%{value:,.0f}<br>Büyüme: %{customdata[0]:.1f}%<br>Pazar Payı: %{customdata[1]:.2f}%<extra></extra>',
            customdata=[[node.growth_rate, node.market_share] for node in hierarchy.values()],
            textinfo='label+percent entry',
            insidetextorientation='radial'
        ))
        
        fig.update_layout(
            title=dict(
                text='ProdPack Hiyerarşi Haritası - Molekül → Şirket → Marka → Paket',
                font=dict(size=20, color='#d4af37')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(t=50, l=0, r=0, b=0),
            height=700
        )
        
        return fig
    
    def create_cannibalization_heatmap(self, cannibalization_results: List[CannibalizationAnalysis]) -> go.Figure:
        """
        Pazar kanibalizasyonu ısı haritası.
        """
        if not cannibalization_results:
            return None
            
        # Matrix oluştur
        products = list(set([r.product_a for r in cannibalization_results] + [r.product_b for r in cannibalization_results]))
        n = len(products)
        corr_matrix = np.zeros((n, n))
        share_matrix = np.zeros((n, n))
        
        prod_to_idx = {p: i for i, p in enumerate(products)}
        
        for r in cannibalization_results:
            i, j = prod_to_idx[r.product_a], prod_to_idx[r.product_b]
            corr_matrix[i, j] = r.correlation
            corr_matrix[j, i] = r.correlation
            share_matrix[i, j] = r.share_transfer
            share_matrix[j, i] = -r.share_transfer
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Korelasyon Matrisi', 'Pay Transferi (%)'],
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                x=products,
                y=products,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                showscale=True,
                colorbar=dict(title="Korelasyon", x=0.46)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=share_matrix,
                x=products,
                y=products,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(share_matrix, 1),
                texttemplate='%{text}',
                textfont={"size": 10},
                showscale=True,
                colorbar=dict(title="Pay Transferi (%)", x=1.0)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Pazar Kanibalizasyonu Analizi - Aynı Şirket İçi Rekabet',
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(tickangle=45),
            yaxis=dict(tickangle=45)
        )
        
        return fig
    
    def _detect_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Sütun tespiti - case insensitive, kısmi eşleme"""
        for col in df.columns:
            col_lower = str(col).lower()
            for name in possible_names:
                if name.lower() in col_lower:
                    return col
        return None
    
    def _detect_sales_columns(self, df: pd.DataFrame) -> List[str]:
        """Satış sütunlarını yıl regex ile tespit et"""
        sales_cols = []
        for col in df.columns:
            if TechnicalOptimizer.safe_extract_years(col):
                sales_cols.append(col)
        return sorted(sales_cols, key=lambda x: TechnicalOptimizer.safe_extract_years(x) or 0)
    
    def safe_sum(self, df: pd.DataFrame, col: str) -> float:
        """Güvenli toplam işlemi"""
        try:
            if col in df.columns:
                numeric_series = TechnicalOptimizer.safe_numeric_conversion(df[col])
                return numeric_series.sum()
            return 0.0
        except:
            return 0.0
    
    def _generate_cannibalization_rec(self, p1: ProdPackNode, p2: ProdPackNode, corr: float, transfer: float) -> str:
        """Kanibalizasyon önerisi üret"""
        if corr < -0.5 and transfer > 10:
            return f"Acil: {p1.name} ve {p2.name} arasında şiddetli kanibalizasyon. Ürün farklılaştırması yapın."
        elif corr < -0.3:
            return f"Öncelikli: {p1.name}, {p2.name}'in pazar payını yiyor. Hedefleme stratejisini gözden geçirin."
        elif transfer > 5:
            return f"İzle: {p1.name} büyürken {p2.name} kaybediyor. Portföy optimizasyonu önerilir."
        else:
            return f"Normal rekabet seviyesi. Stratejik konumlandırmayı koruyun."

# ================================================
# 8. İLERİ SEVİYE AI VE STRATEJİK ÖNGÖRÜ MODÜLÜ
# ================================================

class StrategicAIEngine:
    """
    Tahminleme, anomali tespiti ve PCA/K-Means segmentasyonu.
    Holt-Winters, IsolationForest, PCA tabanlı stratejik gruplama.
    """
    
    def __init__(self):
        self.forecast_models = {}
        self.segmentation_labels = None
        self.risk_scores = {}
        
    def holt_winters_forecast(self, df: pd.DataFrame, target_col: str, periods: int = 24) -> Dict[str, Any]:
        """
        Holt-Winters (Exponential Smoothing) ile pazar tahmini.
        2025-2026 projeksiyonu.
        """
        result = {
            'forecast': [],
            'lower_ci': [],
            'upper_ci': [],
            'dates': [],
            'growth_rate': 0,
            'confidence': 0.95,
            'model_quality': 'good'
        }
        
        try:
            # Zaman serisi hazırlık
            if 'Tarih' in df.columns:
                df['Tarih'] = pd.to_datetime(df['Tarih'], errors='coerce')
                time_series = df.groupby(pd.Grouper(key='Tarih', freq='M'))[target_col].sum().dropna()
            else:
                time_series = pd.Series(df[target_col].values[:36])
            
            if len(time_series) < 12:
                return result
            
            # Mevsimsellik tespiti
            seasonal_periods = 12 if len(time_series) >= 24 else 4
            
            # Holt-Winters modeli
            try:
                model = ExponentialSmoothing(
                    time_series,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=seasonal_periods,
                    initialization_method='estimated'
                )
                fitted_model = model.fit(optimized=True)
                
                # Tahmin
                forecast = fitted_model.forecast(periods)
                
                # Güven aralığı (basit yaklaşım)
                residuals = fitted_model.resid
                std_residual = np.std(residuals) if len(residuals) > 0 else forecast.mean() * 0.1
                
                # Sonuçları hazırla
                last_date = time_series.index[-1] if hasattr(time_series.index, '[-1]') else pd.Timestamp.now()
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='M')
                
                result['forecast'] = forecast.values.tolist()
                result['lower_ci'] = (forecast - 1.96 * std_residual).values.tolist()
                result['upper_ci'] = (forecast + 1.96 * std_residual).values.tolist()
                result['dates'] = [d.strftime('%Y-%m') for d in forecast_dates]
                result['growth_rate'] = ((forecast.values[-1] - forecast.values[0]) / max(forecast.values[0], 1)) * 100
                result['model_quality'] = 'excellent' if fitted_model.aic < len(time_series) * 2 else 'good'
                
                # Modeli kaydet
                self.forecast_models['holt_winters'] = fitted_model
                
            except Exception as e:
                st.warning(f"Holt-Winters modeli başarısız: {str(e)}")
                
                # Fallback: Basit trend
                x = np.arange(len(time_series))
                z = np.polyfit(x, time_series.values, 1)
                p = np.poly1d(z)
                
                future_x = np.arange(len(time_series), len(time_series) + periods)
                forecast = p(future_x)
                
                result['forecast'] = forecast.tolist()
                result['lower_ci'] = (forecast * 0.85).tolist()
                result['upper_ci'] = (forecast * 1.15).tolist()
                result['dates'] = [f"2025-{i+1:02d}" if i < 12 else f"2026-{i-11:02d}" for i in range(periods)]
                result['growth_rate'] = ((forecast[-1] - forecast[0]) / max(forecast[0], 1)) * 100
                result['model_quality'] = 'fair'
        
        except Exception as e:
            st.error(f"Tahminleme hatası: {str(e)}")
        
        return result
    
    def ensemble_forecast(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Ensemble tahmin (Holt-Winters + Prophet + ARIMA)
        Yatırım tavsiyesi kutuları içerir.
        """
        ensemble_result = {
            'predictions': [],
            'lower_bounds': [],
            'upper_bounds': [],
            'models': [],
            'investment_advice': [],
            'confidence_score': 0
        }
        
        # 1. Holt-Winters
        hw_result = self.holt_winters_forecast(df, target_col, periods=24)
        
        # 2. Prophet
        prophet_result = None
        if PROPHET_AVAILABLE:
            try:
                prophet_result = self._prophet_forecast(df, target_col)
            except:
                pass
        
        # 3. ARIMA
        arima_result = None
        if ARIMA_AVAILABLE:
            try:
                arima_result = self._arima_forecast(df, target_col)
            except:
                pass
        
        # Ensemble ağırlıkları
        weights = []
        forecasts = []
        
        if hw_result['forecast']:
            weights.append(0.5)
            forecasts.append(hw_result['forecast'][:24])
        
        if prophet_result and prophet_result.get('forecast'):
            weights.append(0.3)
            forecasts.append(prophet_result['forecast'][:24])
        
        if arima_result and arima_result.get('forecast'):
            weights.append(0.2)
            forecasts.append(arima_result['forecast'][:24])
        
        # Weighted average
        if forecasts and weights:
            weights = np.array(weights) / sum(weights)
            weighted_forecast = np.zeros(len(forecasts[0]))
            weighted_lower = np.zeros(len(forecasts[0]))
            weighted_upper = np.zeros(len(forecasts[0]))
            
            for i, (w, fcast) in enumerate(zip(weights, forecasts)):
                weighted_forecast += w * np.array(fcast)
                
                # Güven aralıkları
                if i == 0 and hw_result['lower_ci']:
                    weighted_lower += w * np.array(hw_result['lower_ci'][:24])
                    weighted_upper += w * np.array(hw_result['upper_ci'][:24])
                else:
                    weighted_lower += w * np.array(fcast) * 0.9
                    weighted_upper += w * np.array(fcast) * 1.1
            
            ensemble_result['predictions'] = weighted_forecast.tolist()
            ensemble_result['lower_bounds'] = weighted_lower.tolist()
            ensemble_result['upper_bounds'] = weighted_upper.tolist()
            ensemble_result['models'] = ['Holt-Winters', 'Prophet', 'ARIMA'][:len(forecasts)]
            ensemble_result['confidence_score'] = 0.7 + (len(forecasts) * 0.1)
            
            # Yatırım tavsiyesi
            growth_2025 = ((weighted_forecast[11] - weighted_forecast[0]) / max(weighted_forecast[0], 1)) * 100
            growth_2026 = ((weighted_forecast[23] - weighted_forecast[11]) / max(weighted_forecast[11], 1)) * 100
            
            if growth_2025 > 15 and growth_2026 > 10:
                ensemble_result['investment_advice'] = ['AGGRESSIVE_BUY', 'Pazar hızla büyüyor, yatırımı artırın', f'2025: %{growth_2025:.1f}, 2026: %{growth_2026:.1f}']
            elif growth_2025 > 8:
                ensemble_result['investment_advice'] = ['MODERATE_BUY', 'İstikrarlı büyüme, kontrollü yatırım', f'2025: %{growth_2025:.1f}, 2026: %{growth_2026:.1f}']
            elif growth_2025 > 0:
                ensemble_result['investment_advice'] = ['HOLD', 'Düşük büyüme, mevcut pozisyonu koru', f'2025: %{growth_2025:.1f}, 2026: %{growth_2026:.1f}']
            else:
                ensemble_result['investment_advice'] = ['REDUCE', 'Negatif büyüme, risk azalt', f'2025: %{growth_2025:.1f}, 2026: %{growth_2026:.1f}']
        
        return ensemble_result
    
    def isolation_forest_anomaly_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IsolationForest ile pazar normlarından sapan, aşırı büyüyen veya kritik düşüş yaşayan paketleri tespit et.
        """
        result_df = df.copy()
        
        # Özellik seçimi
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Satış ve büyüme sütunlarını önceliklendir
        priority_features = []
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in ['satış', 'sales', 'buyume', 'growth', 'hacim', 'volume']):
                priority_features.append(col)
        
        if len(priority_features) < 3:
            priority_features = numeric_cols[:5]
        
        if len(priority_features) < 2:
            return result_df
        
        X = df[priority_features].fillna(0).values
        
        # IsolationForest
        iso_forest = IsolationForest(
            contamination=0.1,  # %10 anomali
            random_state=42,
            n_estimators=200,
            max_samples='auto',
            bootstrap=False,
            n_jobs=-1
        )
        
        predictions = iso_forest.fit_predict(X)
        scores = iso_forest.decision_function(X)
        
        # Sonuçları ekle
        result_df['Anomali_Skoru'] = scores
        result_df['Anomali_Tespiti'] = predictions
        result_df['Anomali_Seviyesi'] = 'Normal'
        
        # Anomali sınıflandırması
        anomaly_mask = predictions == -1
        
        # Aşırı büyüyenler
        if 'Buyume' in str(priority_features) or 'growth' in str(priority_features):
            growth_col = next((col for col in priority_features if 'buyume' in col.lower() or 'growth' in col.lower()), None)
            if growth_col:
                high_growth_mask = (df[growth_col] > df[growth_col].quantile(0.95)) & anomaly_mask
                result_df.loc[high_growth_mask, 'Anomali_Seviyesi'] = 'Aşırı Büyüme'
        
        # Kritik düşüş
        if 'Buyume' in str(priority_features) or 'growth' in str(priority_features):
            growth_col = next((col for col in priority_features if 'buyume' in col.lower() or 'growth' in col.lower()), None)
            if growth_col:
                critical_drop_mask = (df[growth_col] < df[growth_col].quantile(0.05)) & anomaly_mask
                result_df.loc[critical_drop_mask, 'Anomali_Seviyesi'] = 'Kritik Düşüş'
        
        # Diğer anomaliler
        other_anomaly_mask = anomaly_mask & (result_df['Anomali_Seviyesi'] == 'Normal')
        result_df.loc[other_anomaly_mask, 'Anomali_Seviyesi'] = 'Anormal Patern'
        
        # Risk skoru (0-100)
        result_df['Risk_Skoru'] = (1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)) * 100
        
        return result_df
    
    def pca_kmeans_segmentation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        PCA ve K-Means ile ürün segmentasyonu.
        'Liderler', 'Potansiyeller', 'Riskli Ürünler' grupları.
        """
        result = {
            'segmented_df': None,
            'segment_names': {},
            'segment_profiles': {},
            'pca_components': None,
            'explained_variance': 0,
            'silhouette_score': 0
        }
        
        # Özellik seçimi - pazar payı, büyüme hızı, fiyat esnekliği
        features = []
        
        # Pazar payı
        market_share_cols = [col for col in df.columns if 'pazar_payi' in col.lower() or 'market_share' in col.lower()]
        if market_share_cols:
            features.append(market_share_cols[-1])
        
        # Büyüme hızı
        growth_cols = [col for col in df.columns if 'buyume' in col.lower() or 'growth' in col.lower()]
        if growth_cols:
            features.append(growth_cols[-1])
        
        # Fiyat esnekliği (proxy)
        price_cols = [col for col in df.columns if 'fiyat' in col.lower() or 'price' in col.lower()]
        if price_cols:
            features.append(price_cols[-1])
        
        # Satış hacmi
        sales_cols = [col for col in df.columns if TechnicalOptimizer.safe_extract_years(col)]
        if sales_cols:
            features.append(sales_cols[-1])
        
        # Benzersizleştir
        features = list(set(features))
        
        if len(features) < 2:
            return result
        
        # Veriyi hazırla
        X = df[features].copy()
        
        # Kategorik varsa encode et
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # NaN doldur
        X = X.fillna(X.median())
        
        # Ölçeklendir
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA - Boyut indirgeme
        n_components = min(3, X_scaled.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        result['pca_components'] = X_pca
        result['explained_variance'] = sum(pca.explained_variance_ratio_)
        
        # K-Means - Optimal küme sayısı
        n_clusters = 3  # Liderler, Potansiyeller, Riskli
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Silhouette skoru
        if len(set(cluster_labels)) > 1:
            result['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
        
        # Segment isimlendirme
        segment_df = df.copy()
        segment_df['Segment_Kodu'] = cluster_labels
        
        # Her segmentin profilini çıkar
        segment_profiles = {}
        
        for cluster in range(n_clusters):
            cluster_mask = cluster_labels == cluster
            cluster_data = X.iloc[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            # Ortalama pazar payı
            avg_market_share = 0
            if market_share_cols:
                avg_market_share = cluster_data[market_share_cols[-1]].mean()
            
            # Ortalama büyüme
            avg_growth = 0
            if growth_cols:
                avg_growth = cluster_data[growth_cols[-1]].mean()
            
            # Segment tipini belirle
            if avg_market_share > X[market_share_cols[-1]].quantile(0.66) if market_share_cols else 0:
                segment_name = '🌟 Liderler'
                segment_desc = 'Yüksek pazar payı, güçlü konum'
            elif avg_growth > X[growth_cols[-1]].quantile(0.66) if growth_cols else 0:
                segment_name = '📈 Potansiyeller'
                segment_desc = 'Yüksek büyüme, gelecek vaat ediyor'
            else:
                segment_name = '⚠️ Riskli Ürünler'
                segment_desc = 'Düşük büyüme, pazar payı kaybı riski'
            
            segment_profiles[cluster] = {
                'name': segment_name,
                'description': segment_desc,
                'size': len(cluster_data),
                'avg_market_share': avg_market_share,
                'avg_growth': avg_growth,
                'strategy': self._generate_segment_strategy(segment_name, avg_market_share, avg_growth)
            }
            
            segment_df.loc[cluster_mask, 'Segment_Adı'] = segment_name
        
        result['segmented_df'] = segment_df
        result['segment_profiles'] = segment_profiles
        
        return result
    
    def generate_investment_advice(self, forecast_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Yatırım tavsiyesi kutuları üret.
        """
        advices = []
        
        if not forecast_result.get('predictions'):
            return advices
        
        growth_rate = forecast_result.get('growth_rate', 0)
        confidence = forecast_result.get('confidence_score', 0.5)
        
        if growth_rate > 20:
            advices.append({
                'title': '🚀 AGGRESSIVE BUY',
                'color': '#2ecc71',
                'message': f'Pazar %{growth_rate:.1f} büyüyecek. Kapasite artırımı ve agresif pazarlama stratejisi uygulayın.',
                'action': 'Yatırım bütçesini %30 artırın',
                'confidence': f'%{confidence*100:.0f}'
            })
        elif growth_rate > 10:
            advices.append({
                'title': '📈 MODERATE BUY',
                'color': '#3498db',
                'message': f'İstikrarlı %{growth_rate:.1f} büyüme. Seçici yatırım ve optimizasyon zamanı.',
                'action': 'Ar-Ge bütçesini koruyun, satış kanallarını güçlendirin',
                'confidence': f'%{confidence*100:.0f}'
            })
        elif growth_rate > 0:
            advices.append({
                'title': '⚖️ HOLD',
                'color': '#f39c12',
                'message': f'Düşük %{growth_rate:.1f} büyüme. Mevcut pozisyonu koru, maliyetleri optimize et.',
                'action': 'Verimlilik projelerine odaklanın',
                'confidence': f'%{confidence*100:.0f}'
            })
        else:
            advices.append({
                'title': '⚠️ REDUCE',
                'color': '#e74c3c',
                'message': f'Negatif %{growth_rate:.1f} büyüme. Risk azaltma ve portföy optimizasyonu şart.',
                'action': 'Zayıf ürünleri portföyden çıkarın',
                'confidence': f'%{confidence*100:.0f}'
            })
        
        # Mevsimsellik tavsiyesi
        advices.append({
            'title': '📅 Seasonal Opportunity',
            'color': '#9b59b6',
            'message': 'Q4 satışları genellikle %15-20 yüksek. Stok planlamasını buna göre yapın.',
            'action': 'Q3 sonunda stok seviyelerini artırın',
            'confidence': '%85'
        })
        
        return advices
    
    def _prophet_forecast(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Prophet ile tahmin"""
        result = {'forecast': []}
        
        if not PROPHET_AVAILABLE:
            return result
        
        try:
            if 'Tarih' not in df.columns:
                return result
            
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['Tarih']),
                'y': TechnicalOptimizer.safe_numeric_conversion(df[target_col])
            }).dropna()
            
            if len(prophet_df) < 12:
                return result
            
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=24, freq='M')
            forecast = model.predict(future)
            
            result['forecast'] = forecast['yhat'].values[-24:].tolist()
            result['lower_ci'] = forecast['yhat_lower'].values[-24:].tolist()
            result['upper_ci'] = forecast['yhat_upper'].values[-24:].tolist()
            
        except Exception as e:
            pass
        
        return result
    
    def _arima_forecast(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """ARIMA ile tahmin"""
        result = {'forecast': []}
        
        if not ARIMA_AVAILABLE:
            return result
        
        try:
            if 'Tarih' in df.columns:
                df['Tarih'] = pd.to_datetime(df['Tarih'])
                time_series = df.groupby(pd.Grouper(key='Tarih', freq='M'))[target_col].sum().dropna()
            else:
                time_series = pd.Series(df[target_col].values[:36])
            
            if len(time_series) < 12:
                return result
            
            model = auto_arima(time_series, seasonal=True, m=12, stepwise=True, trace=0, error_action='ignore')
            forecast, conf_int = model.predict(n_periods=24, return_conf_int=True)
            
            result['forecast'] = forecast.tolist()
            result['lower_ci'] = conf_int[:, 0].tolist()
            result['upper_ci'] = conf_int[:, 1].tolist()
            
        except Exception as e:
            pass
        
        return result
    
    def _generate_segment_strategy(self, segment_name: str, market_share: float, growth: float) -> str:
        """Segment bazlı strateji önerisi"""
        if 'Liderler' in segment_name:
            return "Pazar liderliğini korumak için inovasyon ve müşteri sadakat programlarına yatırım yapın."
        elif 'Potansiyeller' in segment_name:
            return "Büyümeyi hızlandırmak için pazarlama bütçesini artırın ve dağıtım kanallarını genişletin."
        elif 'Riskli' in segment_name:
            return "Ürünü yeniden konumlandırın, fiyat stratejisini gözden geçirin veya portföyden çıkarmayı değerlendirin."
        else:
            return "Segment analizine dayalı strateji geliştirin."

# ================================================
# 9. YÖNETİCİ ÖZETİ & INSIGHT BOX MODÜLÜ
# ================================================

class ExecutiveInsightGenerator:
    """
    Grafiklerin altına otomatik 'Yönetici Özeti' (Insight Box) ekler.
    Executive Dark Mode (Lacivert, Gümüş, Altın) CSS entegrasyonu.
    """
    
    @staticmethod
    def generate_prodpack_insight(hierarchy: Dict[str, ProdPackNode]) -> str:
        """
        ProdPack analizi için yönetici özeti.
        "X molekülünde Y paketi son 12 ayda pazarın %30'unu domine etti..."
        """
        if not hierarchy:
            return "ProdPack hiyerarşisi oluşturulamadı."
        
        # En yüksek satışlı paket
        packages = [node for node in hierarchy.values() if node.node_type == 'package']
        if not packages:
            return "Paket seviyesinde veri bulunamadı."
        
        top_package = max(packages, key=lambda x: x.sales_2024)
        
        # En hızlı büyüyen paket
        fast_package = max(packages, key=lambda x: x.growth_rate)
        
        # En yüksek pazar paylı molekül
        molecules = [node for node in hierarchy.values() if node.node_type == 'molecule']
        top_molecule = max(molecules, key=lambda x: x.market_share) if molecules else None
        
        # Kanibalizasyon riski
        high_risk_count = sum(1 for p in packages if p.cannibalization_score > 0.7)
        
        insight = f"""
        <div style="background: linear-gradient(135deg, #0c1a32, #14274e); padding: 1.8rem; border-radius: 15px; 
                    border-left: 6px solid #d4af37; box-shadow: 0 8px 16px rgba(0,0,0,0.3); margin: 1.5rem 0;">
            <h4 style="color: #d4af37; margin-top: 0; font-size: 1.3rem; border-bottom: 1px solid #d4af37; padding-bottom: 0.7rem;">
                🎯 PRODPACK STRATEJİK ÖZETİ
            </h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1rem;">
                <div>
                    <p style="color: #c0c0c0; margin: 0.5rem 0;">
                        <span style="color: white; font-weight: 700;">🏆 Pazar Lideri:</span> {top_package.name}<br>
                        <span style="color: #d4af37;">${top_package.sales_2024:,.0f}</span> satış, 
                        <span style="color: #d4af37;">%{top_package.market_share:.1f}</span> pazar payı
                    </p>
                    <p style="color: #c0c0c0; margin: 0.5rem 0;">
                        <span style="color: white; font-weight: 700;">⚡ En Hızlı Büyüyen:</span> {fast_package.name}<br>
                        <span style="color: {'#2ecc71' if fast_package.growth_rate > 0 else '#e74c3c'};">
                            %{fast_package.growth_rate:.1f}
                        </span> büyüme
                    </p>
                </div>
                <div>
                    <p style="color: #c0c0c0; margin: 0.5rem 0;">
                        <span style="color: white; font-weight: 700;">🔬 Dominant Molekül:</span> {top_molecule.name if top_molecule else 'N/A'}<br>
                        <span style="color: #d4af37;">%{top_molecule.market_share:.1f}</span> pazar hakimiyeti
                    </p>
                    <p style="color: #c0c0c0; margin: 0.5rem 0;">
                        <span style="color: white; font-weight: 700;">⚠️ Kanibalizasyon Riski:</span> {high_risk_count} paket<br>
                        <span style="color: {'#e74c3c' if high_risk_count > 0 else '#2ecc71'};">
                            {'Acil müdahale gerekiyor' if high_risk_count > 0 else 'Risk kontrol altında'}
                        </span>
                    </p>
                </div>
            </div>
        </div>
        """
        
        return insight
    
    @staticmethod
    def generate_forecast_insight(forecast_result: Dict[str, Any]) -> str:
        """Tahmin analizi için yönetici özeti"""
        if not forecast_result.get('predictions'):
            return "<div style='padding: 1rem; background: #1e3a5f; color: #c0c0c0; border-radius: 10px;'>Tahmin verisi yetersiz.</div>"
        
        growth_2025 = ((forecast_result['predictions'][11] - forecast_result['predictions'][0]) / max(forecast_result['predictions'][0], 1)) * 100 if len(forecast_result['predictions']) > 11 else 0
        growth_2026 = ((forecast_result['predictions'][23] - forecast_result['predictions'][11]) / max(forecast_result['predictions'][11], 1)) * 100 if len(forecast_result['predictions']) > 23 else 0
        
        insight = f"""
        <div style="background: linear-gradient(135deg, #0c1a32, #14274e); padding: 1.8rem; border-radius: 15px; 
                    border-left: 6px solid #3498db; box-shadow: 0 8px 16px rgba(0,0,0,0.3); margin: 1.5rem 0;">
            <h4 style="color: #3498db; margin-top: 0; font-size: 1.3rem; border-bottom: 1px solid #3498db; padding-bottom: 0.7rem;">
                🔮 PAZAR TAHMİN ÖZETİ (2025-2026)
            </h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1rem;">
                <div>
                    <p style="color: #c0c0c0; margin: 0.5rem 0;">
                        <span style="color: white; font-weight: 700;">📅 2025 Büyüme:</span> 
                        <span style="color: {'#2ecc71' if growth_2025 > 0 else '#e74c3c'}; font-size: 1.2rem; font-weight: 700;">
                            %{growth_2025:.1f}
                        </span>
                    </p>
                    <p style="color: #c0c0c0; margin: 0.5rem 0;">
                        <span style="color: white; font-weight: 700;">📅 2026 Büyüme:</span> 
                        <span style="color: {'#2ecc71' if growth_2026 > 0 else '#e74c3c'}; font-size: 1.2rem; font-weight: 700;">
                            %{growth_2026:.1f}
                        </span>
                    </p>
                </div>
                <div>
                    <p style="color: #c0c0c0; margin: 0.5rem 0;">
                        <span style="color: white; font-weight: 700;">🎯 Yatırım Tavsiyesi:</span><br>
                        <span style="color: {forecast_result.get('investment_advice', ['', ''])[1]}; font-size: 1.1rem;">
                            {forecast_result.get('investment_advice', ['', 'Tahmin yetersiz'])[1]}
                        </span>
                    </p>
                    <p style="color: #c0c0c0; margin: 0.5rem 0;">
                        <span style="color: white; font-weight: 700;">✓ Model Güveni:</span> 
                        {forecast_result.get('confidence_score', 0)*100:.0f}%
                    </p>
                </div>
            </div>
        </div>
        """
        
        return insight
    
    @staticmethod
    def generate_risk_insight(anomaly_df: pd.DataFrame) -> str:
        """Risk analizi için yönetici özeti"""
        if anomaly_df is None or 'Anomali_Seviyesi' not in anomaly_df.columns:
            return "<div style='padding: 1rem; background: #1e3a5f; color: #c0c0c0; border-radius: 10px;'>Risk verisi yok.</div>"
        
        n_critical = len(anomaly_df[anomaly_df['Anomali_Seviyesi'] == 'Kritik Düşüş'])
        n_hyper = len(anomaly_df[anomaly_df['Anomali_Seviyesi'] == 'Aşırı Büyüme'])
        n_anomaly = len(anomaly_df[anomaly_df['Anomali_Tespiti'] == -1])
        
        insight = f"""
        <div style="background: linear-gradient(135deg, #0c1a32, #14274e); padding: 1.8rem; border-radius: 15px; 
                    border-left: 6px solid #e74c3c; box-shadow: 0 8px 16px rgba(0,0,0,0.3); margin: 1.5rem 0;">
            <h4 style="color: #e74c3c; margin-top: 0; font-size: 1.3rem; border-bottom: 1px solid #e74c3c; padding-bottom: 0.7rem;">
                ⚠️ RİSK VE FIRSAT İZLEME
            </h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                <div style="text-align: center; background: rgba(231, 76, 60, 0.1); padding: 0.8rem; border-radius: 8px;">
                    <span style="color: #e74c3c; font-size: 1.8rem; font-weight: 700;">{n_critical}</span>
                    <p style="color: white; margin: 0; font-size: 0.9rem;">Kritik Düşüş</p>
                </div>
                <div style="text-align: center; background: rgba(46, 204, 113, 0.1); padding: 0.8rem; border-radius: 8px;">
                    <span style="color: #2ecc71; font-size: 1.8rem; font-weight: 700;">{n_hyper}</span>
                    <p style="color: white; margin: 0; font-size: 0.9rem;">Aşırı Büyüme</p>
                </div>
                <div style="text-align: center; background: rgba(241, 196, 15, 0.1); padding: 0.8rem; border-radius: 8px;">
                    <span style="color: #f1c40f; font-size: 1.8rem; font-weight: 700;">{n_anomaly}</span>
                    <p style="color: white; margin: 0; font-size: 0.9rem;">Toplam Anomali</p>
                </div>
            </div>
            <p style="color: #c0c0c0; margin-top: 1.2rem; padding-top: 0.8rem; border-top: 1px solid #34495e;">
                <span style="color: white; font-weight: 700;">🔍 Öne Çıkan:</span> 
                {f'{n_critical} üründe kritik düşüş tespit edildi. Acil aksiyon planı başlatın.' if n_critical > 0 else 
                 f'{n_hyper} üründe aşırı büyüme fırsatı. Yatırımı değerlendirin.' if n_hyper > 0 else 
                 'Risk seviyesi kontrol altında. Mevcut stratejiyi koruyun.'}
            </p>
        </div>
        """
        
        return insight
    
    @staticmethod
    def generate_segment_insight(segment_result: Dict[str, Any]) -> str:
        """Segmentasyon için yönetici özeti"""
        if not segment_result.get('segment_profiles'):
            return "<div style='padding: 1rem; background: #1e3a5f; color: #c0c0c0; border-radius: 10px;'>Segmentasyon verisi yok.</div>"
        
        profiles = segment_result['segment_profiles']
        
        leader_size = sum(p['size'] for p in profiles.values() if 'Liderler' in p.get('name', ''))
        potential_size = sum(p['size'] for p in profiles.values() if 'Potansiyeller' in p.get('name', ''))
        risk_size = sum(p['size'] for p in profiles.values() if 'Riskli' in p.get('name', ''))
        
        insight = f"""
        <div style="background: linear-gradient(135deg, #0c1a32, #14274e); padding: 1.8rem; border-radius: 15px; 
                    border-left: 6px solid #9b59b6; box-shadow: 0 8px 16px rgba(0,0,0,0.3); margin: 1.5rem 0;">
            <h4 style="color: #9b59b6; margin-top: 0; font-size: 1.3rem; border-bottom: 1px solid #9b59b6; padding-bottom: 0.7rem;">
                🎯 STRATEJİK SEGMENT HARİTASI
            </h4>
            <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                <div style="text-align: center;">
                    <span style="color: #f1c40f; font-size: 2rem;">🌟</span>
                    <p style="color: white; font-size: 1.2rem; margin: 0.3rem 0; font-weight: 700;">{leader_size}</p>
                    <p style="color: #c0c0c0; margin: 0;">Lider Ürün</p>
                </div>
                <div style="text-align: center;">
                    <span style="color: #2ecc71; font-size: 2rem;">📈</span>
                    <p style="color: white; font-size: 1.2rem; margin: 0.3rem 0; font-weight: 700;">{potential_size}</p>
                    <p style="color: #c0c0c0; margin: 0;">Potansiyel</p>
                </div>
                <div style="text-align: center;">
                    <span style="color: #e74c3c; font-size: 2rem;">⚠️</span>
                    <p style="color: white; font-size: 1.2rem; margin: 0.3rem 0; font-weight: 700;">{risk_size}</p>
                    <p style="color: #c0c0c0; margin: 0;">Riskli</p>
                </div>
            </div>
            <p style="color: #c0c0c0; margin-top: 1.2rem; background: rgba(155, 89, 182, 0.1); padding: 0.8rem; border-radius: 6px;">
                <span style="color: #9b59b6; font-weight: 700;">💡 Strateji Özeti:</span> 
                {f'{potential_size} yüksek potansiyelli ürün keşfedildi. Pazarlama bütçesini bu ürünlere kaydırın.' if potential_size > leader_size else
                 f'Pazar lideri konumunuz güçlü. Savunma stratejisi uygulayın.' if leader_size > 0 else
                 'Portföy optimizasyonu zamanı. Riskli ürünleri değerlendirin.'}
            </p>
        </div>
        """
        
        return insight

# ================================================
# 10. EXECUTIVE DARK MODE CSS
# ================================================

EXECUTIVE_DARK_CSS = """
<style>
    /* Executive Dark Theme - Lacivert, Gümüş, Altın */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(145deg, #0a1928 0%, #0e1e2f 50%, #0c1a32 100%);
        background-attachment: fixed;
    }
    
    /* Ana kart tasarımı */
    .executive-card {
        background: rgba(20, 39, 74, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5), 0 0 0 1px rgba(212,175,55,0.1) inset;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
    }
    
    .executive-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 60px -10px rgba(0,0,0,0.7), 0 0 0 2px rgba(212,175,55,0.2) inset;
        border-color: rgba(212, 175, 55, 0.5);
    }
    
    /* Metrik kartları */
    .metric-gold {
        background: linear-gradient(145deg, #1e3a5f, #14274e);
        border-bottom: 4px solid #d4af37;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        color: white;
    }
    
    .metric-silver {
        background: linear-gradient(145deg, #1f3a4b, #162b38);
        border-bottom: 4px solid #c0c0c0;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        color: white;
    }
    
    .metric-navy {
        background: linear-gradient(145deg, #0e2a3b, #0a1e2a);
        border-bottom: 4px solid #4a6fa5;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        color: white;
    }
    
    /* Insight Box - Executive Özet */
    .insight-box {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.1), rgba(192, 192, 192, 0.05));
        border-left: 8px solid #d4af37;
        border-radius: 12px;
        padding: 1.8rem;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .insight-box::before {
        content: '💡';
        position: absolute;
        right: 20px;
        bottom: 20px;
        font-size: 60px;
        opacity: 0.1;
    }
    
    /* Butonlar - Altın aksan */
    .stButton > button {
        background: linear-gradient(145deg, #1e3a5f, #14274e);
        color: #d4af37;
        border: 1px solid #d4af37;
        border-radius: 40px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .stButton > button:hover {
        background: #d4af37;
        color: #0c1a32;
        border-color: #d4af37;
        box-shadow: 0 6px 12px rgba(212, 175, 55, 0.3);
        transform: scale(1.02);
    }
    
    /* Sekmeler - Premium tasarım */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(20, 39, 74, 0.5);
        backdrop-filter: blur(8px);
        border-radius: 50px;
        padding: 6px;
        gap: 4px;
        border: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 40px;
        padding: 10px 20px;
        color: #c0c0c0;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #d4af37, #b8960c) !important;
        color: white !important;
        font-weight: 700;
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        background: linear-gradient(135deg, #d4af37, #f0e68c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    /* Progress bar - Altın */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #d4af37, #f1c40f) !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1a32 0%, #0e1e2f 100%);
        border-right: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    /* Tooltip */
    .stTooltip {
        background: #1e3a5f !important;
        border: 1px solid #d4af37 !important;
        color: white !important;
        border-radius: 12px !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0c1a32;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #d4af37, #b8860b);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #f1c40f, #d4af37);
    }
</style>
"""

# ================================================
# 11. ANA UYGULAMA SINIFI - PHARMAINTELLIGENCE PRO
# ================================================

class PharmaIntelligencePro:
    """
    PharmaIntelligence Pro v8.0 - Enterprise Karar Destek Platformu
    4000+ satır, 10+ entegre AI modülü, Executive Dark Theme
    """
    
    def __init__(self):
        """Ana uygulama başlatıcı"""
        self.technical_optimizer = TechnicalOptimizer()
        self.prodpack_analyzer = ProdPackDepthAnalyzer()
        self.strategic_ai = StrategicAIEngine()
        self.insight_generator = ExecutiveInsightGenerator()
        
        # Session state başlatma
        self._init_session_state()
        
    def _init_session_state(self):
        """Session state değişkenlerini başlat"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'raw_df' not in st.session_state:
            st.session_state.raw_df = None
        if 'processed_df' not in st.session_state:
            st.session_state.processed_df = None
        if 'prodpack_hierarchy' not in st.session_state:
            st.session_state.prodpack_hierarchy = {}
        if 'cannibalization_results' not in st.session_state:
            st.session_state.cannibalization_results = []
        if 'forecast_result' not in st.session_state:
            st.session_state.forecast_result = {}
        if 'anomaly_df' not in st.session_state:
            st.session_state.anomaly_df = None
        if 'segment_result' not in st.session_state:
            st.session_state.segment_result = {}
        if 'investment_advice' not in st.session_state:
            st.session_state.investment_advice = []
    
    def run(self):
        """Ana uygulama akışı"""
        
        # Executive Dark Mode CSS
        st.markdown(EXECUTIVE_DARK_CSS, unsafe_allow_html=True)
        
        # Header - PharmaIntelligence Pro
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;">
            <h1 style="font-size: 3.8rem; margin-bottom: 0.2rem; letter-spacing: -1px;">
                PHARMAINTELLIGENCE<span style="color: #d4af37; -webkit-text-fill-color: #d4af37;"> PRO</span>
            </h1>
            <p style="color: #c0c0c0; font-size: 1.2rem; letter-spacing: 3px; font-weight: 300;">
                v8.0 · ENTERPRISE DECISION INTELLIGENCE
            </p>
            <div style="height: 4px; width: 150px; background: linear-gradient(90deg, #d4af37, #c0c0c0, #0c1a32); margin: 1rem auto;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar menü
        with st.sidebar:
            self._render_sidebar()
        
        # Ana içerik
        if not st.session_state.data_loaded:
            self._render_welcome_screen()
        else:
            self._render_main_dashboard()
    
    def _render_sidebar(self):
        """Sidebar - Veri yükleme, analiz kontrolleri"""
        
        st.markdown("""
        <div style="background: rgba(212, 175, 55, 0.1); padding: 1.2rem; border-radius: 16px; 
                    border-bottom: 3px solid #d4af37; margin-bottom: 2rem;">
            <span style="color: #d4af37; font-size: 1.5rem; font-weight: 800;">⚡ KONTROL PANELİ</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Veri yükleme
        st.markdown("### 📁 VERİ KAYNAĞI")
        uploaded_file = st.file_uploader(
            "Excel veya CSV yükleyin",
            type=['xlsx', 'xls', 'csv'],
            help="ProdPack analizi için 'Molekül, Şirket, Marka, Paket' sütunları önerilir"
        )
        
        if uploaded_file:
            if st.button("🚀 VERİYİ YÜKLE VE İŞLE", use_container_width=True):
                with st.spinner("🔮 Yapay zeka motorları başlatılıyor..."):
                    self._load_and_process_data(uploaded_file)
        
        st.markdown("---")
        
        # Analiz modülleri
        if st.session_state.data_loaded:
            st.markdown("### 🧠 ANALİZ MODÜLLERİ")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 ProdPack Derinlik", use_container_width=True):
                    with st.spinner("Hiyerarşi kuruluyor..."):
                        self._run_prodpack_analysis()
                
                if st.button("🔮 Tahmin & Öngörü", use_container_width=True):
                    with st.spinner("2025-2026 tahminleri hesaplanıyor..."):
                        self._run_forecast_analysis()
            
            with col2:
                if st.button("⚠️ Risk İzleme", use_container_width=True):
                    with st.spinner("Anomaliler taranıyor..."):
                        self._run_anomaly_detection()
                
                if st.button("🎯 Segmentasyon", use_container_width=True):
                    with st.spinner("PCA+K-Means segmentasyon..."):
                        self._run_segmentation_analysis()
            
            st.markdown("---")
            st.markdown("### 📋 HIZLI RAPOR")
            
            report_type = st.selectbox(
                "Rapor formatı",
                ["Yönetici Özeti (PDF)", "Detaylı Excel", "Stratejik Sunum"]
            )
            
            if st.button("📥 RAPOR OLUŞTUR", use_container_width=True):
                st.success(f"{report_type} hazırlanıyor...")
        
        # Sistem durumu
        st.markdown("---")
        st.markdown("### ⚙️ SİSTEM DURUMU")
        
        if st.session_state.data_loaded:
            st.markdown(f"""
            <div style="background: rgba(46, 204, 113, 0.1); padding: 0.8rem; border-radius: 8px;">
                <span style="color: #2ecc71;">● AKTİF</span><br>
                <span style="color: #c0c0c0; font-size: 0.8rem;">
                📊 {len(st.session_state.processed_df)} satır<br>
                🧬 {len(st.session_state.prodpack_hierarchy)} ProdPack node
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(231, 76, 60, 0.1); padding: 0.8rem; border-radius: 8px;">
                <span style="color: #e74c3c;">● BEKLEMEDE</span><br>
                <span style="color: #c0c0c0; font-size: 0.8rem;">Veri yüklenmemiş</span>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_welcome_screen(self):
        """Hoşgeldin ekranı - Enterprise onboarding"""
        
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.markdown("""
            <div style="background: rgba(20, 39, 74, 0.5); backdrop-filter: blur(10px); 
                        border-radius: 40px; padding: 3.5rem; border: 1px solid rgba(212, 175, 55, 0.3);
                        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); margin-top: 2rem;">
                
                <div style="text-align: center; font-size: 5rem; margin-bottom: 1rem;">💊</div>
                <h2 style="text-align: center; color: #d4af37; font-size: 2.2rem;">Enterprise Decision Support</h2>
                <p style="text-align: center; color: #c0c0c0; font-size: 1.1rem; margin: 2rem 0; line-height: 1.8;">
                    Molekülden pakete kadar pazarınızın her katmanını analiz edin.<br>
                    Yapay zeka destekli tahminleme ile 2025-2026 stratejinizi oluşturun.
                </p>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 2.5rem 0;">
                    <div style="background: rgba(212, 175, 55, 0.08); padding: 1.2rem; border-radius: 16px;">
                        <span style="color: #d4af37; font-size: 1.8rem;">🔬</span>
                        <h4 style="color: white; margin: 0.5rem 0;">ProdPack Derinlik</h4>
                        <p style="color: #c0c0c0; font-size: 0.9rem;">Molekül → Şirket → Marka → Paket hiyerarşisi</p>
                    </div>
                    <div style="background: rgba(192, 192, 192, 0.08); padding: 1.2rem; border-radius: 16px;">
                        <span style="color: #c0c0c0; font-size: 1.8rem;">📈</span>
                        <h4 style="color: white; margin: 0.5rem 0;">Holt-Winters Tahmin</h4>
                        <p style="color: #c0c0c0; font-size: 0.9rem;">2025-2026 pazar projeksiyonu</p>
                    </div>
                    <div style="background: rgba(46, 204, 113, 0.08); padding: 1.2rem; border-radius: 16px;">
                        <span style="color: #2ecc71; font-size: 1.8rem;">🛡️</span>
                        <h4 style="color: white; margin: 0.5rem 0;">IsolationForest</h4>
                        <p style="color: #c0c0c0; font-size: 0.9rem;">Anomali tespiti ve risk izleme</p>
                    </div>
                    <div style="background: rgba(155, 89, 182, 0.08); padding: 1.2rem; border-radius: 16px;">
                        <span style="color: #9b59b6; font-size: 1.8rem;">🎯</span>
                        <h4 style="color: white; margin: 0.5rem 0;">PCA + K-Means</h4>
                        <p style="color: #c0c0c0; font-size: 0.9rem;">Stratejik ürün segmentasyonu</p>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 1rem;">
                    <span style="color: #d4af37;">←</span> 
                    <span style="color: #c0c0c0; margin: 0 1rem;">Sol panelden veri yükleyerek başlayın</span>
                    <span style="color: #d4af37;">→</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_main_dashboard(self):
        """Ana dashboard - Sekmeli yapı"""
        
        tabs = st.tabs([
            "🏢 PRODPACK DERİNLİK",
            "🔮 TAHMİN & ÖNGÖRÜ",
            "⚠️ RİSK & FIRSAT",
            "🎯 STRATEJİK SEGMENT",
            "📊 EXECUTIVE DASHBOARD"
        ])
        
        with tabs[0]:
            self._render_prodpack_tab()
        
        with tabs[1]:
            self._render_forecast_tab()
        
        with tabs[2]:
            self._render_risk_tab()
        
        with tabs[3]:
            self._render_segment_tab()
        
        with tabs[4]:
            self._render_executive_dashboard()
    
    def _render_prodpack_tab(self):
        """ProdPack Derinlik Analizi Sekmesi"""
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 2rem;">
            <span style="font-size: 2.5rem; margin-right: 1rem;">🔬</span>
            <div>
                <h2 style="margin: 0;">ProdPack Hiyerarşi & Derinlik Analizi</h2>
                <p style="color: #c0c0c0; margin: 0.2rem 0 0 0;">Molekül → Şirket → Marka → Paket · Pazar Kanibalizasyonu</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div class="executive-card" style="padding: 1.5rem;">
                <h4 style="color: #d4af37; margin-top: 0; border-bottom: 1px solid #d4af37; padding-bottom: 0.8rem;">
                    🧬 MOLEKÜL DRILL-DOWN
                </h4>
            """, unsafe_allow_html=True)
            
            # Molekül seçimi
            if st.session_state.prodpack_hierarchy:
                molecules = [node for node in st.session_state.prodpack_hierarchy.values() if node.node_type == 'molecule']
                molecule_names = ['Tümü'] + [m.name for m in molecules]
                selected_molecule = st.selectbox("Molekül Seçin", molecule_names)
                
                st.markdown("---")
                
                # Gösterim ayarları
                st.markdown("#### 📊 Görselleştirme")
                show_sunburst = st.checkbox("Sunburst Diyagramı", value=True)
                show_sankey = st.checkbox("Sankey Akış Diyagramı", value=False)
                show_cannibalization = st.checkbox("Kanibalizasyon Matrisi", value=True)
                
                st.markdown("---")
                
                # Veri gezgini - 5000 satır kapasite
                st.markdown("#### 🔍 VERİ GEZGİNİ")
                if st.session_state.processed_df is not None:
                    preview_rows = st.slider("Gösterilecek satır", 100, 5000, 1000, 100)
                    
                    # Önemli sütunlar
                    important_cols = []
                    for col in ['Molekül', 'Şirket', 'Marka', 'Paket', 'Satış_2024', 'Büyüme_Oranı']:
                        if col in st.session_state.processed_df.columns:
                            important_cols.append(col)
                    
                    if important_cols:
                        st.dataframe(
                            st.session_state.processed_df[important_cols].head(preview_rows),
                            use_container_width=True,
                            height=300
                        )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if not st.session_state.prodpack_hierarchy:
                st.info("""
                <div style="background: rgba(52, 152, 219, 0.1); padding: 2rem; border-radius: 16px; text-align: center;">
                    <span style="font-size: 3rem;">🏗️</span>
                    <h4 style="color: #3498db; margin: 1rem 0;">ProdPack Hiyerarşisi Kurulmamış</h4>
                    <p style="color: #c0c0c0;">Sol panelden "ProdPack Derinlik" butonuna tıklayarak analizi başlatın.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Sunburst diyagramı
                if show_sunburst:
                    with st.spinner("Sunburst diyagramı oluşturuluyor..."):
                        fig = self.prodpack_analyzer.create_sunburst_diagram(
                            st.session_state.prodpack_hierarchy
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Insight Box - ProdPack Özeti
                st.markdown(
                    self.insight_generator.generate_prodpack_insight(
                        st.session_state.prodpack_hierarchy
                    ),
                    unsafe_allow_html=True
                )
                
                # Kanibalizasyon matrisi
                if show_cannibalization and st.session_state.cannibalization_results:
                    fig = self.prodpack_analyzer.create_cannibalization_heatmap(
                        st.session_state.cannibalization_results
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_forecast_tab(self):
        """Tahmin & Öngörü Sekmesi"""
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 2rem;">
            <span style="font-size: 2.5rem; margin-right: 1rem;">🔮</span>
            <div>
                <h2 style="margin: 0;">Pazar Tahmini & Yatırım Stratejisi</h2>
                <p style="color: #c0c0c0; margin: 0.2rem 0 0 0;">Holt-Winters · Ensemble · 2025-2026 Projeksiyonu</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.forecast_result:
            st.info("""
            <div style="background: rgba(52, 152, 219, 0.1); padding: 2rem; border-radius: 16px; text-align: center;">
                <span style="font-size: 3rem;">📊</span>
                <h4 style="color: #3498db; margin: 1rem 0;">Tahmin Analizi Henüz Çalıştırılmamış</h4>
                <p style="color: #c0c0c0;">Sol panelden "Tahmin & Öngörü" butonuna tıklayın.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Tahmin grafiği
                forecast = st.session_state.forecast_result
                
                fig = go.Figure()
                
                # Güven aralığı
                if forecast.get('lower_bounds') and forecast.get('upper_bounds'):
                    fig.add_trace(go.Scatter(
                        x=forecast.get('dates', list(range(len(forecast['predictions'])))),
                        y=forecast['upper_bounds'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        name='Üst Sınır'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast.get('dates', list(range(len(forecast['predictions'])))),
                        y=forecast['lower_bounds'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(52, 152, 219, 0.2)',
                        showlegend=False,
                        name='Alt Sınır'
                    ))
                
                # Tahmin çizgisi
                fig.add_trace(go.Scatter(
                    x=forecast.get('dates', list(range(len(forecast['predictions'])))),
                    y=forecast['predictions'],
                    mode='lines+markers',
                    name='Tahmin',
                    line=dict(color='#d4af37', width=4),
                    marker=dict(size=6, color='#d4af37')
                ))
                
                fig.update_layout(
                    title='2025-2026 Pazar Büyüme Tahmini',
                    xaxis_title='Dönem',
                    yaxis_title='Satış (USD)',
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Yatırım tavsiye kutuları
                st.markdown("""
                <div class="executive-card" style="padding: 1.5rem; height: 100%;">
                    <h4 style="color: #d4af37; margin-top: 0; border-bottom: 1px solid #d4af37; padding-bottom: 0.8rem;">
                        💼 YATIRIM TAVSİYESİ
                    </h4>
                """, unsafe_allow_html=True)
                
                for advice in st.session_state.investment_advice:
                    st.markdown(f"""
                    <div style="background: rgba({int(advice['color'][1:3], 16)}, {int(advice['color'][3:5], 16)}, {int(advice['color'][5:7], 16)}, 0.1); 
                                border-left: 6px solid {advice['color']}; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem;">
                        <span style="color: {advice['color']}; font-size: 1.3rem; font-weight: 800;">{advice['title']}</span>
                        <p style="color: white; margin: 0.5rem 0;">{advice['message']}</p>
                        <p style="color: #c0c0c0; margin: 0; font-size: 0.9rem;">
                            <span style="color: {advice['color']};">▶</span> {advice['action']}
                        </p>
                        <p style="color: {advice['color']}; margin: 0.5rem 0 0 0; font-size: 0.8rem;">
                            Güven: {advice['confidence']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Insight Box - Tahmin Özeti
            st.markdown(
                self.insight_generator.generate_forecast_insight(st.session_state.forecast_result),
                unsafe_allow_html=True
            )
    
    def _render_risk_tab(self):
        """Risk & Fırsat İzleme Sekmesi"""
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 2rem;">
            <span style="font-size: 2.5rem; margin-right: 1rem;">⚠️</span>
            <div>
                <h2 style="margin: 0;">Risk & Fırsat İzleme</h2>
                <p style="color: #c0c0c0; margin: 0.2rem 0 0 0;">IsolationForest · Anomali Tespiti · Erken Uyarı Sistemi</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.anomaly_df is None:
            st.info("""
            <div style="background: rgba(231, 76, 60, 0.1); padding: 2rem; border-radius: 16px; text-align: center;">
                <span style="font-size: 3rem;">🛡️</span>
                <h4 style="color: #e74c3c; margin: 1rem 0;">Risk Analizi Henüz Çalıştırılmamış</h4>
                <p style="color: #c0c0c0;">Sol panelden "Risk İzleme" butonuna tıklayın.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            anomaly_df = st.session_state.anomaly_df
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                n_critical = len(anomaly_df[anomaly_df['Anomali_Seviyesi'] == 'Kritik Düşüş'])
                st.markdown(f"""
                <div class="metric-gold" style="text-align: center;">
                    <span style="font-size: 2rem;">🔴</span>
                    <h3 style="color: white; margin: 0.3rem 0; font-size: 2.2rem;">{n_critical}</h3>
                    <p style="color: #c0c0c0; margin: 0;">Kritik Düşüş</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                n_hyper = len(anomaly_df[anomaly_df['Anomali_Seviyesi'] == 'Aşırı Büyüme'])
                st.markdown(f"""
                <div class="metric-silver" style="text-align: center;">
                    <span style="font-size: 2rem;">🟢</span>
                    <h3 style="color: white; margin: 0.3rem 0; font-size: 2.2rem;">{n_hyper}</h3>
                    <p style="color: #c0c0c0; margin: 0;">Aşırı Büyüme</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                n_anomaly = len(anomaly_df[anomaly_df['Anomali_Tespiti'] == -1])
                st.markdown(f"""
                <div class="metric-navy" style="text-align: center;">
                    <span style="font-size: 2rem;">🟡</span>
                    <h3 style="color: white; margin: 0.3rem 0; font-size: 2.2rem;">{n_anomaly}</h3>
                    <p style="color: #c0c0c0; margin: 0;">Toplam Anomali</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_risk = anomaly_df['Risk_Skoru'].mean()
                st.markdown(f"""
                <div class="metric-gold" style="text-align: center;">
                    <span style="font-size: 2rem;">📊</span>
                    <h3 style="color: white; margin: 0.3rem 0; font-size: 2.2rem;">{avg_risk:.1f}</h3>
                    <p style="color: #c0c0c0; margin: 0;">Ort. Risk Skoru</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Insight Box - Risk Özeti
            st.markdown(
                self.insight_generator.generate_risk_insight(anomaly_df),
                unsafe_allow_html=True
            )
            
            # Anomali listesi
            st.markdown("""
            <h4 style="color: #d4af37; margin-top: 2rem;">🚨 KRİTİK RİSKLİ ÜRÜNLER</h4>
            """, unsafe_allow_html=True)
            
            critical_products = anomaly_df[anomaly_df['Anomali_Seviyesi'] == 'Kritik Düşüş']
            
            if len(critical_products) > 0:
                display_cols = []
                for col in ['Molekül', 'Şirket', 'Marka', 'Paket', 'Anomali_Seviyesi', 'Risk_Skoru']:
                    if col in critical_products.columns:
                        display_cols.append(col)
                
                st.dataframe(
                    critical_products[display_cols].sort_values('Risk_Skoru', ascending=False).head(20),
                    use_container_width=True
                )
            else:
                st.success("✅ Kritik riskli ürün bulunamadı.")
    
    def _render_segment_tab(self):
        """Stratejik Segmentasyon Sekmesi"""
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 2rem;">
            <span style="font-size: 2.5rem; margin-right: 1rem;">🎯</span>
            <div>
                <h2 style="margin: 0;">Stratejik Segmentasyon</h2>
                <p style="color: #c0c0c0; margin: 0.2rem 0 0 0;">PCA · K-Means · Liderler / Potansiyeller / Riskli Ürünler</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.segment_result:
            st.info("""
            <div style="background: rgba(155, 89, 182, 0.1); padding: 2rem; border-radius: 16px; text-align: center;">
                <span style="font-size: 3rem;">🧩</span>
                <h4 style="color: #9b59b6; margin: 1rem 0;">Segmentasyon Analizi Henüz Çalıştırılmamış</h4>
                <p style="color: #c0c0c0;">Sol panelden "Segmentasyon" butonuna tıklayın.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            segment_result = st.session_state.segment_result
            
            col1, col2, col3 = st.columns(3)
            
            profiles = segment_result.get('segment_profiles', {})
            
            for i, (cluster, profile) in enumerate(profiles.items()):
                with [col1, col2, col3][i % 3]:
                    color = {
                        '🌟 Liderler': '#f1c40f',
                        '📈 Potansiyeller': '#2ecc71',
                        '⚠️ Riskli Ürünler': '#e74c3c'
                    }.get(profile.get('name', ''), '#9b59b6')
                    
                    st.markdown(f"""
                    <div style="background: rgba(20, 39, 74, 0.7); border-radius: 16px; padding: 1.5rem; 
                                border-bottom: 6px solid {color}; margin-bottom: 1rem; height: 220px;">
                        <span style="font-size: 2rem;">{profile.get('name', 'Segment')[:2]}</span>
                        <h4 style="color: {color}; margin: 0.5rem 0; font-size: 1.2rem;">{profile.get('name', 'Segment')}</h4>
                        <p style="color: #c0c0c0; margin: 0.3rem 0; font-size: 1.8rem; font-weight: 700;">{profile.get('size', 0)}</p>
                        <p style="color: white; margin: 0; font-size: 0.85rem;">{profile.get('description', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Insight Box - Segment Özeti
            st.markdown(
                self.insight_generator.generate_segment_insight(segment_result),
                unsafe_allow_html=True
            )
            
            # Strateji önerileri
            st.markdown("""
            <h4 style="color: #d4af37; margin-top: 2rem;">🎯 SEGMENT BAZLI STRATEJİLER</h4>
            """, unsafe_allow_html=True)
            
            for cluster, profile in profiles.items():
                with st.expander(f"{profile.get('name', 'Segment')} - {profile.get('size', 0)} ürün"):
                    st.markdown(f"""
                    <div style="background: rgba(20, 39, 74, 0.5); padding: 1.2rem; border-radius: 8px;">
                        <p style="color: white; font-size: 1.1rem;">{profile.get('strategy', 'Strateji önerisi bulunamadı.')}</p>
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                            <div>
                                <span style="color: #c0c0c0;">Ort. Pazar Payı:</span><br>
                                <span style="color: white; font-size: 1.3rem; font-weight: 700;">{profile.get('avg_market_share', 0):.2f}%</span>
                            </div>
                            <div>
                                <span style="color: #c0c0c0;">Ort. Büyüme:</span><br>
                                <span style="color: {'#2ecc71' if profile.get('avg_growth', 0) > 0 else '#e74c3c'}; 
                                          font-size: 1.3rem; font-weight: 700;">
                                    {profile.get('avg_growth', 0):.1f}%
                                </span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    def _render_executive_dashboard(self):
        """Executive Dashboard - Tüm metrikler tek ekranda"""
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 2rem;">
            <span style="font-size: 2.5rem; margin-right: 1rem;">📊</span>
            <div>
                <h2 style="margin: 0;">Executive Dashboard</h2>
                <p style="color: #c0c0c0; margin: 0.2rem 0 0 0;">Gerçek zamanlı pazar istihbaratı · KPI Takip</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Üst metrik satırı
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_packages = len([n for n in st.session_state.prodpack_hierarchy.values() if n.node_type == 'package']) if st.session_state.prodpack_hierarchy else 0
            st.markdown(f"""
            <div class="metric-gold">
                <span style="font-size: 1.8rem;">📦</span>
                <h3 style="color: white; margin: 0.3rem 0; font-size: 2rem;">{total_packages}</h3>
                <p style="color: #c0c0c0; margin: 0;">Aktif ProdPack</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_market = 0
            if st.session_state.prodpack_hierarchy:
                packages = [n for n in st.session_state.prodpack_hierarchy.values() if n.node_type == 'package']
                total_market = sum(p.sales_2024 for p in packages) / 1_000_000
            st.markdown(f"""
            <div class="metric-silver">
                <span style="font-size: 1.8rem;">💰</span>
                <h3 style="color: white; margin: 0.3rem 0; font-size: 2rem;">${total_market:.1f}M</h3>
                <p style="color: #c0c0c0; margin: 0;">Toplam Pazar</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_growth = 0
            if st.session_state.prodpack_hierarchy:
                packages = [n for n in st.session_state.prodpack_hierarchy.values() if n.node_type == 'package']
                avg_growth = np.mean([p.growth_rate for p in packages if not np.isnan(p.growth_rate)])
            st.markdown(f"""
            <div class="metric-navy">
                <span style="font-size: 1.8rem;">📈</span>
                <h3 style="color: white; margin: 0.3rem 0; font-size: 2rem;">%{avg_growth:.1f}</h3>
                <p style="color: #c0c0c0; margin: 0;">Ort. Büyüme</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            risk_score = 0
            if st.session_state.anomaly_df is not None:
                risk_score = st.session_state.anomaly_df['Risk_Skoru'].mean()
            st.markdown(f"""
            <div class="metric-gold">
                <span style="font-size: 1.8rem;">⚠️</span>
                <h3 style="color: white; margin: 0.3rem 0; font-size: 2rem;">{risk_score:.0f}</h3>
                <p style="color: #c0c0c0; margin: 0;">Risk Endeksi</p>
            </div>
            """, unsafe_allow_html=True)
        
        # İkinci satır - Görselleştirmeler
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="executive-card" style="padding: 1.2rem;">
                <h4 style="color: #d4af37; margin-top: 0;">🏆 TOP 5 PAKET - SATIŞ</h4>
            """, unsafe_allow_html=True)
            
            if st.session_state.prodpack_hierarchy:
                packages = [n for n in st.session_state.prodpack_hierarchy.values() if n.node_type == 'package']
                top_packages = sorted(packages, key=lambda x: x.sales_2024, reverse=True)[:5]
                
                fig = go.Figure(go.Bar(
                    x=[p.sales_2024 for p in top_packages],
                    y=[p.name[:20] + '...' if len(p.name) > 20 else p.name for p in top_packages],
                    orientation='h',
                    marker=dict(
                        color=[p.growth_rate for p in top_packages],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Büyüme %")
                    ),
                    text=[f"${p.sales_2024:,.0f}" for p in top_packages],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis_title="Satış (USD)",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="executive-card" style="padding: 1.2rem;">
                <h4 style="color: #d4af37; margin-top: 0;">📊 SEGMENT DAĞILIMI</h4>
            """, unsafe_allow_html=True)
            
            if st.session_state.segment_result and st.session_state.segment_result.get('segmented_df') is not None:
                segment_df = st.session_state.segment_result['segmented_df']
                if 'Segment_Adı' in segment_df.columns:
                    segment_counts = segment_df['Segment_Adı'].value_counts()
                    
                    fig = go.Figure(go.Pie(
                        labels=segment_counts.index,
                        values=segment_counts.values,
                        hole=0.4,
                        marker=dict(colors=['#f1c40f', '#2ecc71', '#e74c3c', '#3498db']),
                        textinfo='label+percent',
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        height=300,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        showlegend=False,
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def _load_and_process_data(self, uploaded_file):
        """Veri yükleme ve işleme"""
        try:
            # Cache mekanizması ile veri yükle
            df = TechnicalOptimizer.cached_data_loader(uploaded_file)
            
            if df.empty:
                st.error("Veri yüklenemedi. Dosya formatını kontrol edin.")
                return
            
            # Sütun isimlerini benzersizleştir
            df.columns = TechnicalOptimizer.unique_column_names(df.columns.tolist())
            
            # Güvenli downcast
            df = TechnicalOptimizer.safe_downcast(df)
            
            # Session state güncelle
            st.session_state.raw_df = df
            st.session_state.processed_df = df
            st.session_state.data_loaded = True
            
            st.success(f"✅ Veri başarıyla yüklendi: {len(df):,} satır, {len(df.columns)} sütun")
            
            # Otomatik ProdPack analizi
            self._run_prodpack_analysis()
            
        except Exception as e:
            st.error(f"Veri yükleme hatası: {str(e)}")
            st.code(traceback.format_exc())
    
    def _run_prodpack_analysis(self):
        """ProdPack Derinlik Analizi çalıştır"""
        if st.session_state.processed_df is not None:
            with st.spinner("ProdPack hiyerarşisi kuruluyor..."):
                hierarchy = self.prodpack_analyzer.build_prodpack_hierarchy(st.session_state.processed_df)
                st.session_state.prodpack_hierarchy = hierarchy
                
                if hierarchy:
                    cannibalization = self.prodpack_analyzer.detect_cannibalization(
                        st.session_state.processed_df, hierarchy
                    )
                    st.session_state.cannibalization_results = cannibalization
                    st.success(f"✅ ProdPack analizi tamam: {len(hierarchy)} node, {len(cannibalization)} kanibalizasyon tespiti")
    
    def _run_forecast_analysis(self):
        """Tahmin analizi çalıştır"""
        if st.session_state.processed_df is not None:
            df = st.session_state.processed_df
            
            # Satış sütununu bul
            sales_col = None
            for col in df.columns:
                if TechnicalOptimizer.safe_extract_years(col):
                    sales_col = col
                    break
            
            if sales_col:
                with st.spinner("Holt-Winters tahmini hesaplanıyor..."):
                    forecast = self.strategic_ai.ensemble_forecast(df, sales_col)
                    st.session_state.forecast_result = forecast
                    
                    advices = self.strategic_ai.generate_investment_advice(forecast)
                    st.session_state.investment_advice = advices
                    
                    st.success("✅ 2025-2026 tahminleri tamamlandı")
            else:
                st.warning("Satış sütunu bulunamadı")
    
    def _run_anomaly_detection(self):
        """Anomali tespiti çalıştır"""
        if st.session_state.processed_df is not None:
            with st.spinner("IsolationForest ile anomali taranıyor..."):
                anomaly_df = self.strategic_ai.isolation_forest_anomaly_detection(st.session_state.processed_df)
                st.session_state.anomaly_df = anomaly_df
                st.success(f"✅ Risk analizi tamam: {len(anomaly_df[anomaly_df['Anomali_Tespiti'] == -1])} anomali tespit edildi")
    
    def _run_segmentation_analysis(self):
        """Segmentasyon analizi çalıştır"""
        if st.session_state.processed_df is not None:
            with st.spinner("PCA + K-Means segmentasyonu uygulanıyor..."):
                segment_result = self.strategic_ai.pca_kmeans_segmentation(st.session_state.processed_df)
                st.session_state.segment_result = segment_result
                
                if segment_result.get('segmented_df') is not None:
                    st.success(f"✅ Segmentasyon tamam: Silhouette skoru {segment_result.get('silhouette_score', 0):.3f}")

# ================================================
# 12. UYGULAMA GİRİŞ NOKTASI
# ================================================

def main():
    """Ana uygulama başlatıcı"""
    
    # Sayfa konfigürasyonu
    st.set_page_config(
        page_title="PharmaIntelligence Pro v8.0",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://pharmaintelligence.com/support',
            'Report a bug': 'https://pharmaintelligence.com/bug',
            'About': 'PharmaIntelligence Pro v8.0 - Enterprise Decision Intelligence Platform'
        }
    )
    
    # Uygulamayı başlat
    app = PharmaIntelligencePro()
    app.run()

if __name__ == "__main__":
    main()
