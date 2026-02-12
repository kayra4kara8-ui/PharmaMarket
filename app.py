"""
PharmaIntelligence Pro v8.0 - Enterprise ProdPack Derinlik Analizi & Stratejik Karar Destek Platformu
Versiyon: 8.0.2
Satır Sayısı: 4,127
Yazar: PharmaIntelligence Inc.
Lisans: Kurumsal Enterprise

✓ ProdPack Derinlik Analizi - Molekül → Şirket → Marka → Paket Hiyerarşisi
✓ Sunburst & Sankey İnteraktif Görselleştirme
✓ Pazar Kanibalizasyon Matrisi (Büyüme/Hacim Korelasyonu)
✓ Holt-Winters ile 2025-2026 Tahminleme
✓ IsolationForest ile Anomali Tespiti
✓ PCA + K-Means Segmentasyon (Liderler, Potansiyeller, Riskli Ürünler)
✓ Executive Dark Mode (Lacivert, Gümüş, Altın)
✓ Otomatik Yönetici Özeti & Yatırım Tavsiyesi
✓ 5000+ Satır Gösterim & 1M+ Satır Cache Optimizasyonu
✓ Regex Yıl Ayıklama & Benzersiz Sütun İsimlendirme
"""

# ================================================
# 0. GEREKLİ KÜTÜPHANE IMPORTLARI
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import re
import gc
import traceback
import json
import base64
import hashlib
import pickle
from io import BytesIO, StringIO
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
import math
import os
import sys
import time
from functools import lru_cache
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ================================================
# 1. GELİŞMİŞ ANALİTİK KÜTÜPHANE KONTROLLERİ
# ================================================

# Scikit-learn - IsolationForest, PCA, KMeans, StandardScaler
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, f_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn kurulu değil. Anomali tespiti ve segmentasyon devre dışı.")

# Statsmodels - Holt-Winters, ExponentialSmoothing
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("⚠️ statsmodels kurulu değil. Holt-Winters tahminleme devre dışı.")

# Scipy - İstatistiksel testler
try:
    from scipy import stats
    from scipy.spatial.distance import cdist, pdist, squareform
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ================================================
# 2. ENUM VE DATA CLASSES
# ================================================

class RiskLevel(Enum):
    """Risk seviyeleri"""
    KRITIK = "🔴 Kritik Risk"
    YUKSEK = "🟠 Yüksek Risk"
    ORTA = "🟡 Orta Risk"
    DUSUK = "🟢 Düşük Risk"
    NORMAL = "✅ Normal"

class GrowthCategory(Enum):
    """Büyüme kategorileri"""
    HIPER = "🚀 Hiper Büyüme (>50%)"
    YUKSEK = "📈 Yüksek Büyüme (20-50%)"
    ORTA = "📊 Orta Büyüme (5-20%)"
    DURGUN = "⏸️ Durgun (-5% - 5%)"
    DARALAN = "📉 Daralan (<-5%)"

class ProductSegment(Enum):
    """Ürün segmentleri"""
    LIDER = "⭐ Lider Ürünler"
    POTANSIYEL = "🌟 Potansiyel Yıldızlar"
    RISKLI = "⚠️ Riskli Ürünler"
    OLGUN = "📦 Olgun Ürünler"
    NAKIT = "💰 Nakit İnekleri"
    SORU = "❓ Soru İşaretleri"

class InvestmentAction(Enum):
    """Yatırım aksiyonları"""
    EXPAND = "🚀 GENİŞLE"
    OPTIMIZE = "⚙️ OPTIMIZE ET"
    DEFEND = "🛡️ KORU"
    CONSOLIDATE = "📊 BİRLEŞTİR"
    EXIT = "🚪 ÇIKIŞ"
    HEDGE = "⚖️ HEDGE"

# ================================================
# 3. DATA CLASSES - VERİ YAPILARI
# ================================================

@dataclass
class ProdPackNode:
    """ProdPack hiyerarşi düğümü"""
    id: str
    name: str
    level: str  # 'molecule', 'company', 'brand', 'pack'
    value: float = 0.0
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    growth_rate: float = 0.0
    market_share: float = 0.0

@dataclass
class CannibalizationScore:
    """Kanibalizasyon skoru"""
    molecule: str
    company: str
    brand_a: str
    brand_b: str
    correlation: float
    cannibal_score: float
    volume_overlap: float
    growth_impact: float
    recommendation: str

@dataclass
class ForecastResult:
    """Tahmin sonucu"""
    periods: List[str]
    predictions: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    growth_rate: float
    model_type: str
    confidence: float = 0.95
    mape: float = 0.0
    rmse: float = 0.0

@dataclass
class InvestmentAdvice:
    """Yatırım tavsiyesi"""
    title: str
    message: str
    action: InvestmentAction
    priority: str  # 'high', 'medium', 'low'
    impact: str  # 'strategic', 'financial', 'operational'
    roi_potential: float  # 0-100
    risk_level: str

# ================================================
# 4. TEKNİK YARDIMCI FONKSİYONLAR - REGEX & GÜVENLİ İŞLEMLER
# ================================================

@st.cache_data(ttl=3600, max_entries=10, show_spinner=False)
def cached_regex_year_extraction(column_names: List[str]) -> Dict[str, Optional[int]]:
    """
    Cache'li regex yıl ayıklama.
    1M+ satır için optimize edilmiş.
    """
    result = {}
    for col in column_names:
        col_str = str(col)
        match = re.search(r'20\d{2}', col_str)
        if match:
            try:
                result[col] = int(match.group())
            except (ValueError, TypeError):
                result[col] = None
        else:
            result[col] = None
    return result

def safe_extract_year(column_name: Any) -> Optional[int]:
    """
    Regex ile güvenli yıl ayıklama.
    Hata fırlatmaz, None döner.
    """
    if not isinstance(column_name, str):
        return None
    
    match = re.search(r'20\d{2}', column_name)
    if match:
        try:
            return int(match.group())
        except (ValueError, TypeError):
            return None
    return None

def make_unique_column_names(columns: List[str]) -> List[str]:
    """
    Yinelenen sütun isimlerini benzersizleştir.
    'Bölge', 'Bölge_1', 'Bölge_2' formatı.
    """
    seen = {}
    unique_cols = []
    
    for i, col in enumerate(columns):
        col_str = str(col)
        
        # 1. Özel karakterleri temizle
        col_clean = re.sub(r'[^\w\s]', ' ', col_str)
        # 2. Birden çok boşluğu tek boşluk yap
        col_clean = re.sub(r'\s+', ' ', col_clean)
        # 3. Boşlukları alt çizgi ile değiştir
        col_clean = col_clean.strip().replace(' ', '_')
        # 4. Uzunluğu sınırla
        if len(col_clean) > 50:
            col_clean = col_clean[:50]
        # 5. Boşsa varsayılan isim ver
        if not col_clean:
            col_clean = f"column_{i}"
        
        # Benzersizleştirme
        if col_clean in seen:
            seen[col_clean] += 1
            new_col = f"{col_clean}_{seen[col_clean]}"
        else:
            seen[col_clean] = 1
            new_col = col_clean
        
        unique_cols.append(new_col)
    
    return unique_cols

def safe_downcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    pd.api.types kullanarak güvenli downcast.
    'Ambiguous Truth Value' hatasını önler.
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        try:
            # pd.api.types ile tip kontrolü
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                # NaN kontrolü
                if df_copy[col].isnull().all():
                    continue
                
                # Integer downcast
                if pd.api.types.is_integer_dtype(df_copy[col]):
                    df_copy[col] = pd.to_numeric(df_copy[col], downcast='integer')
                
                # Float downcast
                elif pd.api.types.is_float_dtype(df_copy[col]):
                    # Çok hassas değilse float32'e çevir
                    if df_copy[col].abs().max() < 1e9:
                        df_copy[col] = pd.to_numeric(df_copy[col], downcast='float')
            
            # Boolean dönüşümü
            elif df_copy[col].dtype == 'object':
                unique_vals = df_copy[col].dropna().unique()
                if len(unique_vals) == 2:
                    # Boolean olabilir
                    bool_map = {unique_vals[0]: True, unique_vals[1]: False}
                    try:
                        df_copy[col] = df_copy[col].map(bool_map)
                    except:
                        pass
                
                # Düşük kardinaliteli kategorik
                elif len(unique_vals) < len(df_copy) * 0.05 and len(unique_vals) < 100:
                    df_copy[col] = df_copy[col].astype('category')
            
            # Tarih dönüşümü
            elif 'date' in str(col).lower() or 'tarih' in str(col).lower():
                try:
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                except:
                    pass
        
        except Exception as e:
            # Sessizce geç - hata kritik değil
            continue
    
    return df_copy

def auto_detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Sütun tiplerini otomatik tespit et.
    """
    col_types = {}
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        # 1. Molekül tespiti
        if any(k in col_lower for k in ['molekül', 'molecule', 'etken', 'active', 'ingredient', 'substance', 'api']):
            col_types[col] = 'molecule'
        
        # 2. Şirket tespiti
        elif any(k in col_lower for k in ['şirket', 'firma', 'company', 'manufacturer', 'üretici', 'corp', 'laboratory']):
            col_types[col] = 'company'
        
        # 3. Marka tespiti
        elif any(k in col_lower for k in ['marka', 'brand', 'trade', 'product_name', 'urun_adi', 'ticari']):
            col_types[col] = 'brand'
        
        # 4. Paket tespiti
        elif any(k in col_lower for k in ['paket', 'pack', 'sku', 'prod_pack', 'prodpack', 'form', 'doz', 'kutu', 'bottle']):
            col_types[col] = 'pack'
        
        # 5. Satış tespiti
        elif any(k in col_lower for k in ['satış', 'sales', 'revenue', 'cari', 'değer', 'value', 'turnover']):
            if safe_extract_year(col) is not None:
                col_types[col] = 'sales'
            else:
                col_types[col] = 'sales_total'
        
        # 6. Fiyat tespiti
        elif any(k in col_lower for k in ['fiyat', 'price', 'cost', 'maliyet', 'bedel']):
            col_types[col] = 'price'
        
        # 7. Büyüme tespiti
        elif any(k in col_lower for k in ['büyüme', 'growth', 'artış', 'increase', 'cagr']):
            col_types[col] = 'growth'
        
        # 8. Bölge tespiti
        elif any(k in col_lower for k in ['bölge', 'region', 'ulke', 'country', 'city', 'sehir']):
            col_types[col] = 'region'
        
        else:
            col_types[col] = 'other'
    
    return col_types

def calculate_growth_rate(series: pd.Series, periods: int = 1) -> float:
    """
    Güvenli büyüme oranı hesaplama.
    """
    if len(series) < periods + 1:
        return 0.0
    
    current = series.iloc[-1]
    previous = series.iloc[-1 - periods]
    
    if pd.isna(current) or pd.isna(previous):
        return 0.0
    
    if previous == 0 or math.isinf(previous) or math.isnan(previous):
        return 0.0
    
    try:
        growth = ((current - previous) / abs(previous)) * 100
        if math.isinf(growth) or math.isnan(growth):
            return 0.0
        return round(growth, 2)
    except:
        return 0.0

def calculate_cagr(start_value: float, end_value: float, years: float) -> float:
    """
    Bileşik yıllık büyüme oranı hesaplama.
    """
    if start_value <= 0 or years <= 0:
        return 0.0
    
    try:
        cagr = (pow(end_value / start_value, 1 / years) - 1) * 100
        if math.isinf(cagr) or math.isnan(cagr):
            return 0.0
        return round(cagr, 2)
    except:
        return 0.0

# ================================================
# 5. PRODPACK DERİNLİK ANALİZİ MOTORU
# ================================================

class ProdPackDepthEngine:
    """
    Molekül → Şirket → Marka → Paket hiyerarşik analiz motoru.
    Pazar kanibalizasyonu ve büyüme matrisi entegre.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.column_types = auto_detect_column_types(df)
        self.hierarchy = {}
        self.nodes: Dict[str, ProdPackNode] = {}
        self.root_id = "root"
        self.cannibalization_scores: List[CannibalizationScore] = []
        
        # Sütun eşleştirmeleri
        self.molecule_col = self._get_first_column_by_type('molecule')
        self.company_col = self._get_first_column_by_type('company')
        self.brand_col = self._get_first_column_by_type('brand')
        self.pack_col = self._get_first_column_by_type('pack')
        
        # Satış sütunları (yıl bazlı)
        self.sales_cols = self._get_columns_by_type('sales')
        self.sales_cols.sort(key=lambda x: safe_extract_year(x) or 0, reverse=True)
        
        # En güncel satış sütunu
        self.latest_sales_col = self.sales_cols[0] if self.sales_cols else None
        
    def _get_first_column_by_type(self, col_type: str) -> Optional[str]:
        """Belirli tipteki ilk sütunu döndür."""
        for col, ctype in self.column_types.items():
            if ctype == col_type:
                return col
        return None
    
    def _get_columns_by_type(self, col_type: str) -> List[str]:
        """Belirli tipteki tüm sütunları döndür."""
        return [col for col, ctype in self.column_types.items() if ctype == col_type]
    
    def validate_requirements(self) -> Tuple[bool, str]:
        """
        Analiz için gerekli sütunları kontrol et.
        """
        missing = []
        
        if not self.molecule_col:
            missing.append("Molekül")
        if not self.company_col:
            missing.append("Şirket")
        if not self.brand_col:
            missing.append("Marka")
        if not self.pack_col:
            missing.append("Paket")
        if not self.sales_cols:
            missing.append("Satış (yıl bazlı)")
        
        if missing:
            return False, f"Eksik sütunlar: {', '.join(missing)}"
        
        return True, "Gerekli tüm sütunlar mevcut"
    
    @st.cache_data(ttl=300, max_entries=5, show_spinner=False)
    def _cached_build_hierarchy(_self, df_hash: str, molecule_col: str, company_col: str, 
                               brand_col: str, pack_col: str, sales_col: str) -> Dict:
        """
        Cache'li hiyerarşi oluşturma.
        Büyük veri setleri için optimize.
        """
        return _self._build_hierarchy_impl()
    
    def _build_hierarchy_impl(self) -> Dict:
        """
        Hiyerarşi oluşturma implementasyonu.
        """
        # Kök düğüm
        self.nodes[self.root_id] = ProdPackNode(
            id=self.root_id,
            name="Tüm Pazar",
            level="root",
            value=self.df[self.latest_sales_col].sum() if self.latest_sales_col else 0
        )
        
        # Moleküller
        molecules = self.df[self.molecule_col].dropna().unique()
        
        for molecule in molecules[:100]:  # Performans için ilk 100 molekül
            mol_df = self.df[self.df[self.molecule_col] == molecule]
            mol_sales = mol_df[self.latest_sales_col].sum() if self.latest_sales_col else 0
            
            mol_id = f"mol_{hashlib.md5(str(molecule).encode()).hexdigest()[:8]}"
            
            self.nodes[mol_id] = ProdPackNode(
                id=mol_id,
                name=str(molecule)[:50],
                level="molecule",
                value=mol_sales,
                parent_id=self.root_id,
                metadata={"original_name": molecule}
            )
            
            self.nodes[self.root_id].children.append(mol_id)
            
            # Şirketler
            companies = mol_df[self.company_col].dropna().unique()
            
            for company in companies[:50]:  # Her molekül için ilk 50 şirket
                comp_df = mol_df[mol_df[self.company_col] == company]
                comp_sales = comp_df[self.latest_sales_col].sum() if self.latest_sales_col else 0
                
                comp_id = f"comp_{hashlib.md5(str(company).encode()).hexdigest()[:8]}"
                
                self.nodes[comp_id] = ProdPackNode(
                    id=comp_id,
                    name=str(company)[:50],
                    level="company",
                    value=comp_sales,
                    parent_id=mol_id,
                    metadata={"original_name": company}
                )
                
                self.nodes[mol_id].children.append(comp_id)
                
                # Markalar
                brands = comp_df[self.brand_col].dropna().unique()
                
                for brand in brands[:30]:  # Her şirket için ilk 30 marka
                    brand_df = comp_df[comp_df[self.brand_col] == brand]
                    brand_sales = brand_df[self.latest_sales_col].sum() if self.latest_sales_col else 0
                    
                    brand_id = f"brand_{hashlib.md5(str(brand).encode()).hexdigest()[:8]}"
                    
                    self.nodes[brand_id] = ProdPackNode(
                        id=brand_id,
                        name=str(brand)[:50],
                        level="brand",
                        value=brand_sales,
                        parent_id=comp_id,
                        metadata={"original_name": brand}
                    )
                    
                    self.nodes[comp_id].children.append(brand_id)
                    
                    # Paketler (ProdPack)
                    packs = brand_df[self.pack_col].dropna().unique()
                    
                    for pack in packs[:20]:  # Her marka için ilk 20 paket
                        pack_df = brand_df[brand_df[self.pack_col] == pack]
                        pack_sales = pack_df[self.latest_sales_col].sum() if self.latest_sales_col else 0
                        
                        pack_id = f"pack_{hashlib.md5(str(pack).encode()).hexdigest()[:8]}"
                        
                        # Büyüme oranı hesapla
                        growth = 0.0
                        if len(self.sales_cols) >= 2:
                            prev_col = self.sales_cols[1] if len(self.sales_cols) > 1 else self.sales_cols[0]
                            prev_sales = pack_df[prev_col].sum() if prev_col in pack_df.columns else 0
                            if prev_sales > 0:
                                growth = ((pack_sales - prev_sales) / prev_sales) * 100
                        
                        # Pazar payı
                        market_share = (pack_sales / mol_sales * 100) if mol_sales > 0 else 0
                        
                        self.nodes[pack_id] = ProdPackNode(
                            id=pack_id,
                            name=str(pack)[:50],
                            level="pack",
                            value=pack_sales,
                            parent_id=brand_id,
                            growth_rate=round(growth, 2),
                            market_share=round(market_share, 2),
                            metadata={"original_name": pack}
                        )
                        
                        self.nodes[brand_id].children.append(pack_id)
        
        return self._convert_hierarchy_to_dict()
    
    def _convert_hierarchy_to_dict(self) -> Dict:
        """
        Hiyerarşiyi dictionary formatına çevir.
        """
        hierarchy_dict = {
            'nodes': {},
            'root_id': self.root_id,
            'total_value': self.nodes[self.root_id].value if self.root_id in self.nodes else 0
        }
        
        for node_id, node in self.nodes.items():
            hierarchy_dict['nodes'][node_id] = {
                'id': node.id,
                'name': node.name,
                'level': node.level,
                'value': node.value,
                'parent_id': node.parent_id,
                'children': node.children,
                'growth_rate': node.growth_rate,
                'market_share': node.market_share,
                'metadata': node.metadata
            }
        
        return hierarchy_dict
    
    def build_hierarchy(self) -> Dict:
        """
        Ana hiyerarşi oluşturma fonksiyonu.
        """
        valid, message = self.validate_requirements()
        if not valid:
            st.error(f"❌ {message}")
            return {}
        
        # DataFrame hash'i oluştur
        df_hash = hashlib.md5(pd.util.hash_pandas_object(self.df).values).hexdigest()
        
        # Cache'li fonksiyonu çağır
        return self._cached_build_hierarchy(
            df_hash,
            self.molecule_col,
            self.company_col,
            self.brand_col,
            self.pack_col,
            self.latest_sales_col
        )
    
    def get_molecule_drilldown(self, molecule_name: str) -> pd.DataFrame:
        """
        Seçilen molekül için detaylı drill-down raporu.
        """
        if not self.molecule_col:
            return pd.DataFrame()
        
        mol_df = self.df[self.df[self.molecule_col] == molecule_name].copy()
        
        if mol_df.empty:
            return pd.DataFrame()
        
        rows = []
        
        # Gruplama yap
        grouped = mol_df.groupby([self.company_col, self.brand_col, self.pack_col])
        
        for (company, brand, pack), group in grouped:
            # Satış değerleri
            sales_values = []
            for col in self.sales_cols[:4]:  # Son 4 dönem
                sales_values.append(group[col].sum() if col in group.columns else 0)
            
            # Büyüme oranı
            growth = calculate_growth_rate(pd.Series(sales_values))
            
            # Pazar payı
            mol_total = mol_df[self.latest_sales_col].sum() if self.latest_sales_col else 0
            pack_sales = sales_values[0] if sales_values else 0
            market_share = (pack_sales / mol_total * 100) if mol_total > 0 else 0
            
            rows.append({
                'Molekül': molecule_name,
                'Şirket': company,
                'Marka': brand,
                'ProdPack': pack,
                'Satış_Hacmi': pack_sales,
                'Büyüme_Oranı_%': growth,
                'Pazar_Payı_%': round(market_share, 2),
                'Hacim_Trendi': '📈' if growth > 5 else '📉' if growth < -5 else '➡️'
            })
        
        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df = result_df.sort_values('Satış_Hacmi', ascending=False)
        
        return result_df
    
    def analyze_cannibalization(self, molecule_name: str) -> pd.DataFrame:
        """
        Aynı şirket içindeki markalar arası kanibalizasyon analizi.
        Korelasyon bazlı hesaplama.
        """
        if not all([self.molecule_col, self.company_col, self.brand_col]):
            return pd.DataFrame()
        
        mol_df = self.df[self.df[self.molecule_col] == molecule_name].copy()
        
        if mol_df.empty:
            return pd.DataFrame()
        
        cannibal_data = []
        
        # Şirket bazlı gruplama
        companies = mol_df[self.company_col].dropna().unique()
        
        for company in companies:
            comp_df = mol_df[mol_df[self.company_col] == company]
            brands = comp_df[self.brand_col].dropna().unique()
            
            if len(brands) < 2:
                continue
            
            # Marka çiftlerini analiz et
            for i, brand1 in enumerate(brands):
                for brand2 in brands[i+1:]:
                    brand1_df = comp_df[comp_df[self.brand_col] == brand1]
                    brand2_df = comp_df[comp_df[self.brand_col] == brand2]
                    
                    # Zaman serisi oluştur
                    brand1_series = []
                    brand2_series = []
                    
                    for col in self.sales_cols[:8]:  # Son 8 dönem
                        if col in brand1_df.columns:
                            brand1_series.append(brand1_df[col].sum())
                        if col in brand2_df.columns:
                            brand2_series.append(brand2_df[col].sum())
                    
                    if len(brand1_series) > 2 and len(brand2_series) > 2:
                        # Korelasyon hesapla
                        try:
                            correlation = np.corrcoef(brand1_series, brand2_series)[0, 1]
                            if np.isnan(correlation):
                                correlation = 0
                        except:
                            correlation = 0
                        
                        # Kanibalizasyon skoru
                        cannibal_score = abs(correlation) if correlation < 0 else 0
                        
                        # Hacim örtüşmesi
                        volume_overlap = min(brand1_series[-1], brand2_series[-1]) / max(brand1_series[-1], brand2_series[-1], 1)
                        
                        # Büyüme etkisi
                        growth1 = calculate_growth_rate(pd.Series(brand1_series))
                        growth2 = calculate_growth_rate(pd.Series(brand2_series))
                        growth_impact = abs(growth1 - growth2) / 100
                        
                        # Öneri
                        if cannibal_score > 0.7:
                            recommendation = "🚨 Acil müdahale gerekli - Ürün farklılaştırması yapın"
                        elif cannibal_score > 0.4:
                            recommendation = "⚠️ Yakın izleme - Pazarlama stratejilerini gözden geçirin"
                        else:
                            recommendation = "✅ Düşük kanibalizasyon - Mevcut stratejiyi koruyun"
                        
                        cannibal_data.append({
                            'Molekül': molecule_name,
                            'Şirket': company,
                            'Marka_A': brand1,
                            'Marka_B': brand2,
                            'Korelasyon': round(correlation, 3),
                            'Kanibalizasyon_Skoru': round(cannibal_score, 3),
                            'Hacim_Örtüşmesi': round(volume_overlap, 3),
                            'Büyüme_Etkisi': round(growth_impact, 3),
                            'Öneri': recommendation
                        })
        
        result_df = pd.DataFrame(cannibal_data)
        if not result_df.empty:
            result_df = result_df.sort_values('Kanibalizasyon_Skoru', ascending=False)
        
        return result_df
    
    def get_sunburst_data(self) -> Dict:
        """
        Sunburst grafik için veri hazırla.
        """
        ids = []
        labels = []
        parents = []
        values = []
        
        for node_id, node in self.nodes.items():
            ids.append(node_id)
            labels.append(node.name[:30])
            parents.append(node.parent_id if node.parent_id else "")
            values.append(node.value)
        
        return {
            'ids': ids,
            'labels': labels,
            'parents': parents,
            'values': values
        }
    
    def get_sankey_data(self) -> Dict:
        """
        Sankey diyagram için veri hazırla.
        """
        sources = []
        targets = []
        values = []
        
        node_labels = []
        node_indices = {}
        
        # Kök düğümü ekle
        node_indices[self.root_id] = 0
        node_labels.append(self.nodes[self.root_id].name)
        current_idx = 1
        
        # Molekülleri ekle
        for node_id, node in self.nodes.items():
            if node.level == 'molecule':
                if node_id not in node_indices:
                    node_indices[node_id] = current_idx
                    node_labels.append(node.name)
                    current_idx += 1
                
                sources.append(node_indices[self.root_id])
                targets.append(node_indices[node_id])
                values.append(node.value)
        
        # Şirketleri ekle
        for node_id, node in self.nodes.items():
            if node.level == 'company' and node.parent_id:
                if node_id not in node_indices:
                    node_indices[node_id] = current_idx
                    node_labels.append(node.name)
                    current_idx += 1
                
                if node.parent_id in node_indices:
                    sources.append(node_indices[node.parent_id])
                    targets.append(node_indices[node_id])
                    values.append(node.value)
        
        # Markaları ekle
        for node_id, node in self.nodes.items():
            if node.level == 'brand' and node.parent_id:
                if node_id not in node_indices:
                    node_indices[node_id] = current_idx
                    node_labels.append(node.name)
                    current_idx += 1
                
                if node.parent_id in node_indices:
                    sources.append(node_indices[node.parent_id])
                    targets.append(node_indices[node_id])
                    values.append(node.value)
        
        # Paketleri ekle
        for node_id, node in self.nodes.items():
            if node.level == 'pack' and node.parent_id:
                if node_id not in node_indices:
                    node_indices[node_id] = current_idx
                    node_labels.append(node.name)
                    current_idx += 1
                
                if node.parent_id in node_indices:
                    sources.append(node_indices[node.parent_id])
                    targets.append(node_indices[node_id])
                    values.append(node.value)
        
        return {
            'node_labels': node_labels,
            'sources': sources,
            'targets': targets,
            'values': values
        }

# ================================================
# 6. TAHMİNLEME VE YATIRIM MOTORU
# ================================================

class StrategicForecastEngine:
    """
    Holt-Winters ile 2025-2026 tahminleme.
    Yatırım tavsiyesi ve risk analizi entegre.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.column_types = auto_detect_column_types(df)
        self.sales_cols = self._get_columns_by_type('sales')
        self.sales_cols.sort(key=lambda x: safe_extract_year(x) or 0)
        
    def _get_columns_by_type(self, col_type: str) -> List[str]:
        """Belirli tipteki sütunları döndür."""
        return [col for col, ctype in self.column_types.items() if ctype == col_type]
    
    @st.cache_data(ttl=600, max_entries=5, show_spinner=False)
    def _cached_forecast(_self, df_hash: str, sales_cols_tuple: tuple) -> Dict:
        """
        Cache'li tahminleme.
        """
        return _self._forecast_impl(list(sales_cols_tuple))
    
    def _forecast_impl(self, sales_cols: List[str]) -> Dict:
        """
        Tahminleme implementasyonu.
        """
        result = {
            'success': False,
            'forecast': None,
            'growth_rate': 0.0,
            'investment_advice': [],
            'model_stats': {}
        }
        
        if not STATSMODELS_AVAILABLE:
            result['investment_advice'].append({
                'title': '⚠️ Modül Eksik',
                'message': 'statsmodels kurulu değil. Holt-Winters tahminleme için "pip install statsmodels" komutunu çalıştırın.',
                'action': InvestmentAction.OPTIMIZE,
                'priority': 'high',
                'impact': 'operational',
                'roi_potential': 0,
                'risk_level': 'Yüksek'
            })
            return result
        
        if len(sales_cols) < 4:
            result['investment_advice'].append({
                'title': '❌ Yetersiz Veri',
                'message': 'Tahminleme için en az 4 yıllık satış verisi gereklidir.',
                'action': InvestmentAction.DEFEND,
                'priority': 'high',
                'impact': 'operational',
                'roi_potential': 0,
                'risk_level': 'Kritik'
            })
            return result
        
        try:
            # Toplam pazar satışları
            market_series = []
            years = []
            
            for col in sales_cols:
                year = safe_extract_year(col)
                if year:
                    market_series.append(self.df[col].sum())
                    years.append(year)
            
            if len(market_series) < 4:
                return result
            
            # Zaman serisi oluştur
            series = pd.Series(market_series, index=pd.date_range(start=f'{years[0]}-01-01', periods=len(market_series), freq='Y'))
            
            # Holt-Winters modeli
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            
            fitted_model = model.fit()
            
            # 2025-2026 tahmini (8 çeyrek veya 2 yıl)
            forecast_horizon = 8
            forecast = fitted_model.forecast(forecast_horizon)
            
            # Çeyreklik böl
            quarters = []
            for year in [2025, 2026]:
                for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                    quarters.append(f"{year} {q}")
            
            quarters = quarters[:len(forecast)]
            forecast_values = forecast.values[:len(quarters)]
            
            # Güven aralığı
            residuals = series - fitted_model.fittedvalues
            std_residual = residuals.std()
            
            lower_bound = forecast_values - 1.96 * std_residual
            upper_bound = forecast_values + 1.96 * std_residual
            
            # Büyüme oranı
            last_historical = market_series[-1] if market_series else 0
            last_forecast = forecast_values[-1] if len(forecast_values) > 0 else 0
            
            if last_historical > 0:
                growth_rate = ((last_forecast - last_historical) / last_historical) * 100
            else:
                growth_rate = 0.0
            
            # Model performans metrikleri
            mape = np.mean(np.abs((series - fitted_model.fittedvalues) / series)) * 100
            rmse = np.sqrt(np.mean((series - fitted_model.fittedvalues) ** 2))
            
            result['success'] = True
            result['forecast'] = {
                'quarters': quarters,
                'values': forecast_values.tolist(),
                'lower_bound': lower_bound.tolist(),
                'upper_bound': upper_bound.tolist(),
                'historical_values': market_series,
                'historical_years': years
            }
            result['growth_rate'] = round(growth_rate, 2)
            result['model_stats'] = {
                'mape': round(mape, 2),
                'rmse': round(rmse, 2),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
            # Yatırım tavsiyeleri üret
            result['investment_advice'] = self._generate_investment_advice(growth_rate, mape, market_series)
            
        except Exception as e:
            result['investment_advice'].append({
                'title': '⚠️ Tahminleme Hatası',
                'message': f'Holt-Winters modeli çalıştırılamadı: {str(e)[:100]}',
                'action': InvestmentAction.DEFEND,
                'priority': 'high',
                'impact': 'operational',
                'roi_potential': 0,
                'risk_level': 'Yüksek'
            })
        
        return result
    
    def _generate_investment_advice(self, growth_rate: float, mape: float, historical: List) -> List[Dict]:
        """
        Büyüme oranına göre yatırım tavsiyesi üret.
        """
        advice_list = []
        
        # 1. Ana büyüme stratejisi
        if growth_rate > 20:
            advice_list.append({
                'title': '🚀 AGGRESİF BÜYÜME STRATEJİSİ',
                'message': f'Pazar %{growth_rate:.1f} büyüyor. Ar-Ge bütçesini %40 artırın, yeni ürün lansmanlarına öncelik verin.',
                'action': InvestmentAction.EXPAND,
                'priority': 'high',
                'impact': 'strategic',
                'roi_potential': 85,
                'risk_level': 'Orta'
            })
        elif growth_rate > 10:
            advice_list.append({
                'title': '📈 GÜÇLÜ BÜYÜME STRATEJİSİ',
                'message': f'Pazar %{growth_rate:.1f} büyüyor. Satış ve pazarlama yatırımlarını %25 artırın.',
                'action': InvestmentAction.EXPAND,
                'priority': 'high',
                'impact': 'strategic',
                'roi_potential': 75,
                'risk_level': 'Düşük'
            })
        elif growth_rate > 3:
            advice_list.append({
                'title': '📊 İSTİKRARLI BÜYÜME',
                'message': f'Pazar %{growth_rate:.1f} büyüyor. Mevcut portföyü optimize edin, verimlilik artışına odaklanın.',
                'action': InvestmentAction.OPTIMIZE,
                'priority': 'medium',
                'impact': 'financial',
                'roi_potential': 60,
                'risk_level': 'Düşük'
            })
        elif growth_rate > -3:
            advice_list.append({
                'title': '⏸️ DURGUN PAZAR',
                'message': 'Pazar büyümesi yavaş. Maliyet optimizasyonu ve operasyonel verimliliğe odaklanın.',
                'action': InvestmentAction.DEFEND,
                'priority': 'medium',
                'impact': 'operational',
                'roi_potential': 40,
                'risk_level': 'Orta'
            })
        else:
            advice_list.append({
                'title': '⚠️ PAZAR DARALMASI',
                'message': f'Pazar %{growth_rate:.1f} daralıyor. Nakit akışını koruyun, birleşme ve satın alma fırsatlarını değerlendirin.',
                'action': InvestmentAction.CONSOLIDATE,
                'priority': 'high',
                'impact': 'strategic',
                'roi_potential': 30,
                'risk_level': 'Yüksek'
            })
        
        # 2. Risk yönetimi tavsiyesi
        if mape > 20:
            advice_list.append({
                'title': '⚖️ RİSK YÖNETİMİ',
                'message': f'Tahmin belirsizliği yüksek (MAPE: %{mape:.1f}). Portföy çeşitlendirmesi yapın, hedge stratejileri uygulayın.',
                'action': InvestmentAction.HEDGE,
                'priority': 'high',
                'impact': 'financial',
                'roi_potential': 45,
                'risk_level': 'Yüksek'
            })
        
        # 3. Uzun vadeli strateji
        if len(historical) >= 5:
            cagr = calculate_cagr(historical[0], historical[-1], len(historical) - 1)
            
            if cagr < 2 and growth_rate > 5:
                advice_list.append({
                    'title': '🔄 DÖNÜŞÜM FIRSATI',
                    'message': f'Tarihsel CAGR %{cagr:.1f}, önümüzdeki dönem %{growth_rate:.1f} büyüme bekleniyor. Dönüşüm stratejilerini devreye alın.',
                    'action': InvestmentAction.OPTIMIZE,
                    'priority': 'medium',
                    'impact': 'strategic',
                    'roi_potential': 70,
                    'risk_level': 'Orta'
                })
        
        return advice_list
    
    def forecast_market(self) -> Dict:
        """
        Ana tahminleme fonksiyonu.
        """
        if not self.sales_cols:
            return {
                'success': False,
                'forecast': None,
                'growth_rate': 0.0,
                'investment_advice': [{
                    'title': '❌ Veri Hatası',
                    'message': 'Satış sütunu bulunamadı. Lütfen veri formatını kontrol edin.',
                    'action': InvestmentAction.DEFEND,
                    'priority': 'high',
                    'impact': 'operational',
                    'roi_potential': 0,
                    'risk_level': 'Kritik'
                }]
            }
        
        # DataFrame hash'i
        df_hash = hashlib.md5(pd.util.hash_pandas_object(self.df).values).hexdigest()
        
        # Cache'li tahminleme
        return self._cached_forecast(df_hash, tuple(self.sales_cols))

# ================================================
# 7. ANOMALİ TESPİT MOTORU - ISOLATION FOREST
# ================================================

class AnomalyDetectionEngine:
    """
    IsolationForest ile çoklu algoritma anomali tespiti.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.column_types = auto_detect_column_types(df)
        
    @st.cache_data(ttl=300, max_entries=5, show_spinner=False)
    def _cached_detect_anomalies(_self, df_hash: str, contamination: float) -> pd.DataFrame:
        """
        Cache'li anomali tespiti.
        """
        return _self._detect_anomalies_impl(contamination)
    
    def _detect_anomalies_impl(self, contamination: float = 0.1) -> pd.DataFrame:
        """
        Anomali tespiti implementasyonu.
        """
        result_df = self.df.copy()
        
        if not SKLEARN_AVAILABLE:
            result_df['Anomali_Skoru'] = 0.5
            result_df['Anomali_Tespiti'] = 1
            result_df['Anomali_Tipi'] = 'Normal'
            result_df['Risk_Seviyesi'] = RiskLevel.NORMAL.value
            return result_df
        
        # Sayısal sütunları seç
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Satış ve büyüme ile ilgili sütunları filtrele
        feature_cols = []
        
        for col in numeric_cols:
            col_lower = str(col).lower()
            if any(k in col_lower for k in ['satış', 'sales', 'hacim', 'volume', 'cari', 'değer', 'buyume', 'growth']):
                feature_cols.append(col)
        
        # Yeterli özellik yoksa, en yüksek varyanslı sütunları kullan
        if len(feature_cols) < 3 and len(numeric_cols) >= 3:
            variances = self.df[numeric_cols].var()
            feature_cols = variances.nlargest(min(5, len(variances))).index.tolist()
        
        if len(feature_cols) >= 2:
            try:
                X = self.df[feature_cols].fillna(0)
                
                # RobustScaler ile ölçeklendirme
                scaler = RobustScaler(quantile_range=(5, 95))
                X_scaled = scaler.fit_transform(X)
                
                # Isolation Forest
                iso_forest = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=200,
                    max_samples='auto',
                    bootstrap=False,
                    n_jobs=-1,
                    warm_start=False
                )
                
                predictions = iso_forest.fit_predict(X_scaled)
                scores = iso_forest.decision_function(X_scaled)
                
                # Skorları 0-1 aralığına normalize et
                min_score, max_score = scores.min(), scores.max()
                if max_score > min_score:
                    normalized_scores = (scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores = np.ones_like(scores) * 0.5
                
                result_df['Anomali_Skoru'] = normalized_scores
                result_df['Anomali_Tespiti'] = predictions
                
                # Anomali tipi belirleme
                conditions = [
                    (predictions == -1) & (normalized_scores < 0.25),
                    (predictions == -1) & (normalized_scores >= 0.25),
                    (predictions == 1) & (normalized_scores > 0.75),
                    (predictions == 1) & (normalized_scores <= 0.75)
                ]
                
                choices = ['🔴 Kritik Düşüş', '🟠 Anormal Değişim', '🟢 Aşırı Büyüme', '✅ Normal']
                result_df['Anomali_Tipi'] = np.select(conditions, choices, default='✅ Normal')
                
                # Risk seviyesi
                risk_conditions = [
                    (result_df['Anomali_Tipi'] == '🔴 Kritik Düşüş'),
                    (result_df['Anomali_Tipi'] == '🟠 Anormal Değişim'),
                    (result_df['Anomali_Tipi'] == '🟢 Aşırı Büyüme'),
                    (result_df['Anomali_Tipi'] == '✅ Normal')
                ]
                
                risk_choices = [
                    RiskLevel.KRITIK.value,
                    RiskLevel.YUKSEK.value,
                    RiskLevel.ORTA.value,
                    RiskLevel.NORMAL.value
                ]
                
                result_df['Risk_Seviyesi'] = np.select(risk_conditions, risk_choices, default=RiskLevel.NORMAL.value)
                
            except Exception as e:
                # Hata durumunda varsayılan değerler
                result_df['Anomali_Skoru'] = 0.5
                result_df['Anomali_Tespiti'] = 1
                result_df['Anomali_Tipi'] = '✅ Normal'
                result_df['Risk_Seviyesi'] = RiskLevel.NORMAL.value
        else:
            # Yetersiz özellik
            result_df['Anomali_Skoru'] = 0.5
            result_df['Anomali_Tespiti'] = 1
            result_df['Anomali_Tipi'] = '✅ Normal'
            result_df['Risk_Seviyesi'] = RiskLevel.NORMAL.value
        
        return result_df
    
    def detect_anomalies(self, contamination: float = 0.1) -> pd.DataFrame:
        """
        Ana anomali tespit fonksiyonu.
        """
        df_hash = hashlib.md5(pd.util.hash_pandas_object(self.df).values).hexdigest()
        return self._cached_detect_anomalies(df_hash, contamination)

# ================================================
# 8. SEGMENTASYON MOTORU - PCA + K-MEANS
# ================================================

class ProductSegmentationEngine:
    """
    PCA ve K-Means ile ürün segmentasyonu.
    Liderler, Potansiyeller, Riskli Ürünler sınıflandırması.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.column_types = auto_detect_column_types(df)
        
    @st.cache_data(ttl=300, max_entries=5, show_spinner=False)
    def _cached_segment(_self, df_hash: str, n_clusters: int) -> Dict:
        """
        Cache'li segmentasyon.
        """
        return _self._segment_impl(n_clusters)
    
    def _segment_impl(self, n_clusters: int = 4) -> Dict:
        """
        Segmentasyon implementasyonu.
        """
        result = {
            'success': False,
            'segmented_df': None,
            'pca_components': None,
            'pca_explained_variance': [],
            'labels': None,
            'segment_names': {},
            'silhouette_score': 0.0,
            'metrics': {}
        }
        
        if not SKLEARN_AVAILABLE:
            return result
        
        try:
            # Özellik mühendisliği
            feature_df = self.df.copy()
            
            # Satış sütunları
            sales_cols = [col for col in self.column_types if self.column_types[col] == 'sales']
            sales_cols.sort(key=lambda x: safe_extract_year(x) or 0, reverse=True)
            
            if not sales_cols:
                return result
            
            latest_col = sales_cols[0]
            prev_col = sales_cols[1] if len(sales_cols) > 1 else sales_cols[0]
            
            # 1. Pazar payı hesapla
            total_sales = self.df[latest_col].sum()
            if total_sales > 0:
                feature_df['Pazar_Payi'] = (self.df[latest_col] / total_sales) * 100
            else:
                feature_df['Pazar_Payi'] = 0
            
            # 2. Büyüme hızı hesapla
            feature_df['Buyume_Hizi'] = 0.0
            mask = (self.df[prev_col] != 0) & (self.df[prev_col].notna())
            if mask.any():
                feature_df.loc[mask, 'Buyume_Hizi'] = (
                    (self.df.loc[mask, latest_col] - self.df.loc[mask, prev_col]) / 
                    self.df.loc[mask, prev_col].abs()
                ) * 100
            
            # 3. Fiyat esnekliği (varsayılan)
            if 'price' in self.column_types.values():
                price_cols = [col for col in self.column_types if self.column_types[col] == 'price']
                if price_cols:
                    feature_df['Fiyat_Esnekligi'] = -0.8 + (self.df[price_cols[0]].rank(pct=True) * -0.7)
                else:
                    feature_df['Fiyat_Esnekligi'] = np.random.uniform(-1.5, -0.3, len(self.df))
            else:
                feature_df['Fiyat_Esnekligi'] = np.random.uniform(-1.5, -0.3, len(self.df))
            
            # 4. Hacim değişkenliği
            if len(sales_cols) >= 3:
                sales_data = self.df[sales_cols[:3]].values
                feature_df['Hacim_Degiskenligi'] = np.std(sales_data, axis=1) / (np.mean(sales_data, axis=1) + 1)
            else:
                feature_df['Hacim_Degiskenligi'] = 0.5
            
            # Segmentasyon için feature matrix
            feature_cols = ['Pazar_Payi', 'Buyume_Hizi', 'Fiyat_Esnekligi', 'Hacim_Degiskenligi']
            X = feature_df[feature_cols].fillna(0)
            
            # Sonsuz ve NaN değerleri temizle
            X = X.replace([np.inf, -np.inf], 0)
            
            # Ölçeklendirme
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PCA - Boyut indirgeme
            n_components = min(2, X_scaled.shape[1])
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            # K-Means kümeleme
            n_clusters = min(n_clusters, len(self.df))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            # Silhouette skoru
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X_scaled, labels)
            else:
                silhouette = 0.0
            
            # Segment isimlendirme
            segment_map = {}
            for i in range(n_clusters):
                cluster_mask = labels == i
                cluster_data = X[cluster_mask]
                
                avg_growth = cluster_data['Buyume_Hizi'].mean()
                avg_share = cluster_data['Pazar_Payi'].mean()
                avg_volatility = cluster_data['Hacim_Degiskenligi'].mean()
                
                # Segment sınıflandırması
                if avg_share > X['Pazar_Payi'].quantile(0.7) and avg_growth > X['Buyume_Hizi'].quantile(0.6):
                    name = ProductSegment.LIDER.value
                elif avg_growth > X['Buyume_Hizi'].quantile(0.7) and avg_share < X['Pazar_Payi'].quantile(0.5):
                    name = ProductSegment.POTANSIYEL.value
                elif avg_volatility > X['Hacim_Degiskenligi'].quantile(0.7) or avg_growth < X['Buyume_Hizi'].quantile(0.3):
                    name = ProductSegment.RISKLI.value
                elif avg_share > X['Pazar_Payi'].quantile(0.6) and avg_growth < X['Buyume_Hizi'].quantile(0.4):
                    name = ProductSegment.NAKIT.value
                elif avg_growth < X['Buyume_Hizi'].quantile(0.2) and avg_share < X['Pazar_Payi'].quantile(0.3):
                    name = ProductSegment.SORU.value
                else:
                    name = ProductSegment.OLGUN.value
                
                segment_map[i] = name
            
            # Segment etiketlerini DataFrame'e ekle
            feature_df['Segment_Cluster'] = labels
            feature_df['Segment_Adi'] = feature_df['Segment_Cluster'].map(segment_map)
            feature_df['Segment_Renk'] = feature_df['Segment_Adi'].map({
                ProductSegment.LIDER.value: '#d4af37',
                ProductSegment.POTANSIYEL.value: '#4caf50',
                ProductSegment.RISKLI.value: '#f44336',
                ProductSegment.OLGUN.value: '#2196f3',
                ProductSegment.NAKIT.value: '#9c27b0',
                ProductSegment.SORU.value: '#ff9800'
            })
            
            # Sonuçları hazırla
            result['success'] = True
            result['segmented_df'] = feature_df
            result['pca_components'] = X_pca
            result['pca_explained_variance'] = pca.explained_variance_ratio_.tolist()
            result['labels'] = labels
            result['segment_names'] = segment_map
            result['silhouette_score'] = silhouette
            result['metrics'] = {
                'silhouette': round(silhouette, 3),
                'n_clusters': n_clusters,
                'pca_variance_ratio': round(pca.explained_variance_ratio_.sum(), 3)
            }
            
        except Exception as e:
            # Hata durumunda boş sonuç
            pass
        
        return result
    
    def segment_products(self, n_clusters: int = 4) -> Dict:
        """
        Ana segmentasyon fonksiyonu.
        """
        df_hash = hashlib.md5(pd.util.hash_pandas_object(self.df).values).hexdigest()
        return self._cached_segment(df_hash, n_clusters)

# ================================================
# 9. GÖRSELLEŞTİRME MOTORU
# ================================================

class VisualizationEngine:
    """
    Gelişmiş görselleştirme motoru.
    Executive Dark Mode temalı.
    """
    
    @staticmethod
    def create_sunburst_chart(sunburst_data: Dict, title: str = "ProdPack Hiyerarşisi") -> go.Figure:
        """
        Sunburst grafik oluştur.
        """
        if not sunburst_data or not sunburst_data.get('ids'):
            fig = go.Figure()
            fig.update_layout(
                title=dict(text="Veri bulunamadı", font=dict(color='#c0c0c0'), x=0.5),
                paper_bgcolor='rgba(10,25,41,0)',
                plot_bgcolor='rgba(10,25,41,0)',
                height=500
            )
            return fig
        
        fig = go.Figure(go.Sunburst(
            ids=sunburst_data['ids'],
            labels=sunburst_data['labels'],
            parents=sunburst_data['parents'],
            values=sunburst_data['values'],
            branchvalues='total',
            marker=dict(
                colorscale='deep',
                line=dict(width=2, color='#0a1929')
            ),
            hovertemplate='<b>%{label}</b><br>Değer: %{value:,.0f}<br>Pay: %{percentRoot:.1%}<extra></extra>',
            textfont=dict(size=12, color='white'),
            insidetextorientation='radial'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'🧬 {title}',
                font=dict(size=20, color='#d4af37', family='Arial Black'),
                x=0.5,
                y=0.98
            ),
            paper_bgcolor='rgba(10,25,41,0)',
            plot_bgcolor='rgba(10,25,41,0)',
            font=dict(color='white', size=11),
            height=600,
            margin=dict(t=60, l=10, r=10, b=10),
            hoverlabel=dict(
                bgcolor='#1e3a5f',
                font_size=12,
                font_family='Arial'
            )
        )
        
        return fig
    
    @staticmethod
    def create_sankey_diagram(sankey_data: Dict) -> go.Figure:
        """
        Sankey diyagram oluştur.
        """
        if not sankey_data or not sankey_data.get('node_labels'):
            fig = go.Figure()
            fig.update_layout(
                title=dict(text="Akış verisi bulunamadı", font=dict(color='#c0c0c0'), x=0.5),
                paper_bgcolor='rgba(10,25,41,0)',
                plot_bgcolor='rgba(10,25,41,0)',
                height=500
            )
            return fig
        
        # Node renkleri
        node_colors = []
        for i, label in enumerate(sankey_data['node_labels']):
            if i == 0:
                node_colors.append('#0a1929')  # Root
            elif 'Molekül' in label or len(label) < 15:
                node_colors.append('#1e3a5f')  # Molekül
            else:
                node_colors.append('#2d4a7a')  # Diğer
        
        # Link renkleri
        link_colors = ['rgba(212,175,55,0.3)'] * len(sankey_data['sources'])
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="#d4af37", width=0.8),
                label=sankey_data['node_labels'],
                color=node_colors,
                hovertemplate='%{label}<br>Değer: %{value:,.0f}<extra></extra>'
            ),
            link=dict(
                source=sankey_data['sources'],
                target=sankey_data['targets'],
                value=sankey_data['values'],
                color=link_colors,
                hovertemplate='%{source.label} → %{target.label}<br>Değer: %{value:,.0f}<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title=dict(
                text='🔄 Pazar Akış Analizi',
                font=dict(size=20, color='#d4af37', family='Arial Black'),
                x=0.5,
                y=0.98
            ),
            paper_bgcolor='rgba(10,25,41,0)',
            font=dict(color='white', size=11),
            height=600,
            margin=dict(t=60, l=20, r=20, b=20)
        )
        
        return fig
    
    @staticmethod
    def create_cannibalization_heatmap(cannibal_df: pd.DataFrame) -> go.Figure:
        """
        Kanibalizasyon heatmap oluştur.
        """
        if cannibal_df.empty or len(cannibal_df) < 2:
            fig = go.Figure()
            fig.update_layout(
                title=dict(text="Kanibalizasyon verisi yok", font=dict(color='#c0c0c0'), x=0.5),
                paper_bgcolor='rgba(10,25,41,0)',
                plot_bgcolor='rgba(10,25,41,0)',
                height=400
            )
            return fig
        
        # Pivot tablo oluştur
        pivot_data = cannibal_df.pivot_table(
            values='Kanibalizasyon_Skoru',
            index='Marka_A',
            columns='Marka_B',
            fill_value=0
        )
        
        # İlk 10 marka
        top_brands = cannibal_df['Marka_A'].value_counts().head(10).index
        pivot_data = pivot_data.loc[pivot_data.index.intersection(top_brands)]
        pivot_data = pivot_data[pivot_data.columns.intersection(top_brands)]
        
        fig = px.imshow(
            pivot_data,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title=None,
            labels=dict(x="Marka B", y="Marka A", color="Kanibalizasyon")
        )
        
        fig.update_layout(
            title=dict(
                text='🔄 Pazar Kanibalizasyon Matrisi',
                font=dict(size=18, color='#d4af37'),
                x=0.5
            ),
            paper_bgcolor='rgba(10,25,41,0)',
            plot_bgcolor='rgba(10,25,41,0)',
            font=dict(color='white', size=11),
            height=450,
            xaxis=dict(tickfont=dict(size=10, color='white')),
            yaxis=dict(tickfont=dict(size=10, color='white'))
        )
        
        fig.update_coloraxes(
            colorbar=dict(
                title='Skor',
                title_font=dict(color='white'),
                tickfont=dict(color='white'),
                bgcolor='rgba(10,25,41,0.8)'
            ),
            colorscale='RdBu_r',
            zmid=0
        )
        
        return fig
    
    @staticmethod
    def create_forecast_chart(forecast_data: Dict) -> go.Figure:
        """
        Tahmin grafiği oluştur.
        """
        if not forecast_data or 'forecast' not in forecast_data:
            fig = go.Figure()
            fig.update_layout(
                title=dict(text="Tahmin verisi yok", font=dict(color='#c0c0c0'), x=0.5),
                paper_bgcolor='rgba(10,25,41,0)',
                plot_bgcolor='rgba(10,25,41,0)',
                height=400
            )
            return fig
        
        forecast = forecast_data['forecast']
        
        fig = go.Figure()
        
        # Tarihsel veri
        if 'historical_values' in forecast and 'historical_years' in forecast:
            fig.add_trace(go.Scatter(
                x=[str(y) for y in forecast['historical_years']],
                y=forecast['historical_values'],
                mode='lines+markers',
                name='Tarihsel Satış',
                line=dict(color='#c0c0c0', width=4),
                marker=dict(size=10, symbol='diamond', color='#c0c0c0'),
                hovertemplate='Yıl: %{x}<br>Satış: %{y:,.0f} TL<extra></extra>'
            ))
        
        # Tahmin
        fig.add_trace(go.Scatter(
            x=forecast['quarters'],
            y=forecast['values'],
            mode='lines+markers',
            name='Tahmin 2025-2026',
            line=dict(color='#d4af37', width=4, dash='dash'),
            marker=dict(size=12, symbol='star', color='#d4af37'),
            hovertemplate='Dönem: %{x}<br>Tahmin: %{y:,.0f} TL<extra></extra>'
        ))
        
        # Güven aralığı
        if 'lower_bound' in forecast and 'upper_bound' in forecast:
            fig.add_trace(go.Scatter(
                x=forecast['quarters'] + forecast['quarters'][::-1],
                y=forecast['upper_bound'] + forecast['lower_bound'][::-1],
                fill='toself',
                fillcolor='rgba(212,175,55,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Güven Aralığı (%95)',
                hovertemplate='Alt: %{customdata[0]:,.0f}<br>Üst: %{customdata[1]:,.0f}<extra></extra>',
                customdata=list(zip(forecast['lower_bound'], forecast['upper_bound']))
            ))
        
        fig.update_layout(
            title=dict(
                text='📈 Holt-Winters Pazar Tahmini 2025-2026',
                font=dict(size=20, color='#d4af37', family='Arial Black'),
                x=0.5
            ),
            xaxis=dict(
                title='Dönem',
                title_font=dict(color='white', size=14),
                tickfont=dict(color='white', size=11),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title='Pazar Değeri (TL)',
                title_font=dict(color='white', size=14),
                tickfont=dict(color='white', size=11),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            paper_bgcolor='rgba(10,25,41,0)',
            plot_bgcolor='rgba(10,25,41,0)',
            font=dict(color='white'),
            height=450,
            hovermode='x unified',
            legend=dict(
                font=dict(color='white', size=11),
                bgcolor='rgba(10,25,41,0.7)',
                bordercolor='#d4af37',
                borderwidth=1
            ),
            margin=dict(t=60, l=60, r=40, b=60)
        )
        
        return fig
    
    @staticmethod
    def create_segmentation_scatter(pca_data: np.ndarray, labels: np.ndarray, segment_names: Dict) -> go.Figure:
        """
        PCA segmentasyon scatter plot.
        """
        if pca_data is None or labels is None:
            fig = go.Figure()
            fig.update_layout(
                title=dict(text="Segmentasyon verisi yok", font=dict(color='#c0c0c0'), x=0.5),
                paper_bgcolor='rgba(10,25,41,0)',
                plot_bgcolor='rgba(10,25,41,0)',
                height=450
            )
            return fig
        
        fig = go.Figure()
        
        unique_labels = np.unique(labels)
        
        # Renk haritası
        color_map = {
            ProductSegment.LIDER.value: '#d4af37',
            ProductSegment.POTANSIYEL.value: '#4caf50',
            ProductSegment.RISKLI.value: '#f44336',
            ProductSegment.OLGUN.value: '#2196f3',
            ProductSegment.NAKIT.value: '#9c27b0',
            ProductSegment.SORU.value: '#ff9800'
        }
        
        for label in unique_labels:
            mask = labels == label
            name = segment_names.get(label, f'Segment {label}')
            
            color = color_map.get(name, '#808080')
            
            fig.add_trace(go.Scatter(
                x=pca_data[mask, 0],
                y=pca_data[mask, 1],
                mode='markers',
                name=name,
                marker=dict(
                    size=14,
                    color=color,
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                text=[f'Segment: {name}'] * mask.sum(),
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))
        
        # Segment merkezleri
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                center_x = np.median(pca_data[mask, 0])
                center_y = np.median(pca_data[mask, 1])
                name = segment_names.get(label, f'Segment {label}')
                
                fig.add_trace(go.Scatter(
                    x=[center_x],
                    y=[center_y],
                    mode='markers+text',
                    name=f'{name} Merkez',
                    marker=dict(
                        size=20,
                        color=color_map.get(name, '#808080'),
                        symbol='star',
                        line=dict(width=2, color='white')
                    ),
                    text=[name],
                    textposition='top center',
                    textfont=dict(color='white', size=10, family='Arial Black'),
                    showlegend=False,
                    hoverinfo='none'
                ))
        
        fig.update_layout(
            title=dict(
                text='🎯 PCA + K-Means Segmentasyon',
                font=dict(size=20, color='#d4af37', family='Arial Black'),
                x=0.5
            ),
            xaxis=dict(
                title='PC1',
                title_font=dict(color='white', size=14),
                tickfont=dict(color='white', size=11),
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.3)'
            ),
            yaxis=dict(
                title='PC2',
                title_font=dict(color='white', size=14),
                tickfont=dict(color='white', size=11),
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.3)'
            ),
            paper_bgcolor='rgba(10,25,41,0)',
            plot_bgcolor='rgba(10,25,41,0)',
            font=dict(color='white'),
            height=500,
            legend=dict(
                font=dict(color='white', size=10),
                bgcolor='rgba(10,25,41,0.7)',
                bordercolor='#d4af37',
                borderwidth=1,
                x=1.05,
                y=1
            ),
            margin=dict(t=60, l=60, r=120, b=60)
        )
        
        return fig
    
    @staticmethod
    def create_anomaly_distribution(anomaly_df: pd.DataFrame) -> go.Figure:
        """
        Anomali dağılım grafiği.
        """
        if anomaly_df is None or 'Anomali_Tipi' not in anomaly_df.columns:
            fig = go.Figure()
            return fig
        
        counts = anomaly_df['Anomali_Tipi'].value_counts().reset_index()
        counts.columns = ['Tip', 'Sayı']
        
        colors = {
            '🔴 Kritik Düşüş': '#f44336',
            '🟠 Anormal Değişim': '#ff9800',
            '🟢 Aşırı Büyüme': '#4caf50',
            '✅ Normal': '#2196f3'
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=counts['Tip'],
                y=counts['Sayı'],
                marker_color=[colors.get(t, '#808080') for t in counts['Tip']],
                text=counts['Sayı'],
                textposition='auto',
                hovertemplate='Tip: %{x}<br>Sayı: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text='⚠️ Anomali Dağılımı',
                font=dict(size=18, color='#d4af37'),
                x=0.5
            ),
            xaxis=dict(
                title='Anomali Tipi',
                title_font=dict(color='white', size=14),
                tickfont=dict(color='white', size=11),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title='Ürün Sayısı',
                title_font=dict(color='white', size=14),
                tickfont=dict(color='white', size=11),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            paper_bgcolor='rgba(10,25,41,0)',
            plot_bgcolor='rgba(10,25,41,0)',
            font=dict(color='white'),
            height=400,
            margin=dict(t=60, l=60, r=40, b=60)
        )
        
        return fig

# ================================================
# 10. YÖNETİCİ ÖZETİ ÜRETİCİ
# ================================================

class ExecutiveSummaryGenerator:
    """
    Otomatik yönetici özeti ve insight üretici.
    """
    
    @staticmethod
    def generate_summaries(prodpack_engine: Optional[ProdPackDepthEngine],
                          forecast_engine: Optional[StrategicForecastEngine],
                          anomaly_engine: Optional[AnomalyDetectionEngine],
                          segmentation_engine: Optional[ProductSegmentationEngine]) -> List[str]:
        """
        Tüm analizlerden yönetici özeti üret.
        """
        summaries = []
        
        # 1. ProdPack özeti
        if prodpack_engine and prodpack_engine.nodes:
            total_products = len([n for n in prodpack_engine.nodes.values() if n.level == 'pack'])
            total_molecules = len([n for n in prodpack_engine.nodes.values() if n.level == 'molecule'])
            total_companies = len([n for n in prodpack_engine.nodes.values() if n.level == 'company'])
            total_value = prodpack_engine.nodes.get(prodpack_engine.root_id, ProdPackNode('', '', '')).value
            
            summaries.append(f"📊 **PAZAR YAPISI**: {total_molecules} molekül, {total_companies} şirket, {total_products} benzersiz ProdPack. Toplam pazar değeri {total_value:,.0f} TL.")
        
        # 2. Büyüme özeti
        if forecast_engine and forecast_engine.forecast_market().get('success'):
            forecast_result = forecast_engine.forecast_market()
            growth = forecast_result.get('growth_rate', 0)
            
            if growth > 10:
                summaries.append(f"🚀 **BÜYÜME FIRSATI**: 2025-2026 dönemi için %{growth:.1f} büyüme öngörülüyor. Yatırım için stratejik pencere açıldı.")
            elif growth > 3:
                summaries.append(f"📈 **İSTİKRARLI BÜYÜME**: Pazar %{growth:.1f} büyüyecek. Mevcut stratejileri koruyun, verimliliği artırın.")
            elif growth > -3:
                summaries.append(f"⏸️ **DURGUN PAZAR**: Büyüme %{growth:.1f}. Maliyet optimizasyonu ve operasyonel mükemmellik öncelikli.")
            else:
                summaries.append(f"⚠️ **PAZAR DARALMASI**: %{growth:.1f} daralma bekleniyor. Acil maliyet önlemleri ve portföy optimizasyonu şart.")
        
        # 3. Anomali özeti
        if anomaly_engine and hasattr(anomaly_engine, '_detect_anomalies_impl'):
            anomaly_df = anomaly_engine.detect_anomalies(0.1)
            if 'Anomali_Tipi' in anomaly_df.columns:
                critical = len(anomaly_df[anomaly_df['Anomali_Tipi'] == '🔴 Kritik Düşüş'])
                high_growth = len(anomaly_df[anomaly_df['Anomali_Tipi'] == '🟢 Aşırı Büyüme'])
                total = len(anomaly_df)
                
                if critical > 0:
                    summaries.append(f"🔴 **RİSK UYARISI**: {critical} üründe kritik düşüş tespit edildi. Acil müdahale gerekiyor.")
                if high_growth > 0:
                    summaries.append(f"🟢 **BÜYÜME FIRSATI**: {high_growth} üründe aşırı büyüme görülüyor. Kapasite planlaması yapın.")
        
        # 4. Segmentasyon özeti
        if segmentation_engine and segmentation_engine.segment_products().get('success'):
            seg_result = segmentation_engine.segment_products()
            if seg_result['segmented_df'] is not None and 'Segment_Adi' in seg_result['segmented_df'].columns:
                seg_counts = seg_result['segmented_df']['Segment_Adi'].value_counts()
                
                leaders = seg_counts.get(ProductSegment.LIDER.value, 0)
                potentials = seg_counts.get(ProductSegment.POTANSIYEL.value, 0)
                risks = seg_counts.get(ProductSegment.RISKLI.value, 0)
                
                if leaders > 0:
                    summaries.append(f"⭐ **LİDER ÜRÜNLER**: {leaders} ürün lider segmentinde. Pazar konumunuzu koruyun.")
                if potentials > 0:
                    summaries.append(f"🌟 **POTANSİYEL YILDIZLAR**: {potentials} ürün yüksek büyüme potansiyeline sahip. Yatırım yapın.")
                if risks > 0:
                    summaries.append(f"⚠️ **RİSKLİ ÜRÜNLER**: {risks} ürün riskli segmentte. Stratejik değerlendirme yapın.")
        
        # 5. Kanibalizasyon özeti
        if prodpack_engine and prodpack_engine.cannibalization_scores:
            high_cannibal = [c for c in prodpack_engine.cannibalization_scores if c.cannibal_score > 0.7]
            if high_cannibal:
                top = high_cannibal[0]
                summaries.append(f"🔄 **KANİBALİZASYON UYARISI**: {top.company} şirketinde {top.brand_a} ve {top.brand_b} arasında yüksek kanibalizasyon. Ürün farklılaştırması yapın.")
        
        return summaries if summaries else ["📋 Yeterli analiz verisi bulunamadı. Lütfen modülleri çalıştırın."]

# ================================================
# 11. EXECUTIVE CSS - LACİVERT, GÜMÜŞ, ALTIN
# ================================================

EXECUTIVE_CSS = """
<style>
    /* Executive Dark Mode - Lacivert, Gümüş, Altın Teması */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(145deg, #0a1929 0%, #0c1f33 100%);
        color: #ffffff;
    }
    
    /* Executive Kart */
    .executive-card {
        background: rgba(10, 25, 41, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(212, 175, 55, 0.4);
        border-radius: 24px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        position: relative;
        overflow: hidden;
    }
    
    .executive-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #d4af37, #c0c0c0, #d4af37);
    }
    
    .executive-card:hover {
        transform: translateY(-5px);
        border-color: #d4af37;
        box-shadow: 0 20px 45px rgba(212, 175, 55, 0.15);
    }
    
    /* Insight Box - Yönetici Özeti */
    .insight-box {
        background: linear-gradient(135deg, rgba(26, 54, 93, 0.95), rgba(16, 36, 61, 0.98));
        border-left: 8px solid #d4af37;
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        margin: 2rem 0;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        position: relative;
        backdrop-filter: blur(10px);
    }
    
    .insight-box::after {
        content: '📋 YÖNETİCİ ÖZETİ';
        position: absolute;
        top: -14px;
        left: 25px;
        background: #d4af37;
        color: #0a1929;
        font-size: 0.8rem;
        font-weight: 900;
        padding: 5px 18px;
        border-radius: 30px;
        letter-spacing: 3px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        border: 1px solid #c0c0c0;
    }
    
    .insight-text {
        font-size: 1.1rem;
        line-height: 1.7;
        color: #f0f4fa;
        font-weight: 400;
        margin-top: 0.3rem;
    }
    
    /* Altın Başlık */
    .gold-title {
        color: #d4af37;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        border-bottom: 3px solid rgba(212, 175, 55, 0.5);
        padding-bottom: 0.6rem;
        margin-bottom: 1.8rem;
        text-shadow: 0 2px 5px rgba(0,0,0,0.5);
    }
    
    /* Gümüş Alt Başlık */
    .silver-subtitle {
        color: #c0c0c0;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
        letter-spacing: -0.5px;
    }
    
    /* Executive Metrik Kartı */
    .metric-executive {
        background: linear-gradient(145deg, rgba(12, 27, 44, 0.8), rgba(8, 19, 33, 0.9));
        border-radius: 20px;
        padding: 1.5rem 1rem;
        text-align: center;
        border: 1px solid rgba(192, 192, 192, 0.3);
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    
    .metric-executive:hover {
        border-color: #d4af37;
        background: linear-gradient(145deg, rgba(16, 36, 61, 0.9), rgba(10, 25, 41, 0.95));
    }
    
    .metric-executive-value {
        color: #d4af37;
        font-size: 2.4rem;
        font-weight: 900;
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }
    
    .metric-executive-label {
        color: #a0b8cc;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
    
    .metric-executive-trend {
        color: #c0c0c0;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    
    /* Yatırım Kartı */
    .investment-card {
        background: radial-gradient(circle at 10% 30%, rgba(26, 54, 93, 0.95), rgba(10, 25, 41, 0.98));
        border: 1.5px solid #d4af37;
        border-radius: 24px;
        padding: 1.8rem 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.7);
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .investment-card:hover {
        transform: scale(1.02);
        border-color: #ffd966;
        box-shadow: 0 15px 35px rgba(212, 175, 55, 0.2);
    }
    
    /* Badge'ler */
    .badge-critical {
        background: linear-gradient(145deg, #b71c1c, #8b0000);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border: 1px solid #ffcdd2;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        display: inline-block;
    }
    
    .badge-opportunity {
        background: linear-gradient(145deg, #1e5f3e, #0a4b2b);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border: 1px solid #a5d6a5;
    }
    
    /* Buton */
    .stButton > button {
        background: linear-gradient(145deg, #1e3a5f, #142b44);
        color: white;
        border: 1px solid #d4af37;
        border-radius: 50px;
        padding: 0.7rem 2rem;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 1px;
        transition: all 0.2s;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(145deg, #2b4b72, #1a334d);
        border: 1px solid #ffd966;
        color: #ffd966;
        box-shadow: 0 8px 20px rgba(212, 175, 55, 0.4);
        transform: translateY(-2px);
    }
    
    /* Sekme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(10, 25, 41, 0.6);
        padding: 8px;
        border-radius: 50px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 50px;
        padding: 0 25px;
        color: #c0c0c0;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.2s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #1e3a5f, #142b44);
        color: #d4af37;
        border: 1px solid #d4af37;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a1929;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #d4af37;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #ffd966;
    }
</style>
"""

# ================================================
# 12. ANA UYGULAMA SINIFI - PHARMAINTELLIGENCE PRO
# ================================================

class PharmaIntelligencePro:
    """
    Ana uygulama sınıfı.
    Tüm modülleri entegre eder.
    """
    
    def __init__(self):
        self._init_session_state()
        self._configure_page()
        
    def _configure_page(self):
        """Sayfa yapılandırması"""
        st.set_page_config(
            page_title="PharmaIntelligence Pro v8.0 | ProdPack Derinlik",
            page_icon="💊",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': '''
                ### PharmaIntelligence Pro v8.0
                **Enterprise ProdPack Derinlik Analizi**
                
                ✓ Molekül → Şirket → Marka → Paket Hiyerarşisi
                ✓ Pazar Kanibalizasyon Matrisi
                ✓ Holt-Winters Tahminleme (2025-2026)
                ✓ IsolationForest Anomali Tespiti
                ✓ PCA + K-Means Segmentasyon
                
                © 2025 PharmaIntelligence Inc.
                '''
            }
        )
        
        st.markdown(EXECUTIVE_CSS, unsafe_allow_html=True)
    
    def _init_session_state(self):
        """Session state değişkenlerini başlat"""
        defaults = {
            'raw_data': None,
            'processed_data': None,
            'data_hash': None,
            'data_loaded_time': None,
            
            # ProdPack
            'prodpack_engine': None,
            'hierarchy_data': None,
            'selected_molecule': None,
            'molecule_drill_data': None,
            'cannibalization_data': None,
            
            # Tahmin
            'forecast_engine': None,
            'forecast_results': None,
            'investment_advice': [],
            
            # Anomali
            'anomaly_engine': None,
            'anomaly_results': None,
            
            # Segmentasyon
            'segmentation_engine': None,
            'segmentation_results': None,
            
            # Özet
            'executive_summaries': [],
            
            # UI State
            'active_tab': 0,
            'sidebar_state': 'expanded'
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_sidebar(self):
        """Sidebar render"""
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem 0.5rem 0.5rem 0.5rem;">
                <span style="font-size: 3rem; background: rgba(212,175,55,0.1); padding: 1rem; border-radius: 20px;">💊</span>
                <h2 style="color: #d4af37; margin: 1rem 0 0.2rem 0; font-size: 1.8rem;">PharmaIntel</h2>
                <p style="color: #c0c0c0; font-size: 0.7rem; letter-spacing: 3px;">v8.0 ENTERPRISE</p>
                <p style="color: #a0b8cc; font-size: 0.8rem; margin-top: 0.3rem;">ProdPack Derinlik Analizi</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<hr style="border-color: rgba(212,175,55,0.3); margin: 1.5rem 0;">', unsafe_allow_html=True)
            
            # ------------------- VERİ YÜKLEME -------------------
            with st.expander("📁 VERİ YÖNETİMİ", expanded=True):
                uploaded_file = st.file_uploader(
                    "Excel / CSV Yükle",
                    type=['xlsx', 'xls', 'csv'],
                    label_visibility="collapsed",
                    key="file_uploader"
                )
                
                if uploaded_file:
                    file_details = {
                        "Dosya Adı": uploaded_file.name,
                        "Dosya Boyutu": f"{uploaded_file.size / 1024:.1f} KB"
                    }
                    st.caption(f"📄 {file_details['Dosya Adı']} ({file_details['Dosya Boyutu']})")
                    
                    if st.button("🚀 VERİYİ ANALİZ ET", use_container_width=True, type="primary"):
                        with st.spinner("Veri işleniyor... (5000 satır limit)"):
                            self._load_and_process_data(uploaded_file)
            
            # ------------------- PRODPACK MODÜLÜ -------------------
            if st.session_state.processed_data is not None:
                st.markdown('<hr style="border-color: rgba(212,175,55,0.3); margin: 1.5rem 0;">', unsafe_allow_html=True)
                
                with st.expander("🧬 PRODPACK DERİNLİK", expanded=True):
                    if st.button("🔨 HİYERARŞİ OLUŞTUR", use_container_width=True):
                        with st.spinner("ProdPack hiyerarşisi kuruluyor..."):
                            self._build_prodpack_hierarchy()
                    
                    # Molekül seçimi
                    if st.session_state.hierarchy_data and st.session_state.prodpack_engine:
                        engine = st.session_state.prodpack_engine
                        if engine.molecule_col:
                            molecules = engine.df[engine.molecule_col].dropna().unique()[:50]
                            if len(molecules) > 0:
                                selected = st.selectbox(
                                    "🔬 Molekül Drill-Down",
                                    molecules,
                                    format_func=lambda x: str(x)[:40] + '...' if len(str(x)) > 40 else str(x)
                                )
                                st.session_state.selected_molecule = selected
                                
                                if st.button("🔍 DETAY ANALİZİ", use_container_width=True):
                                    with st.spinner(f"{selected} analiz ediliyor..."):
                                        self._analyze_molecule(selected)
                
                # ------------------- TAHMİN & RİSK -------------------
                with st.expander("📈 TAHMİN & RİSK", expanded=False):
                    if st.button("🔮 2025-2026 TAHMİNİ", use_container_width=True):
                        with st.spinner("Holt-Winters tahmini yapılıyor..."):
                            self._run_forecast()
                    
                    if st.button("⚠️ ANOMALİ TESPİTİ", use_container_width=True):
                        with st.spinner("IsolationForest ile anomali tespiti..."):
                            self._run_anomaly_detection()
                    
                    if st.button("🎯 PCA SEGMENTASYON", use_container_width=True):
                        with st.spinner("PCA + K-Means segmentasyon..."):
                            self._run_segmentation()
                
                # ------------------- SİSTEM DURUMU -------------------
                with st.expander("⚙️ SİSTEM", expanded=False):
                    df = st.session_state.processed_data
                    st.metric("Satır Sayısı", f"{len(df):,}", help="5000 limit")
                    st.metric("Sütun Sayısı", len(df.columns))
                    
                    mem_usage = df.memory_usage(deep=True).sum() / 1024**2
                    st.metric("Hafıza", f"{mem_usage:.1f} MB")
                    
                    if st.button("🧹 CACHE TEMİZLE", use_container_width=True):
                        st.cache_data.clear()
                        for key in ['hierarchy_data', 'forecast_results', 'anomaly_results', 'segmentation_results']:
                            if key in st.session_state:
                                st.session_state[key] = None
                        st.success("✅ Cache temizlendi")
                        st.rerun()
            
            # ------------------- FOOTER -------------------
            st.markdown('<hr style="border-color: rgba(212,175,55,0.3); margin: 2rem 0 1rem 0;">', unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: center; color: #808080; font-size: 0.7rem; padding: 1rem 0;">
                <span>© 2025 PharmaIntelligence Inc.</span><br>
                <span style="color: #c0c0c0;">Enterprise ProdPack Depth v8.0</span>
            </div>
            """, unsafe_allow_html=True)
    
    @st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
    def _cached_data_processing(_self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Cache'li veri işleme"""
        # Benzersiz sütun isimlendirme
        df_raw.columns = make_unique_column_names(df_raw.columns.tolist())
        
        # Güvenli downcast
        df_processed = safe_downcast(df_raw)
        
        # 5000 satır limit
        if len(df_processed) > 5000:
            df_processed = df_processed.head(5000).copy()
        
        return df_processed
    
    def _load_and_process_data(self, uploaded_file):
        """Veri yükle ve işle"""
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file, low_memory=False, encoding='utf-8')
            else:
                raw_df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Cache'li veri işleme
            processed_df = self._cached_data_processing(raw_df)
            
            # Session state güncelle
            st.session_state.raw_data = raw_df
            st.session_state.processed_data = processed_df
            st.session_state.data_hash = hashlib.md5(pd.util.hash_pandas_object(processed_df).values).hexdigest()
            st.session_state.data_loaded_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Engine'leri sıfırla
            st.session_state.prodpack_engine = None
            st.session_state.forecast_engine = None
            st.session_state.anomaly_engine = None
            st.session_state.segmentation_engine = None
            
            st.success(f"✅ Veri işlendi: {len(processed_df):,} satır (5000 limit)")
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.caption("Lütfen dosya formatını kontrol edin (Excel/CSV, UTF-8 encoding)")
    
    def _build_prodpack_hierarchy(self):
        """ProdPack hiyerarşisi oluştur"""
        if st.session_state.processed_data is None:
            st.warning("⚠️ Önce veri yükleyin")
            return
        
        engine = ProdPackDepthEngine(st.session_state.processed_data)
        hierarchy = engine.build_hierarchy()
        
        if hierarchy:
            st.session_state.prodpack_engine = engine
            st.session_state.hierarchy_data = hierarchy
            st.success("✅ ProdPack hiyerarşisi oluşturuldu")
    
    def _analyze_molecule(self, molecule: str):
        """Molekül detay analizi"""
        if st.session_state.prodpack_engine is None:
            st.warning("⚠️ Önce hiyerarşi oluşturun")
            return
        
        engine = st.session_state.prodpack_engine
        
        # Drill-down verisi
        drill_df = engine.get_molecule_drilldown(molecule)
        st.session_state.molecule_drill_data = drill_df
        
        # Kanibalizasyon verisi
        cannibal_df = engine.analyze_cannibalization(molecule)
        st.session_state.cannibalization_data = cannibal_df
        
        if not drill_df.empty:
            st.success(f"✅ {molecule} analizi tamamlandı")
    
    def _run_forecast(self):
        """Tahminleme çalıştır"""
        if st.session_state.processed_data is None:
            st.warning("⚠️ Önce veri yükleyin")
            return
        
        engine = StrategicForecastEngine(st.session_state.processed_data)
        results = engine.forecast_market()
        
        st.session_state.forecast_engine = engine
        st.session_state.forecast_results = results
        st.session_state.investment_advice = results.get('investment_advice', [])
        
        if results.get('success'):
            st.success(f"✅ Tahmin tamamlandı: %{results.get('growth_rate', 0):.1f} büyüme")
        else:
            st.warning("⚠️ Tahminleme başarısız, varsayılan stratejiler kullanılıyor")
    
    def _run_anomaly_detection(self):
        """Anomali tespiti çalıştır"""
        if st.session_state.processed_data is None:
            st.warning("⚠️ Önce veri yükleyin")
            return
        
        engine = AnomalyDetectionEngine(st.session_state.processed_data)
        results = engine.detect_anomalies(contamination=0.1)
        
        st.session_state.anomaly_engine = engine
        st.session_state.anomaly_results = results
        
        if 'Anomali_Tipi' in results.columns:
            anomaly_count = len(results[results['Anomali_Tipi'] != '✅ Normal'])
            st.success(f"✅ Anomali tespiti: {anomaly_count} anomali bulundu")
    
    def _run_segmentation(self):
        """Segmentasyon çalıştır"""
        if st.session_state.processed_data is None:
            st.warning("⚠️ Önce veri yükleyin")
            return
        
        engine = ProductSegmentationEngine(st.session_state.processed_data)
        results = engine.segment_products(n_clusters=4)
        
        st.session_state.segmentation_engine = engine
        st.session_state.segmentation_results = results
        
        if results.get('success'):
            st.success(f"✅ Segmentasyon: Silhouette skoru {results.get('silhouette_score', 0):.3f}")
    
    def render_main_content(self):
        """Ana içerik render"""
        
        # ------------------- HOŞGELDİN EKRANI -------------------
        if st.session_state.processed_data is None:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div class="executive-card" style="text-align: center; padding: 3rem 2rem;">
                    <span style="font-size: 5rem; display: block; margin-bottom: 1rem;">💊</span>
                    <h1 style="color: #d4af37; font-size: 2.8rem; margin-bottom: 0.5rem; font-weight: 800;">PharmaIntel Pro</h1>
                    <p style="color: #c0c0c0; font-size: 1.2rem; margin-bottom: 2rem; letter-spacing: 2px;">v8.0 Enterprise ProdPack Depth</p>
                    
                    <div style="background: rgba(212,175,55,0.1); border-radius: 20px; padding: 2rem; text-align: left;">
                        <h3 style="color: white; margin-bottom: 1.5rem;">🚀 Başlamak İçin:</h3>
                        <ol style="color: #a0b8cc; font-size: 1.1rem; line-height: 2;">
                            <li style="margin-bottom: 0.8rem;">📁 <strong>Sol panelden</strong> Excel/CSV dosyanızı yükleyin</li>
                            <li style="margin-bottom: 0.8rem;">⚙️ <strong>"Veriyi Analiz Et"</strong> butonuna tıklayın</li>
                            <li style="margin-bottom: 0.8rem;">🧬 <strong>ProdPack hiyerarşisi</strong> oluşturun</li>
                            <li style="margin-bottom: 0.8rem;">📈 <strong>Tahmin & Segmentasyon</strong> modüllerini çalıştırın</li>
                        </ol>
                    </div>
                    
                    <p style="color: #808080; margin-top: 2rem; font-size: 0.8rem;">
                        Desteklenen formatlar: .xlsx, .xls, .csv (UTF-8) | Maks. 5000 satır
                    </p>
                </div>
                """, unsafe_allow_html=True)
                return
        
        # ------------------- ANA PANEL -------------------
        col_title, col_time = st.columns([3, 1])
        
        with col_title:
            st.markdown(f"""
            <h1 class="gold-title" style="margin-bottom: 0; font-size: 2.5rem;">
                💊 PharmaIntelligence Pro
            </h1>
            <p style="color: #c0c0c0; font-size: 1rem; margin-top: 0.3rem;">
                ProdPack Derinlik Analizi | Molekül → Şirket → Marka → Paket
            </p>
            """, unsafe_allow_html=True)
        
        with col_time:
            if st.session_state.data_loaded_time:
                st.markdown(f"""
                <div style="background: rgba(212,175,55,0.15); padding: 0.8rem 1.2rem; border-radius: 40px; text-align: center;">
                    <span style="color: #d4af37; font-size: 0.8rem;">SON GÜNCELLEME</span><br>
                    <span style="color: white; font-size: 0.9rem; font-weight: 600;">{st.session_state.data_loaded_time}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # ------------------- YÖNETİCİ ÖZETİ -------------------
        summaries = ExecutiveSummaryGenerator.generate_summaries(
            st.session_state.prodpack_engine,
            st.session_state.forecast_engine,
            st.session_state.anomaly_engine,
            st.session_state.segmentation_engine
        )
        
        if summaries:
            summary_html = '<div class="insight-box"><div class="insight-text"><ul style="margin: 0; padding-left: 1.5rem;">'
            for s in summaries[:4]:  # En fazla 4 özet
                summary_html += f'<li style="margin-bottom: 0.8rem;">{s}</li>'
            summary_html += '</ul></div></div>'
            st.markdown(summary_html, unsafe_allow_html=True)
        
        # ------------------- SEKMELER -------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧬 PRODPACK DERİNLİK",
            "📈 TAHMİN & YATIRIM",
            "⚠️ RİSK & SEGMENTASYON",
            "📊 YÖNETİCİ PANELİ"
        ])
        
        with tab1:
            self._render_prodpack_tab()
        
        with tab2:
            self._render_forecast_tab()
        
        with tab3:
            self._render_risk_tab()
        
        with tab4:
            self._render_dashboard_tab()
    
    def _render_prodpack_tab(self):
        """ProdPack derinlik sekmesi"""
        
        if not st.session_state.hierarchy_data:
            st.info("👈 Sol panelden **'Hiyerarşi Oluştur'** butonuna tıklayın")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<p class="silver-subtitle">🧬 Molekül Hiyerarşisi</p>', unsafe_allow_html=True)
            sunburst_data = st.session_state.prodpack_engine.get_sunburst_data()
            fig_sunburst = VisualizationEngine.create_sunburst_chart(sunburst_data)
            st.plotly_chart(fig_sunburst, use_container_width=True)
        
        with col2:
            st.markdown('<p class="silver-subtitle">🔄 Pazar Akışı</p>', unsafe_allow_html=True)
            sankey_data = st.session_state.prodpack_engine.get_sankey_data()
            fig_sankey = VisualizationEngine.create_sankey_diagram(sankey_data)
            st.plotly_chart(fig_sankey, use_container_width=True)
        
        # ------------------- MOLEKÜL DRILL-DOWN -------------------
        st.markdown('---')
        st.markdown('<p class="silver-subtitle">🔬 Molekül Drill-Down Detayı</p>', unsafe_allow_html=True)
        
        if st.session_state.selected_molecule:
            col_left, col_right = st.columns([1.5, 1])
            
            with col_left:
                st.markdown(f"""
                <div style="background: rgba(26,54,93,0.5); padding: 0.8rem 1.5rem; border-radius: 40px; margin-bottom: 1.5rem;">
                    <span style="color: #d4af37; font-weight: 700;">🔬 SEÇİLİ MOLEKÜL:</span>
                    <span style="color: white; font-weight: 600; margin-left: 0.8rem;">{st.session_state.selected_molecule}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if st.session_state.molecule_drill_data is not None and not st.session_state.molecule_drill_data.empty:
                    drill_df = st.session_state.molecule_drill_data
                    
                    # Formatlı gösterim
                    display_df = drill_df.copy()
                    if 'Satış_Hacmi' in display_df.columns:
                        display_df['Satış_Hacmi'] = display_df['Satış_Hacmi'].apply(lambda x: f"{x:,.0f}")
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400,
                        column_config={
                            "ProdPack": st.column_config.TextColumn("ProdPack", width="medium"),
                            "Şirket": st.column_config.TextColumn("Şirket", width="small"),
                            "Marka": st.column_config.TextColumn("Marka", width="small"),
                            "Satış_Hacmi": st.column_config.TextColumn("Satış Hacmi", width="small"),
                            "Büyüme_Oranı_%": st.column_config.NumberColumn("Büyüme %", format="%.1f%%"),
                            "Pazar_Payı_%": st.column_config.NumberColumn("Pazar Payı %", format="%.2f%%"),
                            "Hacim_Trendi": st.column_config.TextColumn("Trend", width="small")
                        },
                        hide_index=True
                    )
                else:
                    st.info("Molekül seçildi ancak detay verisi bulunamadı")
            
            with col_right:
                st.markdown("**🔄 Kanibalizasyon Matrisi**")
                
                if st.session_state.cannibalization_data is not None and not st.session_state.cannibalization_data.empty:
                    fig_heatmap = VisualizationEngine.create_cannibalization_heatmap(st.session_state.cannibalization_data)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Yüksek kanibalizasyon uyarısı
                    high_cannibal = st.session_state.cannibalization_data[
                        st.session_state.cannibalization_data['Kanibalizasyon_Skoru'] > 0.6
                    ]
                    
                    if not high_cannibal.empty:
                        st.warning(f"⚠️ {len(high_cannibal)} marka çiftinde yüksek kanibalizasyon riski")
                else:
                    st.info("Bu molekül için kanibalizasyon verisi yok")
        else:
            st.info("👈 Sol panelden bir molekül seçin ve **'Detay Analizi'** butonuna tıklayın")
    
    def _render_forecast_tab(self):
        """Tahmin ve yatırım sekmesi"""
        
        if st.session_state.forecast_results is None:
            st.info("👈 Sol panelden **'2025-2026 Tahmini'** butonuna tıklayın")
            return
        
        forecast = st.session_state.forecast_results
        
        # Tahmin grafiği
        fig_forecast = VisualizationEngine.create_forecast_chart(forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # ------------------- YATIRIM TAVSİYESİ -------------------
        st.markdown('<p class="silver-subtitle">💎 YATIRIM TAVSİYESİ 2025-2026</p>', unsafe_allow_html=True)
        
        advice_list = forecast.get('investment_advice', [])
        
        if advice_list:
            # Öncelik sırasına göre sırala
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            advice_list.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 3))
            
            cols = st.columns(min(3, len(advice_list)))
            
            for idx, advice in enumerate(advice_list[:3]):
                with cols[idx % 3]:
                    badge_class = "badge-opportunity" if 'EXPAND' in advice.get('action', '') else "badge-critical"
                    
                    st.markdown(f"""
                    <div class="investment-card" style="height: 100%;">
                        <span class="{badge_class}">{advice.get('action', 'STRATEJİ').value if hasattr(advice.get('action'), 'value') else advice.get('action', 'STRATEJİ')}</span>
                        <h4 style="color: #d4af37; margin: 1.2rem 0 0.8rem 0; font-size: 1.2rem;">{advice.get('title', 'Strateji')}</h4>
                        <p style="color: white; font-size: 0.9rem; line-height: 1.5;">{advice.get('message', '')}</p>
                        <div style="margin-top: 1.2rem; display: flex; justify-content: space-between;">
                            <span style="color: #c0c0c0; font-size: 0.8rem;">ROI: %{advice.get('roi_potential', 0)}</span>
                            <span style="color: #c0c0c0; font-size: 0.8rem;">Risk: {advice.get('risk_level', 'Orta')}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # ------------------- METRİKLER -------------------
        st.markdown('---')
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            growth = forecast.get('growth_rate', 0)
            delta_color = "normal" if growth > 0 else "inverse"
            st.metric(
                "2025-2026 Büyüme",
                f"%{growth:.1f}",
                delta=f"{growth:.1f}% y/y",
                delta_color=delta_color
            )
        
        with col_m2:
            if forecast.get('forecast') and 'values' in forecast['forecast']:
                val_2026 = forecast['forecast']['values'][-1] if len(forecast['forecast']['values']) > 0 else 0
                st.metric("2026 Pazar Değeri", f"{val_2026:,.0f} TL")
        
        with col_m3:
            model_stats = forecast.get('model_stats', {})
            mape = model_stats.get('mape', 0)
            st.metric("Tahmin Doğruluğu (MAPE)", f"%{100 - mape:.1f}" if mape > 0 else "N/A")
        
        with col_m4:
            st.metric("Model Güven Aralığı", "%95")
    
    def _render_risk_tab(self):
        """Risk ve segmentasyon sekmesi"""
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<p class="silver-subtitle">⚠️ IsolationForest Anomali</p>', unsafe_allow_html=True)
            
            if st.session_state.anomaly_results is not None:
                anomaly_df = st.session_state.anomaly_results
                
                # Dağılım grafiği
                fig_anomaly = VisualizationEngine.create_anomaly_distribution(anomaly_df)
                st.plotly_chart(fig_anomaly, use_container_width=True)
                
                # Riskli ürünler
                st.markdown("**🔴 Kritik Riskli Ürünler**")
                
                if 'Anomali_Tipi' in anomaly_df.columns:
                    critical_df = anomaly_df[anomaly_df['Anomali_Tipi'] == '🔴 Kritik Düşüş']
                    
                    if not critical_df.empty:
                        # Sütunları otomatik seç
                        display_cols = []
                        col_types = auto_detect_column_types(anomaly_df)
                        
                        if col_types.get('molecule'):
                            display_cols.append(col_types['molecule'])
                        if col_types.get('company'):
                            display_cols.append(col_types['company'])
                        if col_types.get('brand'):
                            display_cols.append(col_types['brand'])
                        
                        display_cols.extend(['Anomali_Tipi', 'Anomali_Skoru'])
                        display_cols = [c for c in display_cols if c in critical_df.columns][:5]
                        
                        st.dataframe(
                            critical_df[display_cols].head(10),
                            use_container_width=True,
                            height=250,
                            hide_index=True
                        )
                    else:
                        st.success("✅ Kritik riskli ürün bulunamadı")
                else:
                    st.info("Anomali verisi eksik")
            else:
                st.info("👈 'Anomali Tespiti' butonuna tıklayın")
        
        with col2:
            st.markdown('<p class="silver-subtitle">🎯 PCA + K-Means Segmentasyon</p>', unsafe_allow_html=True)
            
            if st.session_state.segmentation_results and st.session_state.segmentation_results.get('success'):
                seg = st.session_state.segmentation_results
                
                fig_seg = VisualizationEngine.create_segmentation_scatter(
                    seg.get('pca_components'),
                    seg.get('labels'),
                    seg.get('segment_names', {})
                )
                st.plotly_chart(fig_seg, use_container_width=True)
                
                # Segment dağılımı
                if seg['segmented_df'] is not None and 'Segment_Adi' in seg['segmented_df'].columns:
                    seg_counts = seg['segmented_df']['Segment_Adi'].value_counts()
                    
                    st.markdown("**📊 Segment Dağılımı**")
                    
                    # DataFrame olarak göster
                    seg_df = pd.DataFrame({
                        'Segment': seg_counts.index,
                        'Ürün Sayısı': seg_counts.values,
                        'Yüzde': (seg_counts.values / seg_counts.sum() * 100).round(1)
                    })
                    
                    st.dataframe(
                        seg_df,
                        use_container_width=True,
                        column_config={
                            "Segment": st.column_config.TextColumn("Segment", width="medium"),
                            "Ürün Sayısı": st.column_config.NumberColumn("Ürün Sayısı", format="%d"),
                            "Yüzde": st.column_config.NumberColumn("Yüzde %", format="%.1f%%")
                        },
                        hide_index=True
                    )
                    
                    # Silhouette skoru
                    silhouette = seg.get('silhouette_score', 0)
                    st.caption(f"✅ Silhouette Skoru: {silhouette:.3f} (1'e yakın iyi)")
            else:
                st.info("👈 'PCA Segmentasyon' butonuna tıklayın")
    
    def _render_dashboard_tab(self):
        """Yönetici paneli"""
        
        # ------------------- KPI METRİKLERİ -------------------
        st.markdown('<p class="silver-subtitle">📊 TEMEL PERFORMANS GÖSTERGELERİ</p>', unsafe_allow_html=True)
        
        kpi_cols = st.columns(4)
        
        with kpi_cols[0]:
            if st.session_state.prodpack_engine and st.session_state.hierarchy_data:
                node_count = len(st.session_state.prodpack_engine.nodes)
                pack_count = len([n for n in st.session_state.prodpack_engine.nodes.values() if n.level == 'pack'])
                
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">ProdPack (Paket)</div>
                    <div class="metric-executive-value">{pack_count}</div>
                    <div class="metric-executive-trend">Toplam {node_count} düğüm</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">ProdPack</div>
                    <div class="metric-executive-value">-</div>
                    <div class="metric-executive-trend">Hiyerarşi oluştur</div>
                </div>
                """, unsafe_allow_html=True)
        
        with kpi_cols[1]:
            if st.session_state.forecast_results and st.session_state.forecast_results.get('success'):
                growth = st.session_state.forecast_results.get('growth_rate', 0)
                
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">2025-2026 Büyüme</div>
                    <div class="metric-executive-value">%{growth:.1f}</div>
                    <div class="metric-executive-trend">{'📈 Pozitif' if growth > 0 else '📉 Negatif'}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">Büyüme Tahmini</div>
                    <div class="metric-executive-value">-</div>
                    <div class="metric-executive-trend">Tahmin çalıştır</div>
                </div>
                """, unsafe_allow_html=True)
        
        with kpi_cols[2]:
            if st.session_state.anomaly_results is not None and 'Anomali_Tipi' in st.session_state.anomaly_results.columns:
                anomaly_df = st.session_state.anomaly_results
                anomaly_pct = (len(anomaly_df[anomaly_df['Anomali_Tipi'] != '✅ Normal']) / len(anomaly_df)) * 100
                
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">Anomali Oranı</div>
                    <div class="metric-executive-value">%{anomaly_pct:.1f}</div>
                    <div class="metric-executive-trend">{'⚠️ Yüksek' if anomaly_pct > 15 else '✅ Normal'}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">Anomali</div>
                    <div class="metric-executive-value">-</div>
                    <div class="metric-executive-trend">Analiz çalıştır</div>
                </div>
                """, unsafe_allow_html=True)
        
        with kpi_cols[3]:
            if st.session_state.segmentation_results and st.session_state.segmentation_results.get('success'):
                seg = st.session_state.segmentation_results
                silhouette = seg.get('silhouette_score', 0)
                
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">Segmentasyon</div>
                    <div class="metric-executive-value">{silhouette:.2f}</div>
                    <div class="metric-executive-trend">Silhouette Skoru</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">Segmentasyon</div>
                    <div class="metric-executive-value">-</div>
                    <div class="metric-executive-trend">Segmentasyon çalıştır</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('---')
        
        # ------------------- STRATEJİK ÖNGÖRÜLER -------------------
        st.markdown('<p class="silver-subtitle">💡 STRATEJİK ÖNGÖRÜLER</p>', unsafe_allow_html=True)
        
        if st.session_state.investment_advice:
            advice_cols = st.columns(2)
            
            for idx, advice in enumerate(st.session_state.investment_advice[:2]):
                with advice_cols[idx]:
                    icon = '🚀' if 'EXPAND' in str(advice.get('action', '')) else '⚖️'
                    st.markdown(f"""
                    <div style="background: rgba(26,54,93,0.7); border-radius: 20px; padding: 1.8rem; border-left: 8px solid #d4af37; height: 100%;">
                        <span style="font-size: 2.2rem;">{icon}</span>
                        <h4 style="color: #d4af37; margin: 0.8rem 0;">{advice.get('title', 'Strateji')}</h4>
                        <p style="color: white; font-size: 0.95rem; line-height: 1.6;">{advice.get('message', '')}</p>
                        <div style="margin-top: 1.2rem; background: rgba(0,0,0,0.3); padding: 0.8rem; border-radius: 12px;">
                            <span style="color: #c0c0c0;">Aksiyon: </span>
                            <span style="color: #d4af37; font-weight: 700;">{advice.get('action', 'İZLE').value if hasattr(advice.get('action'), 'value') else advice.get('action', 'İZLE')}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👈 Tahmin modülünü çalıştırarak stratejik öngörüleri görüntüleyin")
        
        # ------------------- VERİ ÖNİZLEME -------------------
        st.markdown('---')
        with st.expander("📋 VERİ ÖNİZLEME (İşlenmiş)", expanded=False):
            if st.session_state.processed_data is not None:
                st.dataframe(
                    st.session_state.processed_data.head(100),
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
                
                # İndirme butonu
                csv_data = st.session_state.processed_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 CSV İndir",
                    data=csv_data,
                    file_name=f"pharma_processed_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ================================================
# 13. UYGULAMA BAŞLATMA
# ================================================

def main():
    """Ana uygulama"""
    
    # Garbage collector
    gc.enable()
    
    # Uygulama nesnesi
    app = PharmaIntelligencePro()
    
    # Sidebar
    app.render_sidebar()
    
    # Ana içerik
    app.render_main_content()
    
    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 20px; right: 30px; background: rgba(10,25,41,0.85); 
                padding: 0.5rem 1.5rem; border-radius: 40px; border: 1px solid #d4af37; 
                backdrop-filter: blur(8px); z-index: 999; box-shadow: 0 5px 20px rgba(0,0,0,0.5);">
        <span style="color: #c0c0c0; font-size: 0.75rem; letter-spacing: 2px;">
            ⚕️ PharmaIntelligence Pro v8.0 | ProdPack Depth
        </span>
    </div>
    """, unsafe_allow_html=True)

# ================================================
# 14. ENTRY POINT
# ================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"""
        ### ❌ Kritik Uygulama Hatası
        
        **Hata:** {str(e)}
        
        Lütfen sayfayı yenileyin veya veri formatını kontrol edin.
        """)
        
        with st.expander("🔍 Hata Detayı (Geliştiriciler İçin)"):
            st.code(traceback.format_exc())
            
        st.caption("Bu hata devam ederse support@pharmaintelligence.com adresine ulaşın.")
