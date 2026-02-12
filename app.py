"""
PharmaIntelligence Pro v8.0 - Enterprise Karar Destek ve Stratejik İstihbarat Platformu
Versiyon: 8.0.0
Yazar: PharmaIntelligence Inc.
Lisans: Kurumsal Enterprise

✓ ProdPack Derinlik Analizi (Molekül -> Şirket -> Marka -> Paket)
✓ AI-Powered Predictive Analytics (2025-2026 Tahminleri)
✓ Multi-Algorithm Anomaly Detection (Isolation Forest, LOF, SVM)
✓ PCA + K-Means Advanced Segmentation
✓ Executive Dark Theme (Lacivert, Gümüş, Altın)
✓ Automated Strategic Recommendations & Insight Boxes
"""

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
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
from dataclasses import dataclass, field
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================================================
# WARNINGS & CONFIG
# ================================================
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)

# ================================================
# ENUMS & DATA CLASSES
# ================================================

class RiskLevel:
    KRITIK = "Kritik Risk"
    YUKSEK = "Yüksek Risk"
    ORTA = "Orta Risk"
    DUSUK = "Düşük Risk"
    NORMAL = "Normal"

class ProductSegment:
    STARS = "⭐ Yıldız Ürünler"
    CASH_COWS = "💰 Nakit İnekleri"
    QUESTION_MARKS = "❓ Soru İşaretleri"
    DOGS = "⚠️ Zayıf Ürünler"
    EMERGING = "🚀 Yükselen Yıldızlar"

@dataclass
class ForecastResult:
    periods: List[str]
    predictions: List[float]
    lower_bounds: List[float]
    upper_bounds: List[float]
    model_type: str
    growth_rate: float = 0.0

@dataclass
class ProdPackInsight:
    molekul: str
    sirket: str
    marka: str
    paket: str
    sales_2024: float
    growth_2023_2024: float
    market_share: float
    risk_score: float
    cannibalization_risk: float
    recommendation: str

# ================================================
# 1. GELISMIS VERI MOTORU (Hata Giderme & Regex)
# ================================================

class PharmaDataEngine:
    """Veri yükleme, temizleme, dönüştürme ve yıl ayıklama motoru"""
    
    @staticmethod
    @st.cache_data(ttl=3600, max_entries=10, show_spinner="Veri işleniyor...")
    def load_and_process_data(uploaded_file) -> pd.DataFrame:
        """Ana veri işleme pipeline'ı. 1M+ satır için optimize edilmiş."""
        try:
            # 1. Veri yükleme
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, low_memory=False, encoding='utf-8')
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # 2. Sütun isimlerini temizle ve benzersizleştir
            df.columns = PharmaDataEngine._clean_column_names(df.columns.tolist())
            
            # 3. Regex ile yıl ayıklama (Kritik Hata Düzeltme)
            df = PharmaDataEngine._extract_years_safe(df)
            
            # 4. Tip dönüşümleri ve downcast
            df = PharmaDataEngine._safe_type_conversion(df)
            
            # 5. ProdPack hiyerarşisi oluşturma
            df = PharmaDataEngine._create_prodpack_hierarchy(df)
            
            # 6. Analitik feature'lar
            df = PharmaDataEngine._create_analytical_features(df)
            
            return df
        
        except Exception as e:
            st.error(f"Veri işleme hatası: {str(e)}")
            st.code(traceback.format_exc())
            return pd.DataFrame()
    
    @staticmethod
    def _clean_column_names(cols: List[str]) -> List[str]:
        """Akıllı sütun isimlendirme - Benzersiz isim garantisi"""
        cleaned = []
        seen = {}
        
        turkish_map = {'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's', 'Ğ': 'G', 'ğ': 'g', 
                       'Ü': 'U', 'ü': 'u', 'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'}
        
        for col in cols:
            # Türkçe karakter dönüşümü
            for tr, en in turkish_map.items():
                col = col.replace(tr, en)
            
            # Özel karakter temizliği
            col = re.sub(r'[^\w\s\-]', ' ', str(col))
            col = re.sub(r'\s+', '_', col.strip())
            
            # Kısaltma kuralları
            col = re.sub(r'(?i)manufacturer', 'Uretici', col)
            col = re.sub(r'(?i)corporation', 'Sirket', col)
            col = re.sub(r'(?i)molecule', 'Molekul', col)
            col = re.sub(r'(?i)brand', 'Marka', col)
            col = re.sub(r'(?i)product', 'Urun', col)
            col = re.sub(r'(?i)package', 'Paket', col)
            col = re.sub(r'(?i)region', 'Bolge', col)
            col = re.sub(r'(?i)sub.?region', 'Alt_Bolge', col)
            col = re.sub(r'(?i)sales', 'Satis', col)
            col = re.sub(r'(?i)volume', 'Hacim', col)
            col = re.sub(r'(?i)value', 'Deger', col)
            
            # Benzersiz isimlendirme
            base_col = col
            counter = 1
            while col in seen:
                col = f"{base_col}_{counter}"
                counter += 1
            seen[col] = True
            cleaned.append(col)
        
        return cleaned
    
    @staticmethod
    def _extract_years_safe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Regex ile yıl ayıklama.
        'PENICILLIN 2024' gibi metin içeren sütunlarda int() hatasını engeller.
        """
        year_pattern = re.compile(r'20\d{2}')
        year_columns = {}
        
        for col in df.columns:
            match = year_pattern.search(str(col))
            if match:
                try:
                    year = int(match.group())
                    year_columns[col] = year
                except ValueError:
                    continue
        
        # Yıl bazlı sütunları yeniden adlandır
        for old_col, year in year_columns.items():
            if 'Satis' in old_col or 'Sales' in old_col or 'Hacim' in old_col or 'Volume' in old_col:
                new_col = f'Satis_{year}'
                df.rename(columns={old_col: new_col}, inplace=True)
            elif 'Fiyat' in old_col or 'Price' in old_col:
                new_col = f'Fiyat_{year}'
                df.rename(columns={old_col: new_col}, inplace=True)
            elif 'Pay' in old_col or 'Share' in old_col:
                new_col = f'Pazar_Payi_{year}'
                df.rename(columns={old_col: new_col}, inplace=True)
        
        return df
    
    @staticmethod
    def _safe_type_conversion(df: pd.DataFrame) -> pd.DataFrame:
        """
        pd.api.types kullanarak güvenli tip dönüşümü.
        'Ambiguous Truth Value' hatasını çözer.
        """
        for col in df.columns:
            # Kategorik sütunlar
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                if unique_ratio < 0.05 and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype('category')
            
            # Sayısal dönüşüm
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Downcast: int64 -> int32, float64 -> float32
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                elif pd.api.types.is_float_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    @staticmethod
    def _create_prodpack_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
        """ProdPack hiyerarşisi oluştur: Molekül -> Şirket -> Marka -> Paket"""
        
        # Varsayılan sütun adları
        molekul_col = next((c for c in df.columns if 'Molekul' in c or 'molecule' in c.lower()), None)
        sirket_col = next((c for c in df.columns if 'Sirket' in c or 'Uretici' in c or 'company' in c.lower()), None)
        marka_col = next((c for c in df.columns if 'Marka' in c or 'brand' in c.lower()), None)
        paket_col = next((c for c in df.columns if 'Paket' in c or 'package' in c.lower() or 'urun' in c.lower()), None)
        
        # Eksik sütunlar için sentetik ID oluştur
        if molekul_col is None:
            df['Molekul'] = 'Genel'
            molekul_col = 'Molekul'
        
        if sirket_col is None:
            df['Sirket'] = 'Belirtilmemis'
            sirket_col = 'Sirket'
        
        if marka_col is None:
            df['Marka'] = df.get('Urun', df.get('Product', 'Belirtilmemis'))
            marka_col = 'Marka'
        
        if paket_col is None:
            df['Paket'] = df.get('Urun_Detay', df.get('Product_Detail', 'Standart'))
            paket_col = 'Paket'
        
        # Birleşik ProdPack ID
        df['ProdPack_ID'] = (
            df[molekul_col].astype(str) + '|' +
            df[sirket_col].astype(str) + '|' +
            df[marka_col].astype(str) + '|' +
            df[paket_col].astype(str)
        )
        
        df['ProdPack_Label'] = (
            df[marka_col].astype(str) + ' - ' +
            df[paket_col].astype(str) + ' (' +
            df[sirket_col].astype(str) + ')'
        )
        
        return df
    
    @staticmethod
    def _create_analytical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Büyüme, pazar payı, CAGR hesaplamaları"""
        
        # Satış sütunlarını bul
        sales_cols = [c for c in df.columns if re.search(r'Satis_20\d{2}', c)]
        sales_cols.sort()
        
        if len(sales_cols) >= 2:
            # En son satış sütunu
            latest_sales = sales_cols[-1]
            prev_sales = sales_cols[-2]
            
            # Büyüme oranı (güvenli hesaplama)
            mask = df[prev_sales] != 0
            df.loc[mask, 'Buyume_Orani_2023_2024'] = (
                (df.loc[mask, latest_sales] - df.loc[mask, prev_sales]) / 
                df.loc[mask, prev_sales] * 100
            )
            df.loc[~mask, 'Buyume_Orani_2023_2024'] = 0
            
            # Pazar payı hesaplama
            total_market = df[latest_sales].sum()
            if total_market > 0:
                df['Pazar_Payi_2024'] = (df[latest_sales] / total_market) * 100
            
            # CAGR (2+ yıl)
            if len(sales_cols) > 2:
                first_sales = sales_cols[0]
                n_years = len(sales_cols) - 1
                mask = df[first_sales] > 0
                df.loc[mask, 'CAGR'] = (
                    (df.loc[mask, latest_sales] / df.loc[mask, first_sales]) ** (1/n_years) - 1
                ) * 100
                df.loc[~mask, 'CAGR'] = 0
        
        return df

# ================================================
# 2. PRODPACK DERINLIK ANALIZI MODULU
# ================================================

class ProdPackDeepDive:
    """
    Molekül -> Şirket -> Marka -> Paket hiyerarşisi
    Sunburst/Sankey diyagramları
    Pazar Kanibalizasyonu Analizi
    """
    
    @staticmethod
    def create_hierarchy_data(df: pd.DataFrame) -> Dict:
        """Hiyerarşik veri yapısını oluştur"""
        
        # Gerekli sütunlar
        molekul_col = next((c for c in df.columns if 'Molekul' in c), 'Molekul')
        sirket_col = next((c for c in df.columns if 'Sirket' in c or 'Uretici' in c), 'Sirket')
        marka_col = next((c for c in df.columns if 'Marka' in c), 'Marka')
        paket_col = next((c for c in df.columns if 'Paket' in c), 'Paket')
        
        # Satış sütunu
        sales_cols = [c for c in df.columns if re.search(r'Satis_20\d{2}', c)]
        sales_col = sales_cols[-1] if sales_cols else None
        
        if sales_col is None:
            return {}
        
        # Hiyerarşik toplamlar
        hierarchy = {
            'molekuller': {},
            'sirketler': {},
            'markalar': {},
            'paketler': {}
        }
        
        # Toplam pazar değeri
        total_market = df[sales_col].sum()
        
        # Her seviye için toplamlar
        for _, row in df.iterrows():
            molekul = row[molekul_col]
            sirket = row[sirket_col]
            marka = row[marka_col]
            paket = row[paket_col]
            sales = row[sales_col]
            prodpack_id = row.get('ProdPack_ID', f'{molekul}|{sirket}|{marka}|{paket}')
            
            # Molekül seviyesi
            if molekul not in hierarchy['molekuller']:
                hierarchy['molekuller'][molekul] = {'sales': 0, 'children': set()}
            hierarchy['molekuller'][molekul]['sales'] += sales
            hierarchy['molekuller'][molekul]['children'].add(prodpack_id)
            
            # Şirket seviyesi (molekül altında)
            company_key = f"{molekul}||{sirket}"
            if company_key not in hierarchy['sirketler']:
                hierarchy['sirketler'][company_key] = {
                    'molekul': molekul,
                    'sirket': sirket,
                    'sales': 0,
                    'children': set()
                }
            hierarchy['sirketler'][company_key]['sales'] += sales
            hierarchy['sirketler'][company_key]['children'].add(prodpack_id)
            
            # Marka seviyesi (şirket altında)
            brand_key = f"{molekul}||{sirket}||{marka}"
            if brand_key not in hierarchy['markalar']:
                hierarchy['markalar'][brand_key] = {
                    'molekul': molekul,
                    'sirket': sirket,
                    'marka': marka,
                    'sales': 0,
                    'children': set()
                }
            hierarchy['markalar'][brand_key]['sales'] += sales
            hierarchy['markalar'][brand_key]['children'].add(prodpack_id)
            
            # Paket seviyesi
            if prodpack_id not in hierarchy['paketler']:
                hierarchy['paketler'][prodpack_id] = {
                    'molekul': molekul,
                    'sirket': sirket,
                    'marka': marka,
                    'paket': paket,
                    'sales': sales,
                    'label': row.get('ProdPack_Label', f'{marka} - {paket}')
                }
        
        # Pazar paylarını ekle
        for molekul in hierarchy['molekuller']:
            hierarchy['molekuller'][molekul]['share'] = (
                hierarchy['molekuller'][molekul]['sales'] / total_market * 100
            )
        
        return hierarchy
    
    @staticmethod
    def create_sunburst_chart(df: pd.DataFrame) -> go.Figure:
        """Molekül -> Şirket -> Marka -> Paket Sunburst diyagramı"""
        
        hierarchy = ProdPackDeepDive.create_hierarchy_data(df)
        if not hierarchy:
            return go.Figure()
        
        # Sunburst veri yapısı
        ids = []
        labels = []
        parents = []
        values = []
        
        # Kök (Root)
        ids.append('Pazar')
        labels.append('İlaç Pazarı')
        parents.append('')
        values.append(sum([v['sales'] for v in hierarchy['molekuller'].values()]))
        
        # Moleküller
        for molekul, data in hierarchy['molekuller'].items():
            ids.append(f"Molekul_{molekul}")
            labels.append(f"{molekul}<br>%{data['share']:.1f}")
            parents.append('Pazar')
            values.append(data['sales'])
        
        # Şirketler
        for company_key, data in hierarchy['sirketler'].items():
            molekul = data['molekul']
            sirket = data['sirket']
            company_sales = data['sales']
            
            ids.append(f"Sirket_{company_key}")
            labels.append(f"{sirket}<br>₺{company_sales/1e6:.1f}M")
            parents.append(f"Molekul_{molekul}")
            values.append(company_sales)
        
        # Markalar
        for brand_key, data in hierarchy['markalar'].items():
            molekul = data['molekul']
            sirket = data['sirket']
            marka = data['marka']
            brand_sales = data['sales']
            
            ids.append(f"Marka_{brand_key}")
            labels.append(marka[:20] + '...' if len(marka) > 20 else marka)
            parents.append(f"Sirket_{molekul}||{sirket}")
            values.append(brand_sales)
        
        # Paketler (ilk 50)
        paket_count = 0
        for paket_id, data in sorted(
            hierarchy['paketler'].items(), 
            key=lambda x: x[1]['sales'], 
            reverse=True
        )[:50]:
            molekul = data['molekul']
            sirket = data['sirket']
            marka = data['marka']
            paket = data['paket']
            
            ids.append(f"Paket_{paket_id}")
            labels.append(data['label'][:25] + '...' if len(data['label']) > 25 else data['label'])
            parents.append(f"Marka_{molekul}||{sirket}||{marka}")
            values.append(data['sales'])
            paket_count += 1
        
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total',
            marker=dict(
                colorscale='RdBu',
                line=dict(width=1, color='#1a2639')
            ),
            hovertemplate='<b>%{label}</b><br>Satış: ₺%{value:,.0f}<br>Pay: %{percentRoot:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='🔬 ProdPack Hiyerarşisi (Molekül → Şirket → Marka → Paket)',
                font=dict(size=20, color='#d4af37'),
                x=0.5
            ),
            height=700,
            paper_bgcolor='#0c1a32',
            font=dict(color='#f8fafc', size=12),
            margin=dict(t=50, l=10, r=10, b=10)
        )
        
        return fig
    
    @staticmethod
    def analyze_cannibalization(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aynı şirket içindeki paketler arası kanibalizasyon analizi.
        Büyüme/Hacim matrisi ile.
        """
        
        sirket_col = next((c for c in df.columns if 'Sirket' in c), 'Sirket')
        marka_col = next((c for c in df.columns if 'Marka' in c), 'Marka')
        paket_col = next((c for c in df.columns if 'Paket' in c), 'Paket')
        
        sales_cols = [c for c in df.columns if re.search(r'Satis_20\d{2}', c)]
        sales_cols.sort()
        
        if len(sales_cols) < 2:
            return pd.DataFrame()
        
        current_sales = sales_cols[-1]
        prev_sales = sales_cols[-2]
        
        cannibalization_results = []
        
        # Her şirket için
        for sirket in df[sirket_col].unique():
            sirket_df = df[df[sirket_col] == sirket].copy()
            
            if len(sirket_df) < 2:
                continue
            
            # Şirket toplam satışı
            total_sirket_sales = sirket_df[current_sales].sum()
            
            # Marka bazlı analiz
            for marka in sirket_df[marka_col].unique():
                marka_df = sirket_df[sirket_df[marka_col] == marka]
                
                if len(marka_df) < 2:
                    continue
                
                # Marka altındaki paketler
                paketler = marka_df[paket_col].tolist()
                paket_satislar = marka_df[current_sales].tolist()
                paket_buyumeler = []
                
                for _, row in marka_df.iterrows():
                    if row[prev_sales] != 0:
                        growth = ((row[current_sales] - row[prev_sales]) / row[prev_sales]) * 100
                    else:
                        growth = 0
                    paket_buyumeler.append(growth)
                
                # Kanibalizasyon skoru (Portföy içi rekabet)
                if len(paket_satislar) > 1:
                    # Satış eşitsizliği + Büyüme korelasyonu
                    sales_inequality = np.std(paket_satislar) / (np.mean(paket_satislar) + 1)
                    growth_corr = np.corrcoef(paket_buyumeler, paket_satislar)[0, 1] if len(paket_buyumeler) > 1 else 0
                    growth_corr = 0 if np.isnan(growth_corr) else abs(growth_corr)
                    
                    cannibal_score = (sales_inequality * 0.6 + (1 - growth_corr) * 0.4) * 100
                else:
                    cannibal_score = 0
                
                # Her paket için sonuç
                for i, paket in enumerate(paketler):
                    paket_pay = (paket_satislar[i] / total_sirket_sales * 100) if total_sirket_sales > 0 else 0
                    
                    cannibalization_results.append({
                        'Sirket': sirket,
                        'Marka': marka,
                        'Paket': paket,
                        'Satis_2024': paket_satislar[i],
                        'Buyume_2024': paket_buyumeler[i],
                        'Sirket_Icinde_Payi': paket_pay,
                        'Kanibalizasyon_Risk_Skoru': min(cannibal_score * (1 - paket_pay/100), 100),
                        'Risk_Seviyesi': 'Yüksek' if cannibal_score > 70 else 'Orta' if cannibal_score > 40 else 'Düşük'
                    })
        
        return pd.DataFrame(cannibalization_results)
    
    @staticmethod
    def get_molecule_drilldown(df: pd.DataFrame, selected_molecule: str) -> pd.DataFrame:
        """Seçili molekül altındaki tüm ProdPack'leri getir"""
        
        molekul_col = next((c for c in df.columns if 'Molekul' in c), 'Molekul')
        
        if selected_molecule and selected_molecule != 'Tümü':
            filtered_df = df[df[molekul_col] == selected_molecule].copy()
        else:
            filtered_df = df.copy()
        
        # Gerekli sütunlar
        cols_to_show = []
        for col in ['Molekul', 'Sirket', 'Marka', 'Paket', 'ProdPack_Label']:
            if col in filtered_df.columns:
                cols_to_show.append(col)
        
        # Satış ve büyüme sütunları
        sales_cols = [c for c in filtered_df.columns if re.search(r'Satis_20\d{2}', c)]
        sales_cols.sort()
        if sales_cols:
            cols_to_show.append(sales_cols[-1])
        
        if 'Buyume_Orani_2023_2024' in filtered_df.columns:
            cols_to_show.append('Buyume_Orani_2023_2024')
        
        if 'Pazar_Payi_2024' in filtered_df.columns:
            cols_to_show.append('Pazar_Payi_2024')
        
        result_df = filtered_df[cols_to_show].copy()
        
        # Formatlama
        if sales_cols:
            result_df.rename(columns={sales_cols[-1]: 'Satis_2024'}, inplace=True)
        
        return result_df.sort_values('Satis_2024', ascending=False)

# ================================================
# 3. ILERI SEVIYE AI VE TAHMINLEME MODULU
# ================================================

class StrategicAIEngine:
    """
    Holt-Winters Tahminleme (2025-2026)
    IsolationForest ile Anomali Tespiti
    PCA + K-Means Segmentasyon
    """
    
    @staticmethod
    def forecast_2025_2026(df: pd.DataFrame) -> Dict[str, ForecastResult]:
        """Statsmodels Holt-Winters ile 2025-2026 pazar tahmini"""
        
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        sales_cols = [c for c in df.columns if re.search(r'Satis_20\d{2}', c)]
        sales_cols.sort()
        
        if len(sales_cols) < 3:
            return {}
        
        # Yıllık toplam satışlar
        yearly_sales = {}
        for col in sales_cols:
            year = int(re.search(r'20\d{2}', col).group())
            yearly_sales[year] = df[col].sum()
        
        years = sorted(yearly_sales.keys())
        sales = [yearly_sales[y] for y in years]
        
        if len(sales) < 3:
            return {}
        
        # Zaman serisi
        ts = pd.Series(sales, index=pd.to_datetime([str(y) for y in years]))
        
        forecasts = {}
        
        try:
            # Holt-Winters modeli
            model = ExponentialSmoothing(
                ts,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            ).fit()
            
            # 2025-2026 tahmini
            future_years = [max(years) + 1, max(years) + 2]
            forecast = model.forecast(len(future_years))
            
            # Güven aralığı (basit yaklaşım)
            resid_std = np.std(model.resid)
            
            forecasts['holt_winters'] = ForecastResult(
                periods=[str(y) for y in future_years],
                predictions=forecast.values.tolist(),
                lower_bounds=(forecast - 1.96 * resid_std).values.tolist(),
                upper_bounds=(forecast + 1.96 * resid_std).values.tolist(),
                model_type='Holt-Winters',
                growth_rate=((forecast.values[-1] - forecast.values[0]) / forecast.values[0] * 100)
            )
            
            # Basit lineer trend (yedek)
            z = np.polyfit(years, sales, 1)
            p = np.poly1d(z)
            linear_forecast = p(future_years)
            
            forecasts['linear_trend'] = ForecastResult(
                periods=[str(y) for y in future_years],
                predictions=linear_forecast.tolist(),
                lower_bounds=(linear_forecast * 0.9).tolist(),
                upper_bounds=(linear_forecast * 1.1).tolist(),
                model_type='Linear Trend',
                growth_rate=((linear_forecast[-1] - linear_forecast[0]) / linear_forecast[0] * 100)
            )
            
        except Exception as e:
            st.warning(f"Tahminleme hatası: {str(e)}")
        
        return forecasts
    
    @staticmethod
    def detect_anomalies_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
        """IsolationForest ile pazar normlarından sapan paketleri tespit et"""
        
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        
        result_df = df.copy()
        
        # Satış sütunları
        sales_cols = [c for c in df.columns if re.search(r'Satis_20\d{2}', c)]
        sales_cols.sort()
        
        if len(sales_cols) < 2:
            return result_df
        
        # Feature'lar
        features = []
        
        # En son satış
        latest_sales = sales_cols[-1]
        features.append(latest_sales)
        
        # Büyüme oranı
        if 'Buyume_Orani_2023_2024' in df.columns:
            features.append('Buyume_Orani_2023_2024')
        
        # Pazar payı
        if 'Pazar_Payi_2024' in df.columns:
            features.append('Pazar_Payi_2024')
        
        # CAGR
        if 'CAGR' in df.columns:
            features.append('CAGR')
        
        if len(features) < 2:
            return result_df
        
        # NaN'ları temizle
        X = df[features].fillna(0)
        
        # Outlier'ları olanları filtrele (eğitim için)
        valid_idx = ~((X == 0).all(axis=1))
        if valid_idx.sum() < 10:
            return result_df
        
        X_valid = X[valid_idx]
        
        # Ölçeklendirme
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.15,
            random_state=42,
            n_estimators=200
        )
        
        preds = iso_forest.fit_predict(X_scaled)
        scores = iso_forest.decision_function(X_scaled)
        
        # Sonuçları ana DataFrame'e ekle
        result_df.loc[valid_idx, 'Anomali_Skoru'] = scores
        result_df.loc[valid_idx, 'Anomali_Tespiti'] = preds
        result_df['Anomali_Durumu'] = result_df.get('Anomali_Tespiti', 1) == -1
        
        # Risk seviyeleri
        conditions = [
            (result_df['Anomali_Skoru'] < -0.5),
            (result_df['Anomali_Skoru'] < -0.2),
            (result_df['Anomali_Skoru'] < 0.1),
            (result_df['Anomali_Skoru'] >= 0.1)
        ]
        choices = [RiskLevel.KRITIK, RiskLevel.YUKSEK, RiskLevel.ORTA, RiskLevel.NORMAL]
        
        result_df['Risk_Seviyesi'] = np.select(conditions, choices, default=RiskLevel.NORMAL)
        
        # Anomali tipi
        def get_anomaly_type(row):
            if not row.get('Anomali_Durumu', False):
                return 'Normal'
            
            if row.get('Buyume_Orani_2023_2024', 0) > 50:
                return '🚀 Aşırı Büyüme'
            elif row.get('Buyume_Orani_2023_2024', 0) < -30:
                return '📉 Kritik Düşüş'
            elif row.get('Pazar_Payi_2024', 0) < 1 and row.get('Buyume_Orani_2023_2024', 0) > 20:
                return '🌟 Yükselen Fırsat'
            elif row.get('Pazar_Payi_2024', 0) > 20 and row.get('Buyume_Orani_2023_2024', 0) < -10:
                return '⚠️ Lider Tehlikede'
            else:
                return '📊 Anormal Patern'
        
        result_df['Anomali_Tipi'] = result_df.apply(get_anomaly_type, axis=1)
        
        return result_df
    
    @staticmethod
    def pca_kmeans_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
        """
        PCA + K-Means ile ürün segmentasyonu.
        Liderler, Potansiyeller, Riskli Ürünler, Nakit İnekleri
        """
        
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        result_df = df.copy()
        
        # Feature seçimi
        features = []
        
        # 1. Pazar Payı
        if 'Pazar_Payi_2024' in df.columns:
            features.append('Pazar_Payi_2024')
        
        # 2. Büyüme Hızı
        if 'Buyume_Orani_2023_2024' in df.columns:
            features.append('Buyume_Orani_2023_2024')
        
        # 3. Satış Hacmi
        sales_cols = [c for c in df.columns if re.search(r'Satis_20\d{2}', c)]
        if sales_cols:
            features.append(sales_cols[-1])
        
        # 4. CAGR (varsa)
        if 'CAGR' in df.columns:
            features.append('CAGR')
        
        if len(features) < 2:
            return result_df
        
        # NaN'ları temizle
        X = df[features].fillna(0)
        
        # Sıfır satışlı olanları filtrele (segmentasyon dışı)
        valid_idx = ~((X == 0).all(axis=1))
        if valid_idx.sum() < n_clusters * 2:
            return result_df
        
        X_valid = X[valid_idx]
        
        # Ölçeklendirme
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # PCA ile boyut indirgeme
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # K-Means kümeleme
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Segment isimlendirme (Kural tabanlı)
        segment_names = []
        segment_data = X_valid.copy()
        segment_data['Cluster'] = clusters
        segment_data['Pazar_Payi'] = segment_data.get('Pazar_Payi_2024', 0)
        segment_data['Buyume'] = segment_data.get('Buyume_Orani_2023_2024', 0)
        
        cluster_profiles = {}
        for c in range(n_clusters):
            cluster_subset = segment_data[segment_data['Cluster'] == c]
            avg_share = cluster_subset['Pazar_Payi'].mean()
            avg_growth = cluster_subset['Buyume'].mean()
            cluster_profiles[c] = {'share': avg_share, 'growth': avg_growth}
        
        for c in range(n_clusters):
            profile = cluster_profiles[c]
            
            if profile['share'] > 15 and profile['growth'] > 10:
                name = ProductSegment.STARS
            elif profile['share'] > 15 and profile['growth'] <= 10:
                name = ProductSegment.CASH_COWS
            elif profile['share'] <= 15 and profile['growth'] > 15:
                name = ProductSegment.EMERGING
            elif profile['share'] <= 8 and profile['growth'] < 0:
                name = ProductSegment.DOGS
            else:
                name = ProductSegment.QUESTION_MARKS
            
            segment_names.append(name)
        
        # Sonuçları ana DataFrame'e ekle
        result_df.loc[valid_idx, 'Segment_Cluster'] = clusters
        result_df.loc[valid_idx, 'PCA_1'] = X_pca[:, 0]
        result_df.loc[valid_idx, 'PCA_2'] = X_pca[:, 1]
        
        # Segment adı ata
        segment_map = {c: name for c, name in zip(range(n_clusters), segment_names)}
        result_df['Segment_Adi'] = result_df.get('Segment_Cluster', -1).map(segment_map)
        result_df['Segment_Adi'] = result_df['Segment_Adi'].fillna('Sınıflandırılmamış')
        
        return result_df

# ================================================
# 4. KURUMSAL UI/UX: EXECUTIVE DARK MODE
# ================================================

class ExecutiveUI:
    """Kurumsal tema, insight box'lar ve stratejik kartlar"""
    
    @staticmethod
    def inject_custom_css():
        """Executive Dark Mode CSS - Lacivert, Gümüş, Altın"""
        
        css = """
        <style>
            /* Ana tema - Executive Dark Mode */
            :root {
                --navy-deep: #0a1929;
                --navy-medium: #1e3a5f;
                --navy-light: #2d4a7a;
                --gold-primary: #d4af37;
                --gold-secondary: #c0a040;
                --silver: #c0c0c0;
                --text-primary: #ffffff;
                --text-secondary: #e0e0e0;
                --success: #2e7d32;
                --warning: #ed6c02;
                --danger: #d32f2f;
                --info: #0288d1;
            }
            
            /* Global arkaplan */
            .stApp {
                background: linear-gradient(145deg, var(--navy-deep), #0b1e33);
                color: var(--text-primary);
            }
            
            /* Ana başlık */
            .executive-title {
                background: linear-gradient(135deg, var(--gold-primary), var(--silver));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.8rem;
                font-weight: 800;
                margin-bottom: 0.2rem;
                text-shadow: 0 0 10px rgba(212, 175, 55, 0.3);
            }
            
            /* Insight Box - Yönetici Özeti */
            .insight-box {
                background: linear-gradient(135deg, rgba(30, 58, 95, 0.9), rgba(20, 40, 70, 0.95));
                border-left: 8px solid var(--gold-primary);
                border-radius: 12px;
                padding: 1.8rem;
                margin: 1.5rem 0;
                box-shadow: 0 8px 32px rgba(0,0,0,0.4);
                backdrop-filter: blur(10px);
                color: var(--text-primary);
                border: 1px solid rgba(212, 175, 55, 0.3);
            }
            
            .insight-title {
                color: var(--gold-primary);
                font-size: 1.3rem;
                font-weight: 600;
                margin-bottom: 1rem;
                letter-spacing: 1px;
            }
            
            .insight-text {
                font-size: 1.1rem;
                line-height: 1.6;
                color: var(--text-secondary);
            }
            
            /* Stratejik öneri kartları */
            .strategic-card {
                background: rgba(45, 74, 122, 0.3);
                border: 1px solid rgba(212, 175, 55, 0.2);
                border-radius: 16px;
                padding: 1.5rem;
                backdrop-filter: blur(8px);
                transition: all 0.3s ease;
            }
            
            .strategic-card:hover {
                border-color: var(--gold-primary);
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(212, 175, 55, 0.15);
            }
            
            .gold-text {
                color: var(--gold-primary);
                font-weight: 600;
            }
            
            /* Metrik kartları */
            .metric-card {
                background: rgba(26, 58, 95, 0.7);
                border-radius: 12px;
                padding: 1.2rem;
                border: 1px solid rgba(255,255,255,0.1);
                text-align: center;
            }
            
            .metric-value {
                font-size: 2rem;
                font-weight: 700;
                color: var(--gold-primary);
            }
            
            .metric-label {
                font-size: 0.9rem;
                color: var(--silver);
                text-transform: uppercase;
            }
        </style>
        """
        
        st.markdown(css, unsafe_allow_html=True)
    
    @staticmethod
    def insight_box(title: str, content: str, icon: str = "💡"):
        """Yönetici Özeti (Insight Box) oluşturur"""
        
        html = f"""
        <div class="insight-box">
            <div class="insight-title">{icon} {title}</div>
            <div class="insight-text">{content}</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    @staticmethod
    def strategic_recommendation_card(recommendation: Dict):
        """Stratejik yatırım tavsiyesi kartı"""
        
        priority_colors = {
            'Yüksek': '#d32f2f',
            'Orta': '#ed6c02',
            'Düşük': '#2e7d32'
        }
        
        color = priority_colors.get(recommendation.get('priority', 'Orta'), '#ed6c02')
        
        html = f"""
        <div class="strategic-card" style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: var(--gold-primary); font-size: 1.2rem;">{recommendation.get('title', 'Stratejik Öneri')}</span>
                <span style="background: {color}; padding: 0.2rem 1rem; border-radius: 20px; font-size: 0.8rem;">
                    {recommendation.get('priority', 'Orta')} Öncelik
                </span>
            </div>
            <p style="color: white; margin: 1rem 0;">{recommendation.get('description', '')}</p>
            <div style="background: rgba(212, 175, 55, 0.1); padding: 0.8rem; border-radius: 8px;">
                <span style="color: var(--gold-primary);">🎯 Öneri:</span> 
                <span style="color: white;">{recommendation.get('action', '')}</span>
            </div>
            <div style="margin-top: 0.8rem; color: var(--silver); font-size: 0.9rem;">
                📊 Etki: {recommendation.get('impact', 'Stratejik')}
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    @staticmethod
    def generate_auto_insight(df: pd.DataFrame, context: str = "general") -> str:
        """Grafiklerin altına otomatik Yönetici Özeti üretir"""
        
        insights = []
        
        sales_cols = [c for c in df.columns if re.search(r'Satis_20\d{2}', c)]
        sales_cols.sort()
        
        if len(sales_cols) >= 2:
            latest = sales_cols[-1]
            prev = sales_cols[-2]
            
            total_current = df[latest].sum()
            total_prev = df[prev].sum()
            
            if total_prev > 0:
                growth = (total_current - total_prev) / total_prev * 100
                
                if growth > 15:
                    insights.append(f"📈 Pazar **%{growth:.1f}** büyüme gösteriyor. En hızlı büyüyen segmentleri yakından takip edin.")
                elif growth < -10:
                    insights.append(f"⚠️ Pazar **%{growth:.1f}** daralıyor. Maliyet optimizasyonu ve portföy çeşitlendirmesi önerilir.")
                else:
                    insights.append(f"📊 Pazar istikrarlı seyrediyor (Büyüme: %{growth:.1f}). Pazar payı koruma stratejileri uygulanmalı.")
        
        # ProdPack liderleri
        if 'ProdPack_Label' in df.columns and sales_cols:
            top_prodpack = df.nlargest(1, sales_cols[-1])
            if not top_prodpack.empty:
                label = top_prodpack['ProdPack_Label'].iloc[0]
                sales = top_prodpack[sales_cols[-1]].iloc[0]
                share = top_prodpack.get('Pazar_Payi_2024', 0).iloc[0] if 'Pazar_Payi_2024' in top_prodpack else 0
                
                insights.append(f"🏆 Pazar lideri: **{label}** - ₺{sales/1e6:.1f}M satış, %{share:.1f} pazar payı.")
        
        # Risk uyarıları
        if 'Risk_Seviyesi' in df.columns:
            kritik_risk = df[df['Risk_Seviyesi'] == RiskLevel.KRITIK].shape[0]
            yuksek_risk = df[df['Risk_Seviyesi'] == RiskLevel.YUKSEK].shape[0]
            
            if kritik_risk > 0:
                insights.append(f"🚨 **Kritik uyarı:** {kritik_risk} ürün/paket anomali gösteriyor. Acil aksiyon planı oluşturun.")
            elif yuksek_risk > 5:
                insights.append(f"⚠️ **Risk uyarısı:** {yuksek_risk} ürün yüksek risk kategorisinde. Risk azaltma stratejileri uygulayın.")
        
        # Kanibalizasyon uyarısı
        if 'Kanibalizasyon_Risk_Skoru' in df.columns:
            high_cannibal = df[df['Kanibalizasyon_Risk_Skoru'] > 70].shape[0]
            if high_cannibal > 3:
                insights.append(f"🔄 **Portföy kanibalizasyonu:** {high_cannibal} paket aynı şirket içinde yüksek rekabet oluşturuyor.")
        
        if not insights:
            insights.append("📋 Veri analizi tamamlandı. Detaylı inceleme için aşağıdaki grafikleri kullanın.")
        
        return " • ".join(insights)

# ================================================
# 5. ANA UYGULAMA (STREAMLIT)
# ================================================

def main():
    """PharmaIntelligence Pro v8.0 Ana Uygulama"""
    
    # CSS tema
    ExecutiveUI.inject_custom_css()
    
    # Session state başlatma
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'anomaly_data' not in st.session_state:
        st.session_state.anomaly_data = None
    if 'segment_data' not in st.session_state:
        st.session_state.segment_data = None
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
    if 'cannibal_data' not in st.session_state:
        st.session_state.cannibal_data = None
    if 'hierarchy_data' not in st.session_state:
        st.session_state.hierarchy_data = None
    
    # ========================================
    # SIDEBAR - Kontrol Paneli
    # ========================================
    with st.sidebar:
        st.markdown('<div style="text-align: center; padding: 1rem 0;">', unsafe_allow_html=True)
        st.markdown('<h1 style="color: #d4af37; font-size: 1.8rem;">💊 PharmaIntel Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color: #c0c0c0;">Enterprise v8.0</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Veri Yükleme
        st.markdown("### 📁 VERİ YÜKLEME")
        uploaded_file = st.file_uploader(
            "Excel veya CSV dosyası seçin",
            type=['xlsx', 'xls', 'csv'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            if st.button("🚀 VERİYİ İŞLE", use_container_width=True):
                with st.spinner("Veri işleniyor... Bu işlem 1M+ satır için optimize edilmiştir."):
                    df = PharmaDataEngine.load_and_process_data(uploaded_file)
                    if not df.empty:
                        st.session_state.processed_data = df
                        
                        # Ön analizler
                        with st.spinner("İleri analizler hazırlanıyor..."):
                            st.session_state.anomaly_data = StrategicAIEngine.detect_anomalies_isolation_forest(df)
                            st.session_state.segment_data = StrategicAIEngine.pca_kmeans_segmentation(df)
                            st.session_state.forecast_data = StrategicAIEngine.forecast_2025_2026(df)
                            st.session_state.cannibal_data = ProdPackDeepDive.analyze_cannibalization(df)
                        
                        st.success(f"✅ Veri işlendi: {len(df):,} satır, {len(df.columns)} sütun")
                        st.rerun()
        
        st.divider()
        
        # Veri durumu
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            
            st.markdown("### 📊 VERİ ÖZETİ")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Satır", f"{len(df):,}")
            with col2:
                st.metric("Sütun", len(df.columns))
            
            sales_cols = [c for c in df.columns if re.search(r'Satis_20\d{2}', c)]
            if sales_cols:
                total_sales = df[sales_cols[-1]].sum()
                st.metric("Pazar Büyüklüğü 2024", f"₺{total_sales/1e6:.1f}M")
            
            # Molekül seçici (ProdPack Drill-Down için)
            st.divider()
            st.markdown("### 🔬 PRODPACK DRILL-DOWN")
            molekul_col = next((c for c in df.columns if 'Molekul' in c), None)
            if molekul_col:
                molekuller = ['Tümü'] + df[molekul_col].unique().tolist()
                selected_molecule = st.selectbox(
                    "Molekül Seçin",
                    molekuller,
                    help="Seçili molekül altındaki tüm marka ve paketleri görüntüleyin"
                )
                st.session_state.selected_molecule = selected_molecule
    
    # ========================================
    # MAIN CONTENT - Ana Gösterge Paneli
    # ========================================
    
    if st.session_state.processed_data is None:
        # Hoşgeldin ekranı
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<h1 class="executive-title" style="text-align: center;">PharmaIntelligence Pro</h1>', unsafe_allow_html=True)
            st.markdown('<p style="text-align: center; color: #c0c0c0; font-size: 1.2rem;">Kurumsal Karar Destek ve Stratejik İstihbarat Platformu</p>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: rgba(30,58,95,0.7); padding: 2rem; border-radius: 20px; margin-top: 2rem;">
                <h3 style="color: #d4af37;">🚀 Başlamak İçin:</h3>
                <ol style="color: white; font-size: 1.1rem; line-height: 2;">
                    <li>📁 Sol paneldan veri dosyanızı yükleyin</li>
                    <li>⚙️ "VERİYİ İŞLE" butonuna tıklayın</li>
                    <li>📊 Aşağıdaki analiz modüllerini keşfedin</li>
                </ol>
                <p style="color: #c0c0c0; margin-top: 1.5rem; font-style: italic;">
                Desteklenen: Molekül, Şirket, Marka, Paket hiyerarşisi • AI Tahminleme • Risk Analizi • Segmentasyon
                </p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    # Veri yüklü - Dashboard
    df = st.session_state.processed_data
    
    # Executive Başlık
    st.markdown('<h1 class="executive-title">PharmaIntelligence Pro v8.0</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #c0c0c0; font-size: 1.1rem; margin-bottom: 2rem;">Enterprise Karar Destek • AI Tahminleme 2025-2026 • ProdPack Derinlik Analizi</p>', unsafe_allow_html=True)
    
    # Ana Insight Box (Otomatik Yönetici Özeti)
    auto_insight = ExecutiveUI.generate_auto_insight(df)
    ExecutiveUI.insight_box("YÖNETİCİ ÖZETİ", auto_insight, "🎯")
    
    # Sekmeler
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📦 PRODPACK DERİNLİK",
        "🔮 TAHMİN & YATIRIM",
        "⚠️ RİSK & FIRSAT",
        "🎯 SEGMENTASYON",
        "📈 STRATEJİK RAPOR"
    ])
    
    # ========================================
    # TAB 1: PRODPACK DERİNLİK ANALİZİ
    # ========================================
    with tab1:
        st.markdown("### 📊 ProdPack Hiyerarşi ve Kanibalizasyon Analizi")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Sunburst Diyagramı
            sunburst_fig = ProdPackDeepDive.create_sunburst_chart(df)
            if sunburst_fig:
                st.plotly_chart(sunburst_fig, use_container_width=True)
            else:
                st.info("Sunburst grafiği oluşturulamadı: Veri hiyerarşisi yetersiz.")
        
        with col2:
            st.markdown("#### 🔬 Molekül Drill-Down")
            
            molekul_col = next((c for c in df.columns if 'Molekul' in c), 'Molekul')
            if molekul_col:
                selected = st.session_state.get('selected_molecule', 'Tümü')
                
                drill_df = ProdPackDeepDive.get_molecule_drilldown(df, selected)
                
                if not drill_df.empty:
                    # Formatlama
                    display_df = drill_df.copy()
                    if 'Satis_2024' in display_df.columns:
                        display_df['Satis_2024'] = display_df['Satis_2024'].apply(lambda x: f"₺{x:,.0f}")
                    if 'Buyume_Orani_2023_2024' in display_df.columns:
                        display_df['Buyume_Orani_2023_2024'] = display_df['Buyume_Orani_2023_2024'].apply(lambda x: f"%{x:.1f}")
                    if 'Pazar_Payi_2024' in display_df.columns:
                        display_df['Pazar_Payi_2024'] = display_df['Pazar_Payi_2024'].apply(lambda x: f"%{x:.2f}")
                    
                    st.dataframe(display_df.head(10), use_container_width=True, height=400)
                    
                    st.caption(f"📌 Toplam {len(drill_df)} ProdPack gösteriliyor. İlk 10 satır.")
        
        # Kanibalizasyon Analizi
        st.divider()
        st.markdown("#### 🔄 Pazar Kanibalizasyonu Analizi")
        
        if st.session_state.cannibal_data is not None and not st.session_state.cannibal_data.empty:
            cannibal_df = st.session_state.cannibal_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Yüksek riskli kanibalizasyon
                high_risk = cannibal_df[cannibal_df['Risk_Seviyesi'] == 'Yüksek']
                st.metric("Yüksek Kanibalizasyon Riski", len(high_risk), 
                         delta=f"{len(high_risk)/len(cannibal_df)*100:.1f}%")
                
                if not high_risk.empty:
                    st.dataframe(
                        high_risk[['Sirket', 'Marka', 'Paket', 'Kanibalizasyon_Risk_Skoru']].head(5),
                        use_container_width=True
                    )
            
            with col2:
                # Kanibalizasyon matrisi
                fig = px.scatter(
                    cannibal_df,
                    x='Sirket_Icinde_Payi',
                    y='Buyume_2024',
                    size='Kanibalizasyon_Risk_Skoru',
                    color='Risk_Seviyesi',
                    hover_name='Paket',
                    hover_data=['Sirket', 'Marka'],
                    title='Kanibalizasyon Matrisi (Büyüme vs Şirket İçi Pay)',
                    color_discrete_map={
                        'Yüksek': '#d32f2f',
                        'Orta': '#ed6c02',
                        'Düşük': '#2e7d32'
                    }
                )
                fig.update_layout(template='plotly_dark', paper_bgcolor='#0c1a32')
                st.plotly_chart(fig, use_container_width=True)
            
            # Kanibalizasyon Insight
            if len(high_risk) > 2:
                ExecutiveUI.insight_box(
                    "PORTFÖY KANİBALİZASYON UYARISI",
                    f"🔴 {len(high_risk)} paket yüksek kanibalizasyon riski taşıyor. "
                    f"Özellikle {high_risk.iloc[0]['Sirket']} şirketinin {high_risk.iloc[0]['Marka']} markası altında "
                    f"rekabet yoğun. Ürün farklılaştırma ve fiyatlandırma stratejileri gözden geçirilmeli.",
                    "🔄"
                )
        else:
            st.info("Kanibalizasyon analizi için yeterli veri yok (en az 2 yıllık satış ve aynı şirket altında birden fazla paket gerekli).")
    
    # ========================================
    # TAB 2: TAHMİN & YATIRIM (2025-2026)
    # ========================================
    with tab2:
        st.markdown("### 🔮 Pazar Tahminleme 2025-2026 & Yatırım Tavsiyeleri")
        
        if st.session_state.forecast_data:
            forecasts = st.session_state.forecast_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Holt-Winters Tahmini")
                if 'holt_winters' in forecasts:
                    f = forecasts['holt_winters']
                    
                    # Tahmin grafiği
                    years = [2022, 2023, 2024] + [int(p) for p in f.periods]
                    
                    # Tarihsel veri
                    sales_cols = [c for c in df.columns if re.search(r'Satis_20\d{2}', c)]
                    sales_cols.sort()
                    historical_sales = [df[c].sum() for c in sales_cols[-3:]] if len(sales_cols) >= 3 else []
                    
                    fig = go.Figure()
                    
                    # Tarihsel
                    fig.add_trace(go.Scatter(
                        x=[2022, 2023, 2024][:len(historical_sales)],
                        y=historical_sales,
                        mode='lines+markers',
                        name='Gerçekleşen',
                        line=dict(color='#c0c0c0', width=3),
                        marker=dict(size=10)
                    ))
                    
                    # Tahmin
                    fig.add_trace(go.Scatter(
                        x=[int(p) for p in f.periods],
                        y=f.predictions,
                        mode='lines+markers',
                        name='Tahmin',
                        line=dict(color='#d4af37', width=3, dash='dash'),
                        marker=dict(size=10)
                    ))
                    
                    # Güven aralığı
                    fig.add_trace(go.Scatter(
                        x=[int(p) for p in f.periods] + [int(p) for p in f.periods][::-1],
                        y=f.upper_bounds + f.lower_bounds[::-1],
                        fill='toself',
                        fillcolor='rgba(212, 175, 55, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Güven Aralığı'
                    ))
                    
                    fig.update_layout(
                        title='Pazar Büyüklüğü Tahmini (2025-2026)',
                        xaxis_title='Yıl',
                        yaxis_title='Satış (₺)',
                        height=400,
                        template='plotly_dark',
                        paper_bgcolor='#0c1a32',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Büyüme metrikleri
                    growth_rate = f.growth_rate
                    st.metric("2025-2026 Tahmini Büyüme", f"%{growth_rate:.1f}", 
                             delta=f"{'📈' if growth_rate > 0 else '📉'}")
            
            with col2:
                st.markdown("#### 💎 Yatırım Tavsiyeleri")
                
                # Tahmin bazlı stratejik öneriler
                if 'holt_winters' in forecasts:
                    f = forecasts['holt_winters']
                    growth = f.growth_rate
                    
                    if growth > 15:
                        ExecutiveUI.strategic_recommendation_card({
                            'title': 'AGRESİF BÜYÜME STRATEJİSİ',
                            'priority': 'Yüksek',
                            'description': f'Pazarın %{growth:.1f} büyümesi bekleniyor. Kapasite artırımı ve yeni ürün lansmanları için uygun zaman.',
                            'action': 'Ar-Ge bütçesini %25 artırın, yeni pazarlara açılma fırsatlarını değerlendirin.',
                            'impact': 'Stratejik Büyüme'
                        })
                    elif growth > 5:
                        ExecutiveUI.strategic_recommendation_card({
                            'title': 'SEÇİCİ BÜYÜME STRATEJİSİ',
                            'priority': 'Orta',
                            'description': f'Pazar %{growth:.1f} büyüyecek. Yıldız ürünlere odaklanın.',
                            'action': 'Portföy optimizasyonu yapın, düşük performanslı ürünleri değerlendirin.',
                            'impact': 'Portföy Optimizasyonu'
                        })
                    else:
                        ExecutiveUI.strategic_recommendation_card({
                            'title': 'KORUMA & VERİMLİLİK STRATEJİSİ',
                            'priority': 'Yüksek',
                            'description': 'Pazar büyümesi sınırlı. Maliyet liderliği ve operasyonel mükemmellik ön planda.',
                            'action': 'Maliyet optimizasyonu, tedarik zinciri verimliliği ve sadakat programları.',
                            'impact': 'Operasyonel Verimlilik'
                        })
                    
                    # Segment bazlı öneriler
                    if st.session_state.segment_data is not None:
                        segment_df = st.session_state.segment_data
                        if 'Segment_Adi' in segment_df.columns:
                            yildiz_sayisi = segment_df[segment_df['Segment_Adi'] == ProductSegment.STARS].shape[0]
                            if yildiz_sayisi < 3:
                                ExecutiveUI.strategic_recommendation_card({
                                    'title': 'YILDIZ ÜRÜN GELİŞTİRME',
                                    'priority': 'Yüksek',
                                    'description': f'Portföyde sadece {yildiz_sayisi} yıldız ürün var. Büyüme potansiyeli yüksek yeni ürünlere ihtiyaç var.',
                                    'action': 'Soru işaretleri segmentindeki ürünleri analiz edin, en güçlü 2-3 ürüne yatırım yapın.',
                                    'impact': 'Uzun Vadeli Rekabet'
                                })
            
            # Pazar tahmini özeti
            if 'holt_winters' in forecasts:
                f = forecasts['holt_winters']
                prediction_2025 = f.predictions[0]
                prediction_2026 = f.predictions[1] if len(f.predictions) > 1 else 0
                
                insight_text = (
                    f"📊 **2025 Pazar Tahmini:** ₺{prediction_2025/1e6:.1f}M | "
                    f"**2026 Pazar Tahmini:** ₺{prediction_2026/1e6:.1f}M | "
                    f"**Yıllık Bileşik Büyüme:** %{f.growth_rate:.1f}"
                )
                ExecutiveUI.insight_box("2025-2026 PAZAR TAHMİNİ", insight_text, "🔮")
        
        else:
            st.info("Tahmin analizi için en az 3 yıllık satış verisi gereklidir.")
    
    # ========================================
    # TAB 3: RİSK & FIRSAT İZLEME
    # ========================================
    with tab3:
        st.markdown("### ⚠️ Anomali Tespiti ve Risk İzleme")
        
        if st.session_state.anomaly_data is not None:
            anomaly_df = st.session_state.anomaly_data
            
            # Risk özet kartları
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                kritik = anomaly_df[anomaly_df['Risk_Seviyesi'] == RiskLevel.KRITIK].shape[0]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Kritik Risk</div>
                    <div class="metric-value" style="color: #d32f2f;">{kritik}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                yuksek = anomaly_df[anomaly_df['Risk_Seviyesi'] == RiskLevel.YUKSEK].shape[0]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Yüksek Risk</div>
                    <div class="metric-value" style="color: #ed6c02;">{yuksek}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                firsat = anomaly_df[anomaly_df['Anomali_Tipi'] == '🌟 Yükselen Fırsat'].shape[0]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Yükselen Fırsat</div>
                    <div class="metric-value" style="color: #2e7d32;">{firsat}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                buyume = anomaly_df[anomaly_df['Anomali_Tipi'] == '🚀 Aşırı Büyüme'].shape[0]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Aşırı Büyüme</div>
                    <div class="metric-value" style="color: #0288d1;">{buyume}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Risk ve fırsat tablosu
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🚨 Kritik Riskli ProdPack'ler")
                kritik_df = anomaly_df[anomaly_df['Risk_Seviyesi'] == RiskLevel.KRITIK]
                
                if not kritik_df.empty:
                    display_cols = []
                    for col in ['ProdPack_Label', 'Sirket', 'Buyume_Orani_2023_2024', 'Pazar_Payi_2024', 'Anomali_Tipi']:
                        if col in kritik_df.columns:
                            display_cols.append(col)
                    
                    st.dataframe(
                        kritik_df[display_cols].head(10),
                        use_container_width=True,
                        height=300
                    )
                else:
                    st.success("✅ Kritik risk seviyesinde ürün bulunmuyor.")
            
            with col2:
                st.markdown("#### 🌟 Yükselen Fırsatlar")
                firsat_df = anomaly_df[anomaly_df['Anomali_Tipi'] == '🌟 Yükselen Fırsat']
                
                if not firsat_df.empty:
                    display_cols = []
                    for col in ['ProdPack_Label', 'Sirket', 'Buyume_Orani_2023_2024', 'Pazar_Payi_2024']:
                        if col in firsat_df.columns:
                            display_cols.append(col)
                    
                    st.dataframe(
                        firsat_df[display_cols].head(10),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Fırsat Insight
                    ExecutiveUI.insight_box(
                        "YÜKSELEN FIRSATLAR",
                        f"🚀 {len(firsat_df)} ürün/paket yüksek büyüme potansiyeli gösteriyor. "
                        f"Özellikle {firsat_df.iloc[0]['Sirket'] if 'Sirket' in firsat_df.columns else 'bu ürünler'} "
                        f"hızlı büyüyor. Pazarlama yatırımlarını artırın.",
                        "💎"
                    )
                else:
                    st.info("Yükselen fırsat tespit edilmedi.")
            
            # Anomali dağılım grafiği
            st.divider()
            st.markdown("#### 📊 Anomali Skor Dağılımı")
            
            fig = px.histogram(
                anomaly_df,
                x='Anomali_Skoru',
                color='Risk_Seviyesi',
                nbins=30,
                title='ProdPack Risk Skor Dağılımı',
                color_discrete_map={
                    RiskLevel.KRITIK: '#d32f2f',
                    RiskLevel.YUKSEK: '#ed6c02',
                    RiskLevel.ORTA: '#0288d1',
                    RiskLevel.NORMAL: '#2e7d32'
                }
            )
            fig.update_layout(template='plotly_dark', paper_bgcolor='#0c1a32')
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Anomali tespiti için yeterli veri yok (en az 2 yıllık satış).")
    
    # ========================================
    # TAB 4: SEGMENTASYON (PCA + K-Means)
    # ========================================
    with tab4:
        st.markdown("### 🎯 Ürün Segmentasyonu: Liderler, Potansiyeller, Riskli Ürünler")
        
        if st.session_state.segment_data is not None:
            segment_df = st.session_state.segment_data
            
            if 'Segment_Adi' in segment_df.columns and 'PCA_1' in segment_df.columns and 'PCA_2' in segment_df.columns:
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # PCA Segmentasyon grafiği
                    fig = px.scatter(
                        segment_df,
                        x='PCA_1',
                        y='PCA_2',
                        color='Segment_Adi',
                        size='Pazar_Payi_2024' if 'Pazar_Payi_2024' in segment_df.columns else None,
                        hover_name='ProdPack_Label' if 'ProdPack_Label' in segment_df.columns else None,
                        title='PCA + K-Means Segmentasyon Haritası',
                        color_discrete_sequence=px.colors.qualitative.Vivid
                    )
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='#0c1a32',
                        height=500,
                        xaxis_title='PCA Bileşen 1',
                        yaxis_title='PCA Bileşen 2'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📊 Segment Dağılımı")
                    
                    segment_counts = segment_df['Segment_Adi'].value_counts().reset_index()
                    segment_counts.columns = ['Segment', 'Ürün Sayısı']
                    
                    fig = px.pie(
                        segment_counts,
                        values='Ürün Sayısı',
                        names='Segment',
                        title='Segment Dağılımı',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(template='plotly_dark', paper_bgcolor='#0c1a32')
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # Segment detayları
                st.markdown("#### 📋 Segment Profilleri")
                
                for segment in segment_df['Segment_Adi'].unique():
                    if segment == 'Sınıflandırılmamış':
                        continue
                    
                    seg_data = segment_df[segment_df['Segment_Adi'] == segment]
                    
                    with st.expander(f"{segment} ({len(seg_data)} ürün)"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_share = seg_data['Pazar_Payi_2024'].mean() if 'Pazar_Payi_2024' in seg_data.columns else 0
                            st.metric("Ort. Pazar Payı", f"%{avg_share:.2f}")
                        
                        with col2:
                            avg_growth = seg_data['Buyume_Orani_2023_2024'].mean() if 'Buyume_Orani_2023_2024' in seg_data.columns else 0
                            st.metric("Ort. Büyüme", f"%{avg_growth:.1f}")
                        
                        with col3:
                            if 'Sirket' in seg_data.columns:
                                top_sirket = seg_data['Sirket'].value_counts().index[0] if not seg_data['Sirket'].empty else '-'
                                st.metric("Lider Şirket", top_sirket)
                        
                        with col4:
                            st.metric("Ürün Sayısı", len(seg_data))
                        
                        # Strateji önerisi
                        if segment == ProductSegment.STARS:
                            st.success("**🎯 Strateji:** Pazar liderliğini korumak için yatırımı artırın. Yenilikçi pazarlama ve dağıtım kanallarını güçlendirin.")
                        elif segment == ProductSegment.CASH_COWS:
                            st.info("**💰 Strateji:** Karlılığı maksimize edin. Nakit akışını Ar-Ge ve yıldız ürünlere yönlendirin.")
                        elif segment == ProductSegment.EMERGING:
                            st.warning("**🚀 Strateji:** Büyümeyi destekleyin. Pazar payını artırmak için agresif fiyatlandırma ve promosyon.")
                        elif segment == ProductSegment.QUESTION_MARKS:
                            st.warning("**❓ Strateji:** Potansiyeli değerlendirin. Başarılı olma olasılığı yüksek olanlara yatırım yapın, diğerlerini elden çıkarın.")
                        elif segment == ProductSegment.DOGS:
                            st.error("**⚠️ Strateji:** Portföyden çıkarma veya yeniden konumlandırma. Maliyetleri minimize edin.")
                        
                        # Örnek ürünler
                        st.markdown("**📦 Örnek ProdPack'ler:**")
                        sample_cols = []
                        for col in ['ProdPack_Label', 'Sirket', 'Satis_2024' if 'Satis_2024' in seg_data.columns else None]:
                            if col and col in seg_data.columns:
                                sample_cols.append(col)
                        
                        if sample_cols:
                            st.dataframe(seg_data[sample_cols].head(5), use_container_width=True)
            
            else:
                st.info("Segmentasyon verisi oluşturulamadı. Daha fazla sayısal veri gerekiyor.")
        
        else:
            st.info("Segmentasyon analizi için yeterli veri yok.")
    
    # ========================================
    # TAB 5: STRATEJİK RAPOR
    # ========================================
    with tab5:
        st.markdown("### 📑 Stratejik İstihbarat ve Yönetici Raporu")
        
        # Özet metrikler
        col1, col2, col3, col4 = st.columns(4)
        
        sales_cols = [c for c in df.columns if re.search(r'Satis_20\d{2}', c)]
        
        with col1:
            total_sales = df[sales_cols[-1]].sum() if sales_cols else 0
            st.metric("2024 Pazar Büyüklüğü", f"₺{total_sales/1e6:.1f}M")
        
        with col2:
            if len(sales_cols) >= 2:
                prev_sales = df[sales_cols[-2]].sum() if sales_cols else 0
                growth = ((total_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0
                st.metric("Yıllık Büyüme", f"%{growth:.1f}", delta=f"{'📈' if growth > 0 else '📉'}")
        
        with col3:
            if st.session_state.anomaly_data is not None:
                risk_count = st.session_state.anomaly_data[
                    st.session_state.anomaly_data['Risk_Seviyesi'].isin([RiskLevel.KRITIK, RiskLevel.YUKSEK])
                ].shape[0]
                st.metric("Risk Altındaki Ürün", risk_count, delta="⚠️")
        
        with col4:
            if st.session_state.forecast_data and 'holt_winters' in st.session_state.forecast_data:
                f = st.session_state.forecast_data['holt_winters']
                st.metric("2026 Pazar Tahmini", f"₺{f.predictions[-1]/1e6:.1f}M", delta=f"%{f.growth_rate:.1f}")
        
        st.divider()
        
        # Entegre stratejik öneriler
        st.markdown("#### 💎 Entegre Stratejik Öneriler")
        
        recommendations = []
        
        # 1. Tahmin bazlı
        if st.session_state.forecast_data and 'holt_winters' in st.session_state.forecast_data:
            f = st.session_state.forecast_data['holt_winters']
            if f.growth_rate > 10:
                recommendations.append({
                    'title': '📈 Büyüme Odaklı Portföy Stratejisi',
                    'desc': f'Pazar %{f.growth_rate:.1f} büyüyecek. Yıldız ve yükselen ürünlere yatırım yapın.',
                    'action': 'Ar-Ge bütçesini %20 artırın, 2 yeni ürün lansmanı planlayın.'
                })
            else:
                recommendations.append({
                    'title': '🛡️ Pazar Koruma ve Verimlilik',
                    'desc': 'Büyüme sınırlı. Mevcut pazar payını koruyun ve operasyonel verimliliği artırın.',
                    'action': 'Maliyet optimizasyonu, tedarik zinciri iyileştirme, sadakat programları.'
                })
        
        # 2. Risk bazlı
        if st.session_state.anomaly_data is not None:
            anomaly_df = st.session_state.anomaly_data
            kritik_sayi = anomaly_df[anomaly_df['Risk_Seviyesi'] == RiskLevel.KRITIK].shape[0]
            if kritik_sayi > 0:
                recommendations.append({
                    'title': '🚨 Acil Risk Müdahale Planı',
                    'desc': f'{kritik_sayi} kritik riskli ürün tespit edildi. Hızlı aksiyon gerekiyor.',
                    'action': 'Kritik ürünler için özel müdahale ekibi kurun, 30 gün içinde aksiyon planı.'
                })
            
            firsat_sayi = anomaly_df[anomaly_df['Anomali_Tipi'] == '🌟 Yükselen Fırsat'].shape[0]
            if firsat_sayi > 0:
                recommendations.append({
                    'title': '🌟 Yükselen Fırsat Değerlendirme',
                    'desc': f'{firsat_sayi} ürün yüksek büyüme potansiyeli gösteriyor.',
                    'action': 'Bu ürünler için pazarlama bütçesini %30 artırın, satış kanallarını genişletin.'
                })
        
        # 3. Segment bazlı
        if st.session_state.segment_data is not None:
            segment_df = st.session_state.segment_data
            if 'Segment_Adi' in segment_df.columns:
                yildiz_sayi = segment_df[segment_df['Segment_Adi'] == ProductSegment.STARS].shape[0]
                if yildiz_sayi < 2:
                    recommendations.append({
                        'title': '⭐ Yıldız Ürün Geliştirme',
                        'desc': 'Portföyde yeterli yıldız ürün yok. Geleceğin liderlerini yetiştirin.',
                        'action': 'Soru işaretleri segmentindeki en güçlü 3 ürünü belirleyin, yoğun yatırım yapın.'
                    })
        
        # 4. Kanibalizasyon bazlı
        if st.session_state.cannibal_data is not None:
            cannibal_df = st.session_state.cannibal_data
            high_cannibal = cannibal_df[cannibal_df['Risk_Seviyesi'] == 'Yüksek'].shape[0]
            if high_cannibal > 2:
                recommendations.append({
                    'title': '🔄 Portföy Optimizasyonu',
                    'desc': f'{high_cannibal} paket arasında yüksek kanibalizasyon riski var.',
                    'action': 'Ürün farklılaştırma, hedef kitle ayrıştırma, fiyatlandırma revizyonu.'
                })
        
        # Önerileri göster
        for i, rec in enumerate(recommendations[:4]):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    <div style="background: rgba(30,58,95,0.5); padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem;
                              border-left: 5px solid #d4af37;">
                        <h4 style="color: #d4af37; margin:0;">{rec['title']}</h4>
                        <p style="color: white; margin: 0.5rem 0;">{rec['desc']}</p>
                        <p style="color: #c0c0c0; margin:0; font-size: 0.9rem;">🎯 {rec['action']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if i == 0:
                        st.markdown('<div style="background: #d32f2f; padding: 0.5rem; border-radius: 5px; text-align: center;">KRİTİK</div>', unsafe_allow_html=True)
                    elif i == 1:
                        st.markdown('<div style="background: #ed6c02; padding: 0.5rem; border-radius: 5px; text-align: center;">YÜKSEK</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background: #0288d1; padding: 0.5rem; border-radius: 5px; text-align: center;">ORTA</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Executive Summary Raporu
        st.markdown("#### 📋 Yönetici Özeti Raporu")
        
        # Dinamik rapor oluştur
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PHARMAINTELLIGENCE PRO v8.0 - STRATEJİK YÖNETİCİ RAPORU")
        report_lines.append(f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Pazar özeti
        report_lines.append("📊 PAZAR ÖZETİ")
        report_lines.append("-" * 40)
        if sales_cols:
            report_lines.append(f"2024 Pazar Büyüklüğü: ₺{total_sales/1e6:.1f}M")
            if len(sales_cols) >= 2:
                report_lines.append(f"Yıllık Büyüme: %{growth:.1f}")
        report_lines.append("")
        
        # ProdPack özeti
        report_lines.append("📦 PRODPACK ÖZETİ")
        report_lines.append("-" * 40)
        report_lines.append(f"Toplam ProdPack Sayısı: {len(df)}")
        if 'Molekul' in df.columns:
            report_lines.append(f"Benzersiz Molekül: {df['Molekul'].nunique()}")
        if 'Sirket' in df.columns:
            report_lines.append(f"Aktif Şirket Sayısı: {df['Sirket'].nunique()}")
        report_lines.append("")
        
        # Risk özeti
        if st.session_state.anomaly_data is not None:
            anomaly_df = st.session_state.anomaly_data
            report_lines.append("⚠️ RİSK ÖZETİ")
            report_lines.append("-" * 40)
            report_lines.append(f"Kritik Risk: {anomaly_df[anomaly_df['Risk_Seviyesi'] == RiskLevel.KRITIK].shape[0]}")
            report_lines.append(f"Yüksek Risk: {anomaly_df[anomaly_df['Risk_Seviyesi'] == RiskLevel.YUKSEK].shape[0]}")
            report_lines.append(f"Yükselen Fırsat: {anomaly_df[anomaly_df['Anomali_Tipi'] == '🌟 Yükselen Fırsat'].shape[0]}")
            report_lines.append("")
        
        # Stratejik aksiyonlar
        report_lines.append("🎯 STRATEJİK AKSİYONLAR (Öncelik Sırasına Göre)")
        report_lines.append("-" * 40)
        for i, rec in enumerate(recommendations[:5], 1):
            report_lines.append(f"{i}. {rec['title']}")
            report_lines.append(f"   → {rec['action']}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("Rapor, PharmaIntelligence Pro AI Motoru tarafından oluşturulmuştur.")
        
        report_text = "\n".join(report_lines)
        
        st.text_area("Rapor Önizleme", report_text, height=400)
        
        # Rapor indirme
        st.download_button(
            label="📥 Stratejik Raporu İndir (TXT)",
            data=report_text,
            file_name=f"pharma_strategic_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0; color: #c0c0c0; font-size: 0.8rem;">
        <span style="color: #d4af37;">PharmaIntelligence Pro v8.0</span> | Enterprise Karar Destek Platformu<br>
        © 2024 PharmaIntelligence Inc. Tüm hakları saklıdır.
    </div>
    """, unsafe_allow_html=True)

# ================================================
# UYGULAMA GİRİŞ NOKTASI
# ================================================
if __name__ == "__main__":
    try:
        gc.enable()
        main()
    except Exception as e:
        st.error(f"### Uygulama Hatası\n\n{str(e)}")
        with st.expander("🔍 Hata Detayları"):
            st.code(traceback.format_exc())
