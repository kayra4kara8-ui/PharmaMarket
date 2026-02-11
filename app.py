"""
PharmaIntelligence Pro v7.0 - Kurumsal Karar Destek Platformu
Versiyon: 7.0.0
Yazar: PharmaIntelligence Inc.
Lisans: Kurumsal

Yapay zeka destekli tahminleme, anomali tespiti, PCA segmentasyonu 
ve stratejik karar destek önerileri ile lider seviye ilaç pazarı analitiği.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Gelişmiş Analitik
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats

# Yardımcı Araçlar
from datetime import datetime, timedelta
import json
from io import BytesIO
import time
import gc
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math
import re
import hashlib
from collections import defaultdict
from enum import Enum
import functools

# Excel/PDF Dışa Aktarım
import xlsxwriter
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_MEVCUT = True
except ImportError:
    REPORTLAB_MEVCUT = False
    st.warning("PDF dışa aktarımı devre dışı: reportlab kurulu değil.")

# ================================================
# 1. TİP TANIMLAMALARI VE ENUMLAR
# ================================================

class RiskSeviyesi(Enum):
    """Risk seviyeleri enum'ı"""
    DUSUK = "Düşük Risk"
    ORTA = "Orta Risk"
    YUKSEK = "Yüksek Risk"
    KRITIK = "Kritik Risk"

class UrunSegmenti(Enum):
    """Ürün segmentasyonu enum'ı"""
    YILDIZ = "Yıldız Ürünler"
    NAKIT_INEGI = "Nakit İnekleri"
    RISKLI = "Riskli Ürünler"
    SORU_ISARETI = "Soru İşaretleri"
    GELISMEKTE = "Gelişmekte Olan"
    DUSUS = "Düşüşte Olan"

class AnalizTuru(Enum):
    """Analiz türleri enum'ı"""
    TAHMIN = "tahmin"
    ANOMALI = "anomali"
    SEGMENTASYON = "segmentasyon"
    REKABET = "rekabet"
    FIYAT = "fiyat"

# ================================================
# 2. DATA ENGINE - VERİ İŞLEME MOTORU
# ================================================

class DataEngine:
    """
    Kurumsal seviye veri işleme motoru.
    Gelişmiş optimizasyon, temizleme ve dönüştürme yetenekleri.
    """
    
    @staticmethod
    def ensure_unique_columns(columns: List[str]) -> List[str]:
        """
        Sütun isimlerini benzersiz hale getir.
        
        Args:
            columns: Orijinal sütun isimleri listesi
            
        Returns:
            Benzersiz sütun isimleri listesi
        """
        seen = {}
        result = []
        
        for col in columns:
            original = col
            counter = 1
            
            # İsim çakışması olana kadar sayı ekle
            while col in seen:
                col = f"{original}_{counter}"
                counter += 1
            
            seen[col] = True
            result.append(col)
        
        return result
    
    @staticmethod
    def sutun_isimleri_temizle(columns: List[str]) -> List[str]:
        """
        Yinelenenleri kaldırarak sütun isimlerini temizle ve standardize et.
        
        Args:
            columns: Sütun isimleri listesi
            
        Returns:
            Temizlenmiş, benzersiz sütun isimleri listesi
        """
        temizlenmis = []
        gorulen_isimler = {}
        
        for sutun in columns:
            if not isinstance(sutun, str):
                sutun = str(sutun)
            
            # Türkçe karakter dönüşümü
            tr_haritalama = {
                'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
            }
            for tr, en in tr_haritalama.items():
                sutun = sutun.replace(tr, en)
            
            # Boşluk temizleme
            sutun = sutun.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            sutun = ' '.join(sutun.split())
            
            # Standart haritalamalar uygula
            sutun = DataEngine._sutun_haritalamalari_uygula(sutun)
            sutun = sutun.strip()
            
            # Benzersiz isimlendirme
            orijinal_sutun = str(sutun)
            if orijinal_sutun in gorulen_isimler:
                gorulen_isimler[orijinal_sutun] += 1
                sutun = f"{orijinal_sutun}_{gorulen_isimler[orijinal_sutun]}"
            else:
                gorulen_isimler[orijinal_sutun] = 0
            
            temizlenmis.append(sutun)
        
        # Son benzersizlik kontrolü
        return DataEngine.ensure_unique_columns(temizlenmis)
    
    @staticmethod
    def _sutun_haritalamalari_uygula(sutun: str) -> str:
        """Alan-spesifik sütun ismi haritalamaları uygula."""
        
        # Regex ile yıl bulma - FIXED
        yil_eslesme = re.search(r'20\d{2}', sutun)
        yil = yil_eslesme.group() if yil_eslesme else None
        
        # Satış sütunları
        if "MAT Q3" in sutun and "USD MNF" in sutun and yil:
            if "Unit Avg Price" not in sutun and "SU Avg Price" not in sutun:
                return f"Satış_{yil}"
        
        # Birim sütunları
        elif "MAT Q3" in sutun and "Units" in sutun and yil:
            return f"Birim_{yil}"
        
        # Ortalama Fiyat sütunları
        elif "MAT Q3" in sutun and "Unit Avg Price USD MNF" in sutun and yil:
            return f"Ort_Fiyat_{yil}"
        
        # Standart Birimler
        elif "MAT Q3" in sutun and "Standard Units" in sutun and yil:
            return f"Standart_Birim_{yil}"
        
        # SB Ortalama Fiyat
        elif "MAT Q3" in sutun and "SU Avg Price USD MNF" in sutun and yil:
            return f"SB_Ort_Fiyat_{yil}"
        
        # Diğer sütunlar
        elif "Source.Name" in sutun:
            return "Kaynak"
        elif "Country" in sutun:
            return "Ulke"
        elif "Sector" in sutun:
            return "Sektor"
        elif "Corporation" in sutun:
            return "Sirket"
        elif "Manufacturer" in sutun:
            return "Uretici"
        elif "Molecule List" in sutun:
            return "Molekul_Listesi"
        elif "Molecule" in sutun:
            return "Molekul"
        elif "Chemical Salt" in sutun:
            return "Kimyasal_Tuz"
        elif "International Product" in sutun:
            return "Uluslararasi_Urun"
        elif "Specialty Product" in sutun:
            return "Ozel_Urun"
        elif "NFC123" in sutun:
            return "NFC123"
        elif "International Pack" in sutun:
            return "Uluslararasi_Paket"
        elif "International Strength" in sutun:
            return "Uluslararasi_Guc"
        elif "International Size" in sutun:
            return "Uluslararasi_Boyut"
        elif "International Volume" in sutun:
            return "Uluslararasi_Hacim"
        elif "International Prescription" in sutun:
            return "Uluslararasi_Recepte"
        elif "Panel" in sutun:
            return "Panel"
        elif "Region" in sutun and "Sub-Region" not in sutun:
            return "Bolge"
        elif "Sub-Region" in sutun:
            return "Alt_Bolge_1"  # Benzersiz isim
        
        return sutun
    
    @staticmethod
    def sutundan_yil_cikar(sutun_adi: str) -> Optional[int]:
        """
        Regex kullanarak sütun adından 4 haneli yıl çıkar.
        
        Args:
            sutun_adi: Sütun adı
            
        Returns:
            Çıkarılan yıl veya None
        """
        try:
            # Regex ile yıl bul
            eslesme = re.search(r'20\d{2}', sutun_adi)
            if eslesme:
                return int(eslesme.group())
            return None
        except:
            return None
    
    @staticmethod
    def dataframe_optimize_et(df: pd.DataFrame) -> pd.DataFrame:
        """
        Akıllı tip dönüşümü ile gelişmiş DataFrame optimizasyonu.
        
        Args:
            df: Girdi DataFrame
            
        Returns:
            Optimize edilmiş DataFrame
        """
        try:
            orijinal_hafiza = df.memory_usage(deep=True).sum() / 1024**2
            
            # Sütun isimlerini temizle ve benzersizleştir
            df.columns = DataEngine.sutun_isimleri_temizle(df.columns)
            
            # Kategorik sütunları optimize et
            for sutun in df.select_dtypes(include=['object']).columns:
                try:
                    benzersiz_sayisi = df[sutun].nunique()
                    toplam_satir = len(df)
                    
                    # Kardinalite <%70 ise kategoriye dönüştür
                    if benzersiz_sayisi < toplam_satir * 0.7:
                        df[sutun] = df[sutun].astype('category')
                except:
                    continue
            
            # Güvenli numerik optimizasyon
            for sutun in df.select_dtypes(include=[np.number]).columns:
                try:
                    # NaN kontrolü
                    if df[sutun].isna().all():
                        continue
                    
                    # Tamsayı optimizasyonu
                    if pd.api.types.is_integer_dtype(df[sutun]):
                        # NaN içermeyen değerler
                        non_nan_vals = df[sutun].dropna()
                        
                        if len(non_nan_vals) == 0:
                            continue
                            
                        sutun_min = non_nan_vals.min()
                        sutun_max = non_nan_vals.max()
                        
                        # Güvenli aralık kontrolü
                        if sutun_min >= 0:
                            if sutun_max <= 255:
                                df[sutun] = df[sutun].astype(np.uint8)
                            elif sutun_max <= 65535:
                                df[sutun] = df[sutun].astype(np.uint16)
                            elif sutun_max <= 4294967295:
                                df[sutun] = df[sutun].astype(np.uint32)
                        else:
                            if sutun_min >= -128 and sutun_max <= 127:
                                df[sutun] = df[sutun].astype(np.int8)
                            elif sutun_min >= -32768 and sutun_max <= 32767:
                                df[sutun] = df[sutun].astype(np.int16)
                            elif sutun_min >= -2147483648 and sutun_max <= 2147483647:
                                df[sutun] = df[sutun].astype(np.int32)
                    
                    # Float optimizasyonu
                    elif pd.api.types.is_float_dtype(df[sutun]):
                        df[sutun] = df[sutun].astype(np.float32)
                        
                except Exception as e:
                    continue
            
            # String temizleme
            for sutun in df.select_dtypes(include=['object']).columns:
                try:
                    df[sutun] = df[sutun].astype(str).str.strip()
                except:
                    pass
            
            optimize_edilmis_hafiza = df.memory_usage(deep=True).sum() / 1024**2
            hafiza_tasarrufu = orijinal_hafiza - optimize_edilmis_hafiza
            
            if hafiza_tasarrufu > 0:
                st.success(f"💾 Hafıza optimizasyonu: {orijinal_hafiza:.1f}MB → {optimize_edilmis_hafiza:.1f}MB (%{hafiza_tasarrufu/orijinal_hafiza*100:.1f} tasarruf)")
            
            return df
            
        except Exception as e:
            st.warning(f"Optimizasyon uyarısı: {str(e)}")
            return df
    
    @staticmethod
    def veri_yukle(dosya: Any, ornek_boyut: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Büyük veri setlerini yükler.
        
        Args:
            dosya: Yüklenen dosya nesnesi
            ornek_boyut: Örnekleme için satır limiti
            
        Returns:
            Optimize edilmiş DataFrame veya None
        """
        try:
            baslangic_zamani = time.time()
            
            if dosya.name.endswith('.csv'):
                if ornek_boyut:
                    df = pd.read_csv(dosya, nrows=ornek_boyut)
                else:
                    with st.spinner("📥 CSV verisi yükleniyor..."):
                        df = pd.read_csv(dosya, low_memory=False)
                        
            elif dosya.name.endswith(('.xlsx', '.xls')):
                if ornek_boyut:
                    df = pd.read_excel(dosya, nrows=ornek_boyut, engine='openpyxl')
                else:
                    with st.spinner(f"📥 Tüm veri seti yükleniyor..."):
                        df = pd.read_excel(dosya, engine='openpyxl')
            else:
                st.error("❌ Desteklenmeyen dosya formatı")
                return None
            
            # DataFrame'i optimize et
            df = DataEngine.dataframe_optimize_et(df)
            
            yukleme_suresi = time.time() - baslangic_zamani
            st.success(f"✅ Veri yüklendi: {len(df):,} satır, {len(df.columns)} sütun ({yukleme_suresi:.2f}s)")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            return None
    
    @staticmethod
    def analitik_veri_hazirla(df: pd.DataFrame) -> pd.DataFrame:
        """
        Hesaplanmış metriklerle analitik için veri hazırla.
        
        Args:
            df: Girdi DataFrame
            
        Returns:
            Hesaplanmış sütunlarla geliştirilmiş DataFrame
        """
        try:
            # Regex ile satış sütunlarını bul
            satis_sutunlari = []
            for sutun in df.columns:
                if re.search(r'Satış_20\d{2}', sutun):
                    satis_sutunlari.append(sutun)
            
            if not satis_sutunlari:
                st.warning("⚠️ Satış sütunu bulunamadı")
                return df
            
            # Yılları çıkar
            yillar = []
            for sutun in satis_sutunlari:
                yil = DataEngine.sutundan_yil_cikar(sutun)
                if yil:
                    yillar.append(yil)
            
            yillar = sorted(set(yillar))
            
            if len(yillar) < 2:
                st.info("ℹ️ Büyüme hesaplaması için en az 2 yıl gerekli")
                return df
            
            # Büyüme oranlarını hesapla
            for i in range(1, len(yillar)):
                onceki_yil = yillar[i-1]
                mevcut_yil = yillar[i]
                
                onceki_sutun = f"Satış_{onceki_yil}"
                mevcut_sutun = f"Satış_{mevcut_yil}"
                
                if onceki_sutun in df.columns and mevcut_sutun in df.columns:
                    buyume_sutun = f'Buyume_{onceki_yil}_{mevcut_yil}'
                    # Güvenli bölme
                    mask = df[onceki_sutun] != 0
                    df.loc[mask, buyume_sutun] = ((df.loc[mask, mevcut_sutun] - df.loc[mask, onceki_sutun]) / df.loc[mask, onceki_sutun]) * 100
                    df.loc[~mask, buyume_sutun] = np.nan
            
            # CAGR (BSBH) hesaplaması
            if len(yillar) >= 2:
                ilk_yil = yillar[0]
                son_yil = yillar[-1]
                ilk_sutun = f"Satış_{ilk_yil}"
                son_sutun = f"Satış_{son_yil}"
                
                if ilk_sutun in df.columns and son_sutun in df.columns:
                    donem_sayisi = son_yil - ilk_yil
                    mask = df[ilk_sutun] > 0
                    df.loc[mask, 'BSBH'] = (np.power(df.loc[mask, son_sutun] / df.loc[mask, ilk_sutun], 1/donem_sayisi) - 1) * 100
                    df.loc[~mask, 'BSBH'] = np.nan
            
            # Pazar payı
            if yillar:
                son_yil = yillar[-1]
                son_satis_sutun = f"Satış_{son_yil}"
                
                if son_satis_sutun in df.columns:
                    toplam_satis = df[son_satis_sutun].sum()
                    if toplam_satis > 0:
                        df['Pazar_Payi'] = (df[son_satis_sutun] / toplam_satis) * 100
            
            # Ortalama fiyatları hesapla
            for yil in yillar:
                satis_sutun = f"Satış_{yil}"
                birim_sutun = f"Birim_{yil}"
                
                if satis_sutun in df.columns and birim_sutun in df.columns:
                    fiyat_sutun = f'Ort_Fiyat_{yil}'
                    if fiyat_sutun not in df.columns:
                        mask = df[birim_sutun] > 0
                        df.loc[mask, fiyat_sutun] = df.loc[mask, satis_sutun] / df.loc[mask, birim_sutun]
                        df.loc[~mask, fiyat_sutun] = np.nan
            
            # Performans skoru
            numerik_sutunlar = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numerik_sutunlar) >= 3:
                try:
                    # Sadece son yılın satış, büyüme ve fiyatını kullan
                    kritik_sutunlar = []
                    if satis_sutunlari:
                        kritik_sutunlar.append(satis_sutunlari[-1])
                    
                    buyume_sutunlari = [sutun for sutun in df.columns if 'Buyume_' in sutun]
                    if buyume_sutunlari:
                        kritik_sutunlar.append(buyume_sutunlari[-1])
                    
                    fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat_' in sutun]
                    if fiyat_sutunlari:
                        kritik_sutunlar.append(fiyat_sutunlari[-1])
                    
                    if len(kritik_sutunlar) >= 2:
                        olceklendirici = StandardScaler()
                        numerik_veri = df[kritik_sutunlar].fillna(0)
                        if len(numerik_veri) > 0:
                            olceklenmis_veri = olceklendirici.fit_transform(numerik_veri)
                            df['Performans_Skoru'] = olceklenmis_veri.mean(axis=1)
                except Exception:
                    pass
            
            return df
            
        except Exception as e:
            st.warning(f"Analitik hazırlama uyarısı: {str(e)}")
            return df

# ================================================
# 3. ANALYTICS ENGINE - ANALİTİK MOTORU
# ================================================

class AnalyticsEngine:
    """
    Gelişmiş analitik motoru.
    AI tahminleme, anomali tespiti, PCA segmentasyonu.
    """
    
    @staticmethod
    def pazar_tahmini_uret(df: pd.DataFrame, donemler: int = 2) -> Optional[Dict]:
        """
        Pazarın gelecek değerlerini tahmin et.
        
        Args:
            df: Girdi DataFrame
            donemler: Tahmin edilecek dönem sayısı
            
        Returns:
            Tahmin sonuçları sözlüğü
        """
        try:
            satis_sutunlari = [sutun for sutun in df.columns if re.search(r'Satış_20\d{2}', sutun)]
            if len(satis_sutunlari) < 3:
                return None
            
            # Yılları çıkar ve sırala
            yillar = []
            for sutun in satis_sutunlari:
                yil = DataEngine.sutundan_yil_cikar(sutun)
                if yil:
                    yillar.append(yil)
            
            yillar = sorted(set(yillar))
            
            if len(yillar) < 3:
                return None
            
            # Yıllara göre toplam satış
            yillik_satis = {}
            for yil in yillar:
                sutun = f"Satış_{yil}"
                if sutun in df.columns:
                    yillik_satis[yil] = df[sutun].sum()
            
            # Zaman serisi oluştur
            zaman_serisi = pd.Series(yillik_satis)
            
            # Exponential Smoothing modeli
            model = ExponentialSmoothing(
                zaman_serisi,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            
            uydurulmus_model = model.fit()
            
            # Tahmin
            son_yil = yillar[-1]
            tahmin_yillari = [son_yil + i + 1 for i in range(donemler)]
            tahmin_degerleri = uydurulmus_model.forecast(steps=donemler)
            
            # Güven aralıkları
            artiklar = uydurulmus_model.fittedvalues - zaman_serisi
            artiklar_std = np.std(artiklar)
            guven_araligi = 1.96 * artiklar_std
            
            # Büyüme oranları
            buyume_oranlari = []
            for i in range(len(tahmin_degerleri)):
                if i == 0:
                    onceki_deger = zaman_serisi.iloc[-1]
                else:
                    onceki_deger = tahmin_degerleri.iloc[i-1]
                
                buyume = ((tahmin_degerleri.iloc[i] - onceki_deger) / onceki_deger * 100) if onceki_deger > 0 else 0
                buyume_oranlari.append(buyume)
            
            sonuc = {
                'tarihsel_yillar': yillar,
                'tarihsel_degerler': list(zaman_serisi.values),
                'tahmin_yillari': tahmin_yillari,
                'tahmin_degerleri': list(tahmin_degerleri.values),
                'alt_sinir': list(tahmin_degerleri.values - guven_araligi),
                'ust_sinir': list(tahmin_degerleri.values + guven_araligi),
                'buyume_oranlari': buyume_oranlari,
                'model_hatasi': float(artiklar_std),
                'pazar_genisleme_orani': np.mean(buyume_oranlari) if buyume_oranlari else 0
            }
            
            return sonuc
            
        except Exception as e:
            st.warning(f"Tahminleme hatası: {str(e)}")
            return None
    
    @staticmethod
    def anomali_tespiti_uret(df: pd.DataFrame, kontaminasyon: float = 0.1) -> Optional[pd.DataFrame]:
        """
        Isolation Forest ile anomali tespiti.
        
        Args:
            df: Girdi DataFrame
            kontaminasyon: Anomali oranı tahmini
            
        Returns:
            Anomali skorları ile DataFrame
        """
        try:
            # Özellik seçimi
            ozellikler = []
            
            satis_sutunlari = [sutun for sutun in df.columns if re.search(r'Satış_20\d{2}', sutun)]
            if satis_sutunlari:
                ozellikler.append(satis_sutunlari[-1])
            
            buyume_sutunlari = [sutun for sutun in df.columns if 'Buyume_' in sutun]
            if buyume_sutunlari:
                ozellikler.append(buyume_sutunlari[-1])
            
            if 'Pazar_Payi' in df.columns:
                ozellikler.append('Pazar_Payi')
            
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat_' in sutun]
            if fiyat_sutunlari:
                ozellikler.append(fiyat_sutunlari[-1])
            
            if len(ozellikler) < 2:
                return None
            
            anomali_verisi = df[ozellikler].fillna(0)
            
            if len(anomali_verisi) < 10:
                return None
            
            # Isolation Forest
            izolasyon_ormani = IsolationForest(
                contamination=kontaminasyon,
                random_state=42,
                n_estimators=100
            )
            
            anomali_tahminleri = izolasyon_ormani.fit_predict(anomali_verisi)
            anomali_skorlari = izolasyon_ormani.score_samples(anomali_verisi)
            
            sonuc_df = df.copy()
            sonuc_df['Anomali_Tahmini'] = anomali_tahminleri
            sonuc_df['Anomali_Skoru'] = anomali_skorlari
            
            # Risk kategorileri
            sonuc_df['Risk_Seviyesi'] = pd.cut(
                sonuc_df['Anomali_Skoru'],
                bins=[-np.inf, -0.5, -0.2, 0, 0.5, np.inf],
                labels=['Kritik Risk', 'Yüksek Risk', 'Orta Risk', 'Düşük Risk', 'Normal']
            )
            
            # Anomali tipi belirleme
            sonuc_df['Anomali_Tipi'] = sonuc_df.apply(
                lambda row: AnalyticsEngine._anomali_tipi_belirle(row, satis_sutunlari, buyume_sutunlari, fiyat_sutunlari),
                axis=1
            )
            
            return sonuc_df
            
        except Exception as e:
            st.warning(f"Anomali tespiti hatası: {str(e)}")
            return None
    
    @staticmethod
    def _anomali_tipi_belirle(row, satis_sutunlari, buyume_sutunlari, fiyat_sutunlari) -> str:
        """Anomali tipini belirle."""
        if row['Risk_Seviyesi'] not in ['Kritik Risk', 'Yüksek Risk']:
            return 'Normal'
        
        son_satis = satis_sutunlari[-1] if satis_sutunlari else None
        son_buyume = buyume_sutunlari[-1] if buyume_sutunlari else None
        son_fiyat = fiyat_sutunlari[-1] if fiyat_sutunlari else None
        
        sinyaller = []
        
        if son_satis and pd.notnull(row.get(son_satis)):
            satis_zscore = abs((row[son_satis] - row[son_satis].mean()) / row[son_satis].std() if row[son_satis].std() > 0 else 0)
            if satis_zscore > 3:
                sinyaller.append('Aşırı Satış')
        
        if son_buyume and pd.notnull(row.get(son_buyume)):
            if row[son_buyume] > 50:
                sinyaller.append('Aşırı Büyüme')
            elif row[son_buyume] < -30:
                sinyaller.append('Aşırı Daralma')
        
        if son_fiyat and pd.notnull(row.get(son_fiyat)):
            fiyat_zscore = abs((row[son_fiyat] - row[son_fiyat].mean()) / row[son_fiyat].std() if row[son_fiyat].std() > 0 else 0)
            if fiyat_zscore > 3:
                sinyaller.append('Aşırı Fiyat')
        
        if not sinyaller:
            return 'Genel Anomali'
        
        return ', '.join(sinyaller[:2])
    
    @staticmethod
    def pca_segmentasyonu_uret(df: pd.DataFrame, kume_sayisi: int = 4) -> Optional[pd.DataFrame]:
        """
        PCA tabanlı ürün segmentasyonu.
        
        Args:
            df: Girdi DataFrame
            kume_sayisi: Küme sayısı
            
        Returns:
            Segmentlenmiş DataFrame
        """
        try:
            # Özellik seçimi
            ozellikler = []
            
            satis_sutunlari = [sutun for sutun in df.columns if re.search(r'Satış_20\d{2}', sutun)]
            if satis_sutunlari:
                ozellikler.append(satis_sutunlari[-1])  # Son satış
            
            buyume_sutunlari = [sutun for sutun in df.columns if 'Buyume_' in sutun]
            if buyume_sutunlari:
                ozellikler.append(buyume_sutunlari[-1])  # Son büyüme
            
            if 'Pazar_Payi' in df.columns:
                ozellikler.append('Pazar_Payi')
            
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat_' in sutun]
            if fiyat_sutunlari:
                ozellikler.append(fiyat_sutunlari[-1])  # Son fiyat
            
            if len(ozellikler) < 3:
                return None
            
            segmentasyon_verisi = df[ozellikler].fillna(0)
            
            if len(segmentasyon_verisi) < kume_sayisi * 10:
                return None
            
            # Ölçeklendirme
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(segmentasyon_verisi)
            
            # PCA
            pca = PCA(n_components=min(3, len(ozellikler)))
            X_pca = pca.fit_transform(X_scaled)
            
            # K-Means kümeleme
            kmeans = KMeans(n_clusters=kume_sayisi, random_state=42, n_init=10)
            kumeler = kmeans.fit_predict(X_pca)
            
            sonuc_df = df.copy()
            sonuc_df['Kume'] = kumeler
            sonuc_df['PCA_1'] = X_pca[:, 0] if X_pca.shape[1] > 0 else 0
            sonuc_df['PCA_2'] = X_pca[:, 1] if X_pca.shape[1] > 1 else 0
            
            # Segment isimlendirme
            segment_isimleri = AnalyticsEngine._segment_isimlendir(kumeler, segmentasyon_verisi)
            sonuc_df['Segment'] = sonuc_df['Kume'].map(segment_isimleri)
            
            # Segment özellikleri
            segment_ozellikleri = {}
            for kume in range(kume_sayisi):
                kume_verisi = segmentasyon_verisi[sonuc_df['Kume'] == kume]
                if len(kume_verisi) > 0:
                    segment_ozellikleri[kume] = {
                        'ortalama_satis': kume_verisi[ozellikler[0]].mean(),
                        'ortalama_buyume': kume_verisi[ozellikler[1]].mean() if len(ozellikler) > 1 else 0,
                        'ortalama_pazar_payi': kume_verisi['Pazar_Payi'].mean() if 'Pazar_Payi' in ozellikler else 0
                    }
            
            sonuc_df.attrs['segment_ozellikleri'] = segment_ozellikleri
            sonuc_df.attrs['pca_aciklanan_varyans'] = pca.explained_variance_ratio_.sum()
            
            return sonuc_df
            
        except Exception as e:
            st.warning(f"Segmentasyon hatası: {str(e)}")
            return None
    
    @staticmethod
    def _segment_isimlendir(kumeler, veri) -> Dict:
        """Kümelere stratejik isimler ver."""
        kume_istatistikleri = {}
        
        for kume in np.unique(kumeler):
            kume_verisi = veri[kumeler == kume]
            
            if len(kume_verisi) == 0:
                kume_istatistikleri[kume] = {'isim': f'Kume_{kume}'}
                continue
            
            # Ortalama değerler
            ortalama_satis = kume_verisi.iloc[:, 0].mean() if len(kume_verisi.columns) > 0 else 0
            ortalama_buyume = kume_verisi.iloc[:, 1].mean() if len(kume_verisi.columns) > 1 else 0
            
            # Segment belirleme
            if ortalama_satis > veri.iloc[:, 0].quantile(0.75) and ortalama_buyume > 10:
                isim = UrunSegmenti.YILDIZ.value
            elif ortalama_satis > veri.iloc[:, 0].quantile(0.75) and ortalama_buyume < 5:
                isim = UrunSegmenti.NAKIT_INEGI.value
            elif ortalama_buyume > 20:
                isim = UrunSegmenti.GELISMEKTE.value
            elif ortalama_buyume < -10:
                isim = UrunSegmenti.DUSUS.value
            elif ortalama_satis < veri.iloc[:, 0].quantile(0.25):
                isim = UrunSegmenti.SORU_ISARETI.value
            else:
                isim = UrunSegmenti.RISKLI.value
            
            kume_istatistikleri[kume] = {'isim': isim, 'satis': ortalama_satis, 'buyume': ortalama_buyume}
        
        # Haritalama
        return {k: v['isim'] for k, v in kume_istatistikleri.items()}
    
    @staticmethod
    def metrikleri_hesapla(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Kapsamlı pazar metriklerini hesapla.
        
        Args:
            df: Girdi DataFrame
            
        Returns:
            Metrikler sözlüğü
        """
        metrikler = {}
        
        try:
            metrikler['Toplam_Satir'] = len(df)
            metrikler['Toplam_Sutun'] = len(df.columns)
            
            # Satış metrikleri
            satis_sutunlari = [sutun for sutun in df.columns if re.search(r'Satış_20\d{2}', sutun)]
            if satis_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                yil = DataEngine.sutundan_yil_cikar(son_satis_sutun)
                
                metrikler['Son_Satis_Yili'] = yil
                metrikler['Toplam_Pazar_Degeri'] = float(df[son_satis_sutun].sum())
                metrikler['Urun_Basi_Ort_Satis'] = float(df[son_satis_sutun].mean())
                metrikler['Medyan_Satis'] = float(df[son_satis_sutun].median())
                metrikler['Satis_Std'] = float(df[son_satis_sutun].std())
                metrikler['Satis_CBA'] = float(df[son_satis_sutun].quantile(0.75) - df[son_satis_sutun].quantile(0.25))
            
            # Büyüme metrikleri
            buyume_sutunlari = [sutun for sutun in df.columns if 'Buyume_' in sutun]
            if buyume_sutunlari:
                son_buyume_sutun = buyume_sutunlari[-1]
                metrikler['Ort_Buyume_Orani'] = float(df[son_buyume_sutun].mean())
                metrikler['Pozitif_Buyume_Urunleri'] = int((df[son_buyume_sutun] > 0).sum())
                metrikler['Yuksek_Buyume_Urunleri'] = int((df[son_buyume_sutun] > 20).sum())
            
            # Pazar konsantrasyonu
            if 'Sirket' in df.columns and satis_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                sirket_satisleri = df.groupby('Sirket')[son_satis_sutun].sum()
                toplam_satis = sirket_satisleri.sum()
                
                if toplam_satis > 0:
                    pazar_paylari = (sirket_satisleri / toplam_satis * 100)
                    metrikler['HHI_Indeksi'] = float((pazar_paylari ** 2).sum())
                    
                    for n in [3, 5, 10]:
                        if len(sirket_satisleri) >= n:
                            metrikler[f'Ilk_{n}_Pay'] = float(sirket_satisleri.nlargest(n).sum() / toplam_satis * 100)
            
            # Fiyat metrikleri
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat_' in sutun]
            if fiyat_sutunlari:
                son_fiyat_sutun = fiyat_sutunlari[-1]
                metrikler['Ort_Fiyat'] = float(df[son_fiyat_sutun].mean())
                metrikler['Fiyat_Varyansi'] = float(df[son_fiyat_sutun].var())
            
            return metrikler
            
        except Exception as e:
            st.warning(f"Metrik hesaplama uyarısı: {str(e)}")
            return metrikler
    
    @staticmethod
    def icgoruleri_uret(df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Stratejik içgörüler üret.
        
        Args:
            df: Girdi DataFrame
            
        Returns:
            İçgörü listesi
        """
        icgoruler = []
        
        try:
            satis_sutunlari = [sutun for sutun in df.columns if re.search(r'Satış_20\d{2}', sutun)]
            if not satis_sutunlari:
                return icgoruler
            
            son_satis_sutun = satis_sutunlari[-1]
            yil = DataEngine.sutundan_yil_cikar(son_satis_sutun) or "Son Yıl"
            
            # En iyi ürünler
            en_iyi_urunler = df.nlargest(5, son_satis_sutun)
            toplam_satis = df[son_satis_sutun].sum()
            
            if toplam_satis > 0:
                en_iyi_pay = (en_iyi_urunler[son_satis_sutun].sum() / toplam_satis * 100)
                icgoruler.append({
                    'tur': 'basarili',
                    'baslik': f'🏆 Pazar Liderleri - {yil}',
                    'aciklama': f"İlk 5 ürün pazarın %{en_iyi_pay:.1f}'ini kontrol ediyor.",
                    'oneri': 'Lider ürünlerin pazar payını koruma stratejileri geliştirin.'
                })
            
            # Hızlı büyüyenler
            buyume_sutunlari = [sutun for sutun in df.columns if 'Buyume_' in sutun]
            if buyume_sutunlari:
                son_buyume_sutun = buyume_sutunlari[-1]
                hizli_buyuyen = df.nlargest(5, son_buyume_sutun)
                ort_buyume = hizli_buyuyen[son_buyume_sutun].mean()
                
                icgoruler.append({
                    'tur': 'bilgi',
                    'baslik': f'🚀 Yükselen Yıldızlar',
                    'aciklama': f"En hızlı büyüyen 5 ürün ortalama %{ort_buyume:.1f} büyüme kaydediyor.",
                    'oneri': 'Bu ürünlere yatırımı artırarak gelecekteki pazar liderlerini şekillendirin.'
                })
            
            # Pazar yapısı
            if 'HHI_Indeksi' in AnalyticsEngine.metrikleri_hesapla(df):
                hhi = AnalyticsEngine.metrikleri_hesapla(df)['HHI_Indeksi']
                if hhi > 2500:
                    durum = "Monopolistik"
                    oneri = "Rekabeti artırmak için yeni oyunculara fırsatlar yaratın."
                elif hhi > 1800:
                    durum = "Oligopol"
                    oneri = "Pazar payı koruma ve stratejik ittifaklar geliştirin."
                else:
                    durum = "Rekabetçi"
                    oneri = "Farklılaşma ve inovasyonla öne çıkın."
                
                icgoruler.append({
                    'tur': 'uyari',
                    'baslik': f'🏢 Pazar Yapısı Analizi',
                    'aciklama': f"HHI İndeksi: {hhi:.0f} ({durum} pazar)",
                    'oneri': oneri
                })
            
            return icgoruler
            
        except Exception as e:
            st.warning(f"İçgörü üretme uyarısı: {str(e)}")
            return []

# ================================================
# 4. VIZ ENGINE - GÖRSELLEŞTİRME MOTORU
# ================================================

class VizEngine:
    """
    Profesyonel görselleştirme motoru.
    Kurumsal seviye grafikler ve gösterge panelleri.
    """
    
    @staticmethod
    def metrik_kartlari_olustur(metrikler: Dict[str, Any]) -> None:
        """Metrik kartları oluştur."""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            toplam_satis = metrikler.get('Toplam_Pazar_Degeri', 0)
            satis_yili = metrikler.get('Son_Satis_Yili', '')
            st.markdown(f"""
            <div class="cam-kart metrik-kart birincil">
                <div class="metrik-etiket">TOPLAM PAZAR</div>
                <div class="metrik-deger">${toplam_satis/1e6:.1f}M</div>
                <div class="metrik-detay">{satis_yili} Pazar Değeri</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ort_buyume = metrikler.get('Ort_Buyume_Orani', 0)
            st.markdown(f"""
            <div class="cam-kart metrik-kart {'pozitif' if ort_buyume > 0 else 'negatif'}">
                <div class="metrik-etiket">ORT. BÜYÜME</div>
                <div class="metrik-deger">{ort_buyume:+.1f}%</div>
                <div class="metrik-detay">Yıllık Büyüme Oranı</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            hhi = metrikler.get('HHI_Indeksi', 0)
            st.markdown(f"""
            <div class="cam-kart metrik-kart {'yuksek-risk' if hhi > 2500 else 'orta-risk' if hhi > 1800 else 'dusuk-risk'}">
                <div class="metrik-etiket">REKABET DÜZEYİ</div>
                <div class="metrik-deger">{hhi:.0f}</div>
                <div class="metrik-detay">HHI İndeksi</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            yuksek_buyume = metrikler.get('Yuksek_Buyume_Urunleri', 0)
            toplam_urun = metrikler.get('Toplam_Satir', 1)
            yuzde = (yuksek_buyume / toplam_urun * 100) if toplam_urun > 0 else 0
            st.markdown(f"""
            <div class="cam-kart metrik-kart basarili">
                <div class="metrik-etiket">YÜKSEK BÜYÜME</div>
                <div class="metrik-deger">{yuksek_buyume}</div>
                <div class="metrik-detay">%{yuzde:.1f} Oranında</div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def tahmin_grafigi_olustur(tahmin_sonuclari: Dict) -> Optional[go.Figure]:
        """Tahmin grafiği oluştur."""
        
        try:
            fig = go.Figure()
            
            # Tarihsel veri
            fig.add_trace(go.Scatter(
                x=tahmin_sonuclari['tarihsel_yillar'],
                y=tahmin_sonuclari['tarihsel_degerler'],
                mode='lines+markers',
                name='Tarihsel',
                line=dict(color='#2d7dd2', width=3),
                marker=dict(size=10)
            ))
            
            # Tahmin
            fig.add_trace(go.Scatter(
                x=tahmin_sonuclari['tahmin_yillari'],
                y=tahmin_sonuclari['tahmin_degerleri'],
                mode='lines+markers',
                name='Tahmin',
                line=dict(color='#2acaea', width=3, dash='dash'),
                marker=dict(size=10)
            ))
            
            # Güven aralığı
            fig.add_trace(go.Scatter(
                x=tahmin_sonuclari['tahmin_yillari'] + tahmin_sonuclari['tahmin_yillari'][::-1],
                y=tahmin_sonuclari['ust_sinir'] + tahmin_sonuclari['alt_sinir'][::-1],
                fill='toself',
                fillcolor='rgba(42, 202, 234, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='%95 Güven Aralığı'
            ))
            
            fig.update_layout(
                title='Pazar Tahmini ve Güven Aralıkları',
                xaxis_title='Yıl',
                yaxis_title='Toplam Pazar Değeri (USD)',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Tahmin grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def anomali_grafigi_olustur(anomali_df: pd.DataFrame) -> Optional[go.Figure]:
        """Anomali tespiti grafiği oluştur."""
        
        try:
            if 'Anomali_Skoru' not in anomali_df.columns:
                return None
            
            satis_sutunlari = [sutun for sutun in anomali_df.columns if re.search(r'Satış_20\d{2}', sutun)]
            buyume_sutunlari = [sutun for sutun in anomali_df.columns if 'Buyume_' in sutun]
            
            if not satis_sutunlari or not buyume_sutunlari:
                return None
            
            son_satis_sutun = satis_sutunlari[-1]
            son_buyume_sutun = buyume_sutunlari[-1]
            
            # Örnekleme
            cizim_df = anomali_df if len(anomali_df) <= 1000 else anomali_df.sample(1000, random_state=42)
            
            fig = px.scatter(
                cizim_df,
                x=son_satis_sutun,
                y=son_buyume_sutun,
                color='Risk_Seviyesi',
                size=abs(cizim_df['Anomali_Skoru']) * 20,
                hover_name='Molekul' if 'Molekul' in cizim_df.columns else None,
                title='Risk Analizi - Satış vs Büyüme',
                labels={
                    son_satis_sutun: 'Satış (USD)',
                    son_buyume_sutun: 'Büyüme (%)'
                },
                color_discrete_map={
                    'Kritik Risk': '#ff0000',
                    'Yüksek Risk': '#ff6b6b',
                    'Orta Risk': '#ffd93d',
                    'Düşük Risk': '#6bcf7f',
                    'Normal': '#4ecdc4'
                }
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Anomali grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def pca_segmentasyon_grafigi(segment_df: pd.DataFrame) -> Optional[go.Figure]:
        """PCA segmentasyon grafiği oluştur."""
        
        try:
            if 'PCA_1' not in segment_df.columns or 'PCA_2' not in segment_df.columns:
                return None
            
            fig = px.scatter(
                segment_df,
                x='PCA_1',
                y='PCA_2',
                color='Segment',
                hover_name='Molekul' if 'Molekul' in segment_df.columns else None,
                title='PCA Tabanlı Ürün Segmentasyonu',
                labels={'PCA_1': 'Birinci Temel Bileşen', 'PCA_2': 'İkinci Temel Bileşen'}
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Segmentasyon grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def sunburst_grafigi_olustur(df: pd.DataFrame) -> Optional[go.Figure]:
        """Sunburst hiyerarşi grafiği oluştur."""
        
        try:
            if 'Sirket' not in df.columns or 'Molekul' not in df.columns:
                return None
            
            satis_sutunlari = [sutun for sutun in df.columns if re.search(r'Satış_20\d{2}', sutun)]
            if not satis_sutunlari:
                return None
            
            son_satis_sutun = satis_sutunlari[-1]
            
            # Şirket bazlı toplamlar
            sirket_toplamlari = df.groupby('Sirket')[son_satis_sutun].sum().reset_index()
            sirket_toplamlari.columns = ['id', 'value']
            sirket_toplamlari['parent'] = ''
            
            # Molekül bazlı veriler
            molekul_verileri = df.groupby(['Sirket', 'Molekul'])[son_satis_sutun].sum().reset_index()
            molekul_verileri.columns = ['id', 'parent', 'value']
            
            # Birleştirme
            hiyerarsi_df = pd.concat([
                sirket_toplamlari,
                molekul_verileri[['id', 'parent', 'value']]
            ], ignore_index=True)
            
            fig = go.Figure(go.Sunburst(
                labels=hiyerarsi_df['id'],
                parents=hiyerarsi_df['parent'],
                values=hiyerarsi_df['value'],
                branchvalues="total",
                maxdepth=2,
                insidetextorientation='radial'
            ))
            
            fig.update_layout(
                title='Pazar Hiyerarşisi - Şirket > Molekül',
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Sunburst grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def radar_grafigi_olustur(df: pd.DataFrame, sirketler: List[str]) -> Optional[go.Figure]:
        """Radar karşılaştırma grafiği oluştur."""
        
        try:
            if not sirketler or 'Sirket' not in df.columns:
                return None
            
            # Metrik seçimi
            metrikler = []
            satis_sutunlari = [sutun for sutun in df.columns if re.search(r'Satış_20\d{2}', sutun)]
            buyume_sutunlari = [sutun for sutun in df.columns if 'Buyume_' in sutun]
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat_' in sutun]
            
            if satis_sutunlari:
                metrikler.append(('Satış', satis_sutunlari[-1]))
            if buyume_sutunlari:
                metrikler.append(('Büyüme', buyume_sutunlari[-1]))
            if fiyat_sutunlari:
                metrikler.append(('Fiyat', fiyat_sutunlari[-1]))
            if 'Pazar_Payi' in df.columns:
                metrikler.append(('Pazar Payı', 'Pazar_Payi'))
            
            if len(metrikler) < 3:
                return None
            
            fig = go.Figure()
            
            for sirket in sirketler[:5]:  # En fazla 5 şirket
                sirket_df = df[df['Sirket'] == sirket]
                if len(sirket_df) == 0:
                    continue
                
                degerler = []
                for metrik_adi, sutun_adi in metrikler:
                    if sutun_adi in sirket_df.columns:
                        degerler.append(sirket_df[sutun_adi].mean())
                    else:
                        degerler.append(0)
                
                # Normalizasyon
                max_degerler = [df[sutun].max() for _, sutun in metrikler]
                normalize_edilmis = [(v / m * 100) if m > 0 else 0 for v, m in zip(degerler, max_degerler)]
                
                fig.add_trace(go.Scatterpolar(
                    r=normalize_edilmis,
                    theta=[isim for isim, _ in metrikler],
                    fill='toself',
                    name=sirket
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title='Şirket Performans Karşılaştırması',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Radar grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def karar_destek_notu_olustur(baslik: str, icerik: str, oneri: str, tur: str = 'bilgi') -> None:
        """Karar destek notu oluştur."""
        
        ikonlar = {
            'bilgi': 'ℹ️',
            'basarili': '✅',
            'uyari': '⚠️',
            'tehlike': '🚨',
            'oneri': '💡'
        }
        
        ikon = ikonlar.get(tur, '💡')
        
        st.markdown(f"""
        <div class="cam-kart karar-destek-notu {tur}">
            <div class="karar-destek-baslik">
                {ikon} <strong>{baslik}</strong>
            </div>
            <div class="karar-destek-icerik">
                {icerik}
            </div>
            <div class="karar-destek-oneri">
                <strong>📋 Stratejik Öneri:</strong> {oneri}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================================================
# 5. PHARMA UI - KULLANICI ARAYÜZÜ
# ================================================

class PharmaUI:
    """Ana kullanıcı arayüzü yöneticisi."""
    
    @staticmethod
    def baslik_goster():
        """Ana başlığı göster."""
        st.markdown("""
        <div class="animate-fade-in">
            <h1 class="pharma-baslik">💊 PHARMAINTELLIGENCE PRO v7.0</h1>
            <p class="pharma-alt-baslik">
            Kurumsal Karar Destek Platformu • AI Tahminleme • Risk Analizi • Stratejik İçgörüler
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def yan_cubugu_olustur():
        """Yan çubuğu oluştur."""
        with st.sidebar:
            st.markdown('<h2 class="sidebar-title">🎛️ KONTROL PANELİ</h2>', unsafe_allow_html=True)
            
            with st.expander("📁 VERİ YÜKLEME", expanded=True):
                yuklenen_dosya = st.file_uploader(
                    "Excel/CSV Dosyası Yükle",
                    type=['xlsx', 'xls', 'csv'],
                    help="Profesyonel veri analizi için"
                )
                
                if yuklenen_dosya:
                    st.info(f"📂 {yuklenen_dosya.name}")
                    
                    if st.button("🚀 Veriyi Yükle & Analiz Et", type="primary", use_container_width=True):
                        with st.spinner("Veri seti işleniyor..."):
                            veri = DataEngine.veri_yukle(yuklenen_dosya)
                            
                            if veri is not None and len(veri) > 0:
                                veri = DataEngine.analitik_veri_hazirla(veri)
                                
                                # Oturum durumunu güncelle
                                st.session_state.veri = veri
                                st.session_state.filtrelenmis_veri = veri
                                st.session_state.metrikler = AnalyticsEngine.metrikleri_hesapla(veri)
                                st.session_state.icgoruler = AnalyticsEngine.icgoruleri_uret(veri)
                                
                                st.success(f"✅ {len(veri):,} satır analize hazır!")
                                st.rerun()
            
            if 'veri' in st.session_state and st.session_state.veri is not None:
                PharmaUI._filtreler_olustur()
            
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; font-size: 0.8rem; color: #64748b;">
            <strong>PharmaIntelligence Pro v7.0</strong><br>
            Enterprise Decision Support<br>
            © 2024 Tüm hakları saklıdır
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def _filtreler_olustur():
        """Filtreleri oluştur."""
        with st.sidebar.expander("🎯 GELİŞMİŞ FİLTRELEME", expanded=False):
            veri = st.session_state.veri
            
            # Global arama
            arama_terimi = st.text_input(
                "🔎 Global Arama",
                placeholder="Molekül, Şirket, Ülke...",
                key="global_arama"
            )
            
            # Ülke filtresi
            if 'Ulke' in veri.columns:
                ulkeler = sorted(veri['Ulke'].dropna().unique().tolist())
                secili_ulkeler = st.multiselect(
                    "🌍 Ülkeler",
                    ulkeler,
                    default=ulkeler[:min(5, len(ulkeler))],
                    help="Çoklu seçim"
                )
            
            # Şirket filtresi
            if 'Sirket' in veri.columns:
                sirketler = sorted(veri['Sirket'].dropna().unique().tolist())
                secili_sirketler = st.multiselect(
                    "🏢 Şirketler",
                    sirketler,
                    default=sirketler[:min(5, len(sirketler))],
                    help="Çoklu seçim"
                )
            
            # Satış filtresi
            satis_sutunlari = [sutun for sutun in veri.columns if re.search(r'Satış_20\d{2}', sutun)]
            if satis_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                min_satis = float(veri[son_satis_sutun].min())
                max_satis = float(veri[son_satis_sutun].max())
                
                satis_araligi = st.slider(
                    f"Satış Filtresi ({son_satis_sutun})",
                    min_value=min_satis,
                    max_value=max_satis,
                    value=(min_satis, max_satis),
                    key="satis_filtre"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Filtrele", use_container_width=True):
                    PharmaUI._filtreleri_uygula(
                        arama_terimi,
                        secili_ulkeler if 'secili_ulkeler' in locals() else None,
                        secili_sirketler if 'secili_sirketler' in locals() else None,
                        satis_araligi if 'satis_araligi' in locals() else None
                    )
            
            with col2:
                if st.button("🗑️ Temizle", use_container_width=True):
                    PharmaUI._filtreleri_temizle()
    
    @staticmethod
    def _filtreleri_uygula(arama_terimi, ulkeler, sirketler, satis_araligi):
        """Filtreleri uygula."""
        if 'veri' not in st.session_state:
            return
        
        df = st.session_state.veri.copy()
        
        # Arama
        if arama_terimi:
            mask = pd.Series(False, index=df.index)
            for sutun in df.columns:
                try:
                    mask = mask | df[sutun].astype(str).str.contains(arama_terimi, case=False, na=False)
                except:
                    continue
            df = df[mask]
        
        # Ülke filtresi
        if ulkeler and len(ulkeler) > 0:
            df = df[df['Ulke'].isin(ulkeler)]
        
        # Şirket filtresi
        if sirketler and len(sirketler) > 0:
            df = df[df['Sirket'].isin(sirketler)]
        
        # Satış filtresi
        if satis_araligi:
            satis_sutunlari = [sutun for sutun in df.columns if re.search(r'Satış_20\d{2}', sutun)]
            if satis_sutunlari:
                min_val, max_val = satis_araligi
                df = df[(df[satis_sutunlari[-1]] >= min_val) & (df[satis_sutunlari[-1]] <= max_val)]
        
        st.session_state.filtrelenmis_veri = df
        st.session_state.metrikler = AnalyticsEngine.metrikleri_hesapla(df)
        st.session_state.icgoruler = AnalyticsEngine.icgoruleri_uret(df)
        
        st.success(f"✅ Filtrelenmiş: {len(df):,} satır")
        st.rerun()
    
    @staticmethod
    def _filtreleri_temizle():
        """Filtreleri temizle."""
        if 'veri' in st.session_state:
            st.session_state.filtrelenmis_veri = st.session_state.veri.copy()
            st.session_state.metrikler = AnalyticsEngine.metrikleri_hesapla(st.session_state.veri)
            st.session_state.icgoruler = AnalyticsEngine.icgoruleri_uret(st.session_state.veri)
            st.success("✅ Filtreler temizlendi")
            st.rerun()
    
    @staticmethod
    def ana_icerik_goster():
        """Ana içeriği göster."""
        if 'veri' not in st.session_state or st.session_state.veri is None:
            PharmaUI._hosgeldin_ekrani()
            return
        
        veri = st.session_state.filtrelenmis_veri
        metrikler = st.session_state.metrikler
        
        # Sekmeler
        sekme1, sekme2, sekme3, sekme4, sekme5, sekme6, sekme7 = st.tabs([
            "📊 ÖZET",
            "🔮 TAHMİNLEME",
            "⚠️ RİSK ANALİZİ",
            "🎯 SEGMENTASYON",
            "📈 GÖRSELLEŞTİRME",
            "📋 VERİ ANALİZİ",
            "📑 RAPORLAMA"
        ])
        
        with sekme1:
            PharmaUI._ozet_sekmesi(veri, metrikler)
        
        with sekme2:
            PharmaUI._tahminleme_sekmesi(veri)
        
        with sekme3:
            PharmaUI._risk_analizi_sekmesi(veri)
        
        with sekme4:
            PharmaUI._segmentasyon_sekmesi(veri)
        
        with sekme5:
            PharmaUI._gorsellestirme_sekmesi(veri)
        
        with sekme6:
            PharmaUI._veri_analizi_sekmesi(veri)
        
        with sekme7:
            PharmaUI._raporlama_sekmesi(veri, metrikler)
    
    @staticmethod
    def _hosgeldin_ekrani():
        """Hoşgeldin ekranı."""
        st.markdown("""
        <div class="hosgeldin-container">
            <div class="hosgeldin-icon">💊</div>
            <h2 style="color: #f1f5f9; margin-bottom: 1rem;">PharmaIntelligence Pro v7.0</h2>
            <p style="color: #cbd5e1; margin-bottom: 2rem; line-height: 1.6;">
            Kurumsal Karar Destek Platformuna hoş geldiniz.<br>
            Verinizi yükleyerek AI tahminleme, risk analizi ve stratejik içgörülere erişin.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def _ozet_sekmesi(df: pd.DataFrame, metrikler: Dict):
        """Özet sekmesi."""
        st.markdown('<h2 class="bolum-baslik">📊 Performans Göstergeleri</h2>', unsafe_allow_html=True)
        
        # Metrik kartları
        VizEngine.metrik_kartlari_olustur(metrikler)
        
        # İçgörüler
        st.markdown('<h3 class="alt-bolum-baslik">💡 Stratejik İçgörüler</h3>', unsafe_allow_html=True)
        
        if 'icgoruler' in st.session_state and st.session_state.icgoruler:
            for icgoru in st.session_state.icgoruler[:4]:
                VizEngine.karar_destek_notu_olustur(
                    icgoru['baslik'],
                    icgoru['aciklama'],
                    icgoru.get('oneri', ''),
                    icgoru['tur']
                )
        
        # Veri önizleme
        st.markdown('<h3 class="alt-bolum-baslik">📋 Veri Önizleme</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            satir_sayisi = st.slider("Satır Sayısı", 10, 5000, 100, 10)
            
            # Önemli sütunlar
            onemli_sutunlar = []
            for sutun in ['Molekul', 'Sirket', 'Ulke', 'Satış_2024', 'Buyume_2023_2024', 'Pazar_Payi']:
                if sutun in df.columns:
                    onemli_sutunlar.append(sutun)
            
            if len(onemli_sutunlar) < 5:
                onemli_sutunlar.extend([col for col in df.columns[:5] if col not in onemli_sutunlar])
            
            secili_sutunlar = st.multiselect(
                "Sütunlar",
                df.columns.tolist(),
                default=onemli_sutunlar[:min(5, len(onemli_sutunlar))]
            )
        
        with col2:
            if secili_sutunlar:
                gosterim_df = df[secili_sutunlar].head(satir_sayisi)
            else:
                gosterim_df = df.head(satir_sayisi)
            
            st.dataframe(gosterim_df, use_container_width=True, height=400)
    
    @staticmethod
    def _tahminleme_sekmesi(df: pd.DataFrame):
        """Tahminleme sekmesi."""
        st.markdown('<h2 class="bolum-baslik">🔮 Pazar Tahminleme AI</h2>', unsafe_allow_html=True)
        
        VizEngine.karar_destek_notu_olustur(
            "AI Tahminleme Sistemi",
            "Exponential Smoothing algoritması ile gelecek pazar değerleri tahmin edilmektedir. Güven aralıkları olası sonuçları gösterir.",
            "Tahminler stratejik planlama için kullanılmalı, mutlak gerçekler olarak değerlendirilmemelidir.",
            "bilgi"
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            tahmin_donemleri = st.slider("Tahmin Dönemleri", 1, 5, 2)
            
            if st.button("🔮 Tahmin Oluştur", type="primary", use_container_width=True):
                with st.spinner("AI tahminleme yapılıyor..."):
                    tahmin_sonuclari = AnalyticsEngine.pazar_tahmini_uret(df, tahmin_donemleri)
                    
                    if tahmin_sonuclari:
                        st.session_state.tahmin_sonuclari = tahmin_sonuclari
                        st.success("✅ Tahmin oluşturuldu!")
                    else:
                        st.error("Tahmin oluşturulamadı. En az 3 yıl veri gereklidir.")
        
        with col2:
            if 'tahmin_sonuclari' in st.session_state and st.session_state.tahmin_sonuclari:
                fig = VizEngine.tahmin_grafigi_olustur(st.session_state.tahmin_sonuclari)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Tahmin detayları
                tahmin = st.session_state.tahmin_sonuclari
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric(
                        "Pazar Genişleme Oranı",
                        f"%{tahmin['pazar_genisleme_orani']:.1f}",
                        "Yıllık Ortalama"
                    )
                
                with col_b:
                    st.metric(
                        "2025 Tahmini",
                        f"${tahmin['tahmin_degerleri'][0]/1e6:.1f}M",
                        f"%{tahmin['buyume_oranlari'][0]:.1f} Büyüme"
                    )
                
                with col_c:
                    st.metric(
                        "Model Hatası",
                        f"±${tahmin['model_hatasi']/1e6:.2f}M",
                        "%95 Güven Aralığı"
                    )
                
                VizEngine.karar_destek_notu_olustur(
                    "Tahmin Değerlendirmesi",
                    f"Pazarın {tahmin_donemleri} yıllık ortalama büyüme beklentisi: %{tahmin['pazar_genisleme_orani']:.1f}",
                    "Büyüme beklentilerine göre yatırım planlarınızı gözden geçirin. Yüksek büyüme potansiyeli olan segmentlere odaklanın.",
                    "oneri"
                )
    
    @staticmethod
    def _risk_analizi_sekmesi(df: pd.DataFrame):
        """Risk analizi sekmesi."""
        st.markdown('<h2 class="bolum-baslik">⚠️ Risk ve Anomali Tespiti</h2>', unsafe_allow_html=True)
        
        VizEngine.karar_destek_notu_olustur(
            "AI Risk Analizi",
            "Isolation Forest algoritması ile pazardaki anormal davranışlar tespit edilmektedir. Kritik riskli ürünler acil müdahale gerektirir.",
            "Riskli ürünleri düzenli olarak izleyin ve erken uyarı sistemleri kurun.",
            "uyari"
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            risk_seviyesi = st.slider("Anomali Hassasiyeti", 0.05, 0.3, 0.1, 0.05)
            
            if st.button("🔍 Risk Analizi Yap", type="primary", use_container_width=True):
                with st.spinner("Risk analizi yapılıyor..."):
                    anomali_df = AnalyticsEngine.anomali_tespiti_uret(df, risk_seviyesi)
                    
                    if anomali_df is not None:
                        st.session_state.anomali_df = anomali_df
                        st.success(f"✅ {len(anomali_df[anomali_df['Risk_Seviyesi'].isin(['Kritik Risk', 'Yüksek Risk'])])} riskli ürün tespit edildi!")
                    else:
                        st.error("Risk analizi yapılamadı.")
        
        with col2:
            if 'anomali_df' in st.session_state and st.session_state.anomali_df is not None:
                fig = VizEngine.anomali_grafigi_olustur(st.session_state.anomali_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk detayları
                anomali_df = st.session_state.anomali_df
                
                # Risk dağılımı
                risk_dagilimi = anomali_df['Risk_Seviyesi'].value_counts()
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("Kritik Risk", risk_dagilimi.get('Kritik Risk', 0))
                
                with col_b:
                    st.metric("Yüksek Risk", risk_dagilimi.get('Yüksek Risk', 0))
                
                with col_c:
                    st.metric("Orta Risk", risk_dagilimi.get('Orta Risk', 0))
                
                with col_d:
                    st.metric("Toplam Ürün", len(anomali_df))
                
                # Kritik riskli ürünler
                kritik_urunler = anomali_df[anomali_df['Risk_Seviyesi'] == 'Kritik Risk']
                if len(kritik_urunler) > 0:
                    st.markdown("#### 🚨 Kritik Riskli Ürünler")
                    
                    gosterilecek_sutunlar = []
                    for sutun in ['Molekul', 'Sirket', 'Anomali_Tipi', 'Anomali_Skoru']:
                        if sutun in kritik_urunler.columns:
                            gosterilecek_sutunlar.append(sutun)
                    
                    satis_sutunlari = [sutun for sutun in kritik_urunler.columns if re.search(r'Satış_20\d{2}', sutun)]
                    if satis_sutunlari:
                        gosterilecek_sutunlar.append(satis_sutunlari[-1])
                    
                    st.dataframe(
                        kritik_urunler[gosterilecek_sutunlar].sort_values('Anomali_Skoru').head(10),
                        use_container_width=True
                    )
    
    @staticmethod
    def _segmentasyon_sekmesi(df: pd.DataFrame):
        """Segmentasyon sekmesi."""
        st.markdown('<h2 class="bolum-baslik">🎯 PCA Tabanlı Ürün Segmentasyonu</h2>', unsafe_allow_html=True)
        
        VizEngine.karar_destek_notu_olustur(
            "Akıllı Segmentasyon",
            "PCA (Temel Bileşen Analizi) ve K-Means algoritmaları ile ürünler stratejik segmentlere ayrılmaktadır.",
            "Her segment için farklı pazarlama ve yatırım stratejileri geliştirin.",
            "bilgi"
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            kume_sayisi = st.slider("Segment Sayısı", 3, 8, 4)
            
            if st.button("🎯 Segmentasyon Yap", type="primary", use_container_width=True):
                with st.spinner("PCA segmentasyonu yapılıyor..."):
                    segment_df = AnalyticsEngine.pca_segmentasyonu_uret(df, kume_sayisi)
                    
                    if segment_df is not None:
                        st.session_state.segment_df = segment_df
                        st.success("✅ Segmentasyon tamamlandı!")
                    else:
                        st.error("Segmentasyon yapılamadı.")
        
        with col2:
            if 'segment_df' in st.session_state and st.session_state.segment_df is not None:
                fig = VizEngine.pca_segmentasyon_grafigi(st.session_state.segment_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Segment analizi
                segment_df = st.session_state.segment_df
                
                if 'segment_ozellikleri' in segment_df.attrs:
                    segment_ozellikleri = segment_df.attrs['segment_ozellikleri']
                    
                    st.markdown("#### 📊 Segment Performansı")
                    
                    segment_analizi = []
                    for kume, ozellikler in segment_ozellikleri.items():
                        segment_analizi.append({
                            'Segment': segment_df[segment_df['Kume'] == kume]['Segment'].iloc[0] if len(segment_df[segment_df['Kume'] == kume]) > 0 else f"Kume_{kume}",
                            'Ürün Sayısı': len(segment_df[segment_df['Kume'] == kume]),
                            'Ort. Satış': f"${ozellikler.get('ortalama_satis', 0)/1e6:.2f}M",
                            'Ort. Büyüme': f"%{ozellikler.get('ortalama_buyume', 0):.1f}"
                        })
                    
                    if segment_analizi:
                        st.table(pd.DataFrame(segment_analizi))
                    
                    VizEngine.karar_destek_notu_olustur(
                        "Segment Stratejileri",
                        "Her segmentin farklı yönetim stratejileri gerektirir. Yıldız ürünlere yatırım yapın, Nakit İneklerinden kar elde edin.",
                        "Segmentlere özel pazarlama kanalları ve satış ekipleri oluşturun. Performansa göre kaynak dağılımını optimize edin.",
                        "oneri"
                    )
    
    @staticmethod
    def _gorsellestirme_sekmesi(df: pd.DataFrame):
        """Görselleştirme sekmesi."""
        st.markdown('<h2 class="bolum-baslik">📈 İleri Görselleştirme</h2>', unsafe_allow_html=True)
        
        # Sunburst grafiği
        st.markdown("#### 🌐 Pazar Hiyerarşisi")
        sunburst_fig = VizEngine.sunburst_grafigi_olustur(df)
        if sunburst_fig:
            st.plotly_chart(sunburst_fig, use_container_width=True)
        else:
            st.info("Sunburst grafiği için Sirket ve Molekül sütunları gereklidir.")
        
        # Radar grafiği
        st.markdown("#### 📊 Şirket Karşılaştırması")
        
        if 'Sirket' in df.columns:
            sirketler = df['Sirket'].value_counts().nlargest(10).index.tolist()
            secili_sirketler = st.multiselect(
                "Karşılaştırılacak şirketler",
                sirketler,
                default=sirketler[:min(3, len(sirketler))]
            )
            
            if secili_sirketler:
                radar_fig = VizEngine.radar_grafigi_olustur(df, secili_sirketler)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
        
        # Satış trendi
        st.markdown("#### 📈 Satış Trend Analizi")
        
        satis_sutunlari = [sutun for sutun in df.columns if re.search(r'Satış_20\d{2}', sutun)]
        if len(satis_sutunlari) >= 2:
            # Yıllık toplam satışlar
            yillik_satis = []
            for sutun in sorted(satis_sutunlari):
                yil = DataEngine.sutundan_yil_cikar(sutun)
                if yil:
                    yillik_satis.append({'Yil': yil, 'Satis': df[sutun].sum()})
            
            if yillik_satis:
                trend_df = pd.DataFrame(yillik_satis)
                
                fig = px.line(
                    trend_df,
                    x='Yil',
                    y='Satis',
                    markers=True,
                    title='Yıllık Satış Trendi'
                )
                
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _veri_analizi_sekmesi(df: pd.DataFrame):
        """Veri analizi sekmesi."""
        st.markdown('<h2 class="bolum-baslik">📋 Detaylı Veri Analizi</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 İstatistiksel Özet")
            
            # Sayısal sütunların istatistikleri
            numerik_df = df.select_dtypes(include=[np.number])
            if not numerik_df.empty:
                st.dataframe(numerik_df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("#### 🔍 Veri Kalitesi")
            
            veri_kalitesi = []
            
            # Eksik değerler
            eksik_degerler = df.isnull().sum()
            toplam_satir = len(df)
            
            for sutun in df.columns[:10]:  # İlk 10 sütun
                eksik_yuzde = (eksik_degerler[sutun] / toplam_satir * 100) if toplam_satir > 0 else 0
                veri_kalitesi.append({
                    'Sütun': sutun,
                    'Eksik %': f"{eksik_yuzde:.1f}%",
                    'Benzersiz': df[sutun].nunique()
                })
            
            if veri_kalitesi:
                st.table(pd.DataFrame(veri_kalitesi))
        
        # Korelasyon matrisi
        st.markdown("#### 📈 Korelasyon Analizi")
        
        numerik_df = df.select_dtypes(include=[np.number])
        if len(numerik_df.columns) >= 2:
            # En önemli 8 sütun
            onemli_sutunlar = []
            
            satis_sutunlari = [sutun for sutun in numerik_df.columns if re.search(r'Satış_20\d{2}', sutun)]
            if satis_sutunlari:
                onemli_sutunlar.append(satis_sutunlari[-1])
            
            buyume_sutunlari = [sutun for sutun in numerik_df.columns if 'Buyume_' in sutun]
            if buyume_sutunlari:
                onemli_sutunlar.append(buyume_sutunlari[-1])
            
            if 'Pazar_Payi' in numerik_df.columns:
                onemli_sutunlar.append('Pazar_Payi')
            
            fiyat_sutunlari = [sutun for sutun in numerik_df.columns if 'Ort_Fiyat_' in sutun]
            if fiyat_sutunlari:
                onemli_sutunlar.append(fiyat_sutunlari[-1])
            
            if len(onemli_sutunlar) >= 2:
                korelasyon_df = numerik_df[onemli_sutunlar[:8]].corr()
                
                fig = px.imshow(
                    korelasyon_df,
                    text_auto='.2f',
                    color_continuous_scale='RdBu',
                    title='Korelasyon Matrisi'
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _raporlama_sekmesi(df: pd.DataFrame, metrikler: Dict):
        """Raporlama sekmesi."""
        st.markdown('<h2 class="bolum-baslik">📑 Raporlama ve Dışa Aktarım</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Excel Raporu", use_container_width=True):
                PharmaUI._excel_raporu_uret(df, metrikler)
        
        with col2:
            if st.button("📈 CSV Verisi", use_container_width=True):
                PharmaUI._csv_verisi_uret(df)
        
        with col3:
            if st.button("🔄 Analizi Sıfırla", use_container_width=True):
                PharmaUI._analizi_sifirla()
        
        # Hızlı istatistikler
        st.markdown("#### 📊 Hızlı İstatistikler")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Toplam Satır", f"{len(df):,}")
        
        with stat_col2:
            st.metric("Toplam Sütun", len(df.columns))
        
        with stat_col3:
            hafiza = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Hafıza", f"{hafiza:.1f} MB")
        
        with stat_col4:
            st.metric("Benzersiz Moleküller", df['Molekul'].nunique() if 'Molekul' in df.columns else "N/A")
    
    @staticmethod
    def _excel_raporu_uret(df: pd.DataFrame, metrikler: Dict):
        """Excel raporu üret."""
        try:
            with st.spinner("Excel raporu oluşturuluyor..."):
                output = BytesIO()
                
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Özet sayfası
                    ozet_data = [
                        ['Toplam Pazar Değeri', f"${metrikler.get('Toplam_Pazar_Degeri', 0)/1e6:.2f}M"],
                        ['Ortalama Büyüme', f"%{metrikler.get('Ort_Buyume_Orani', 0):.1f}"],
                        ['HHI İndeksi', f"{metrikler.get('HHI_Indeksi', 0):.0f}"],
                        ['Toplam Ürün', metrikler.get('Toplam_Satir', 0)],
                        ['Oluşturulma Tarihi', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                    ]
                    
                    ozet_df = pd.DataFrame(ozet_data, columns=['Metrik', 'Değer'])
                    ozet_df.to_excel(writer, sheet_name='Yönetici Özeti', index=False)
                    
                    # Detaylı veri
                    df.to_excel(writer, sheet_name='Detaylı Veri', index=False)
                    
                    # En iyi ürünler
                    satis_sutunlari = [sutun for sutun in df.columns if re.search(r'Satış_20\d{2}', sutun)]
                    if satis_sutunlari:
                        if 'Molekul' in df.columns and 'Sirket' in df.columns:
                            en_iyi_df = df[['Molekul', 'Sirket', satis_sutunlari[-1]]].nlargest(50, satis_sutunlari[-1])
                        else:
                            en_iyi_df = df[[satis_sutunlari[-1]]].nlargest(50, satis_sutunlari[-1])
                        en_iyi_df.to_excel(writer, sheet_name='En İyi 50 Ürün', index=False)
                
                output.seek(0)
                
                zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                st.download_button(
                    label="⬇️ Excel Raporunu İndir",
                    data=output,
                    file_name=f"pharma_rapor_{zaman_damgasi}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                st.success("✅ Excel raporu hazır!")
                
        except Exception as e:
            st.error(f"Excel raporu oluşturma hatası: {str(e)}")
    
    @staticmethod
    def _csv_verisi_uret(df: pd.DataFrame):
        """CSV verisi üret."""
        csv = df.to_csv(index=False)
        zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        st.download_button(
            label="⬇️ CSV Verisini İndir",
            data=csv,
            file_name=f"pharma_veri_{zaman_damgasi}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    @staticmethod
    def _analizi_sifirla():
        """Analizi sıfırla."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ================================================
# 6. ANA UYGULAMA
# ================================================

def main():
    """Ana uygulama fonksiyonu."""
    
    # Sayfa yapılandırması
    st.set_page_config(
        page_title="PharmaIntelligence Pro | Kurumsal Karar Destek",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Tema CSS
    PROFESYONEL_CSS = """
    <style>
        :root {
            --birincil-koyu: #0c1a32;
            --ikincil-koyu: #14274e;
            --vurgu-altin: #d4af37;
            --vurgu-gumus: #c0c0c0;
            --vurgu-lacivert: #2d7dd2;
            --basarili: #2dd2a3;
            --uyari: #f2c94c;
            --tehlike: #eb5757;
            --bilgi: #2d7dd2;
        }
        
        .stApp {
            background: linear-gradient(135deg, var(--birincil-koyu), var(--ikincil-koyu));
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #f8fafc;
        }
        
        .cam-kart {
            background: rgba(30, 58, 95, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(212, 175, 55, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            padding: 1.5rem;
            transition: all 0.3s ease;
        }
        
        .cam-kart:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.7);
            border-color: var(--vurgu-altin);
        }
        
        .pharma-baslik {
            font-size: 3rem;
            background: linear-gradient(135deg, var(--vurgu-altin), var(--vurgu-gumus));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 900;
            margin-bottom: 0.5rem;
        }
        
        .pharma-alt-baslik {
            font-size: 1.2rem;
            color: #cbd5e1;
            margin-bottom: 2rem;
        }
        
        .bolum-baslik {
            font-size: 1.8rem;
            color: var(--vurgu-altin);
            font-weight: 800;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid var(--vurgu-altin);
        }
        
        .alt-bolum-baslik {
            font-size: 1.4rem;
            color: var(--vurgu-gumus);
            margin: 1.5rem 0 1rem 0;
        }
        
        .metrik-kart {
            text-align: center;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        
        .metrik-kart.birincil {
            background: linear-gradient(135deg, rgba(45, 125, 210, 0.3), rgba(42, 202, 234, 0.3));
            border-left: 5px solid var(--vurgu-lacivert);
        }
        
        .metrik-kart.pozitif {
            background: linear-gradient(135deg, rgba(45, 210, 163, 0.3), rgba(76, 201, 240, 0.3));
            border-left: 5px solid var(--basarili);
        }
        
        .metrik-kart.negatif {
            background: linear-gradient(135deg, rgba(235, 87, 87, 0.3), rgba(242, 201, 76, 0.3));
            border-left: 5px solid var(--tehlike);
        }
        
        .metrik-etiket {
            font-size: 0.9rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }
        
        .metrik-deger {
            font-size: 2.2rem;
            font-weight: 800;
            color: #f8fafc;
            margin: 0.5rem 0;
        }
        
        .metrik-detay {
            font-size: 0.8rem;
            color: #64748b;
        }
        
        .karar-destek-notu {
            margin: 1rem 0;
            padding: 1.5rem;
            border-radius: 12px;
        }
        
        .karar-destek-notu.bilgi {
            background: rgba(45, 125, 210, 0.2);
            border-left: 5px solid var(--bilgi);
        }
        
        .karar-destek-notu.oneri {
            background: rgba(45, 210, 163, 0.2);
            border-left: 5px solid var(--basarili);
        }
        
        .karar-destek-notu.uyari {
            background: rgba(242, 201, 76, 0.2);
            border-left: 5px solid var(--uyari);
        }
        
        .karar-destek-baslik {
            font-size: 1.2rem;
            color: #f8fafc;
            margin-bottom: 0.5rem;
        }
        
        .karar-destek-icerik {
            color: #cbd5e1;
            margin-bottom: 1rem;
            line-height: 1.6;
        }
        
        .karar-destek-oneri {
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 8px;
            border-left: 3px solid var(--vurgu-altin);
        }
        
        .hosgeldin-container {
            text-align: center;
            padding: 4rem 2rem;
        }
        
        .hosgeldin-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
    </style>
    """
    
    st.markdown(PROFESYONEL_CSS, unsafe_allow_html=True)
    
    # Oturum durumu başlat
    if 'veri' not in st.session_state:
        st.session_state.veri = None
    if 'filtrelenmis_veri' not in st.session_state:
        st.session_state.filtrelenmis_veri = None
    if 'metrikler' not in st.session_state:
        st.session_state.metrikler = None
    if 'icgoruler' not in st.session_state:
        st.session_state.icgoruler = []
    
    # UI yöneticisini başlat
    PharmaUI.baslik_goster()
    PharmaUI.yan_cubugu_olustur()
    PharmaUI.ana_icerik_goster()

# ================================================
# 7. UYGULAMA BAŞLATMA
# ================================================

if __name__ == "__main__":
    try:
        gc.enable()
        main()
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        
        if st.button("🔄 Uygulamayı Yeniden Başlat", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
