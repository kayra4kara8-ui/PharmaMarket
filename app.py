"""
PharmaIntelligence Pro - Kurumsal İlaç Pazarı Analitik Platformu
Versiyon: 6.0.0
Yazar: PharmaIntelligence Inc.
Lisans: Kurumsal

Yapay zeka destekli öngörüler, tahminleme, anomali tespiti ve kapsamlı raporlama
ile gelişmiş ilaç pazarı analitiği.
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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

# Yardımcı Araçlar
from datetime import datetime, timedelta
import json
from io import BytesIO
import time
import gc
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import re
import hashlib
from collections import defaultdict

# Excel/PDF Dışa Aktarım
import xlsxwriter
# Reportlab'i şartlı olarak içe aktar
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
# 1. YAPILANDIRMA & TEMA
# ================================================

st.set_page_config(
    page_title="PharmaIntelligence Pro | Kurumsal Analitik",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://pharmaintelligence.com/kurumsal-destek',
        'Report a bug': "https://pharmaintelligence.com/kurumsal-hata-bildirimi",
        'About': """
        ### PharmaIntelligence Kurumsal v6.0
        • Yapay Zeka Destekli Tahminleme
        • Anomali Tespiti
        • Uluslararası Ürün Analitiği
        • Gelişmiş Segmentasyon
        • Otomatik Raporlama
        • Makine Öğrenimi Entegrasyonu
        © 2024 PharmaIntelligence Inc. Tüm Hakları Saklıdır
        """
    }
)

# Profesyonel Tema CSS
PROFESYONEL_CSS = """
<style>
    /* === KÖK DEĞİŞKENLER === */
    :root {
        --birincil-koyu: #0c1a32;
        --ikincil-koyu: #14274e;
        --vurgu-mavi: #2d7dd2;
        --vurgu-mavi-acik: #4a9fe3;
        --vurgu-mavi-koyu: #1a5fa0;
        --vurgu-cyan: #2acaea;
        --vurgu-teal: #30c9c9;
        --basarili: #2dd2a3;
        --uyari: #f2c94c;
        --tehlike: #eb5757;
        --bilgi: #2d7dd2;
        
        --metin-birincil: #f8fafc;
        --metin-ikincil: #cbd5e1;
        --metin-soluk: #64748b;
        
        --arkaplan-birincil: #0c1a32;
        --arkaplan-ikincil: #14274e;
        --arkaplan-kart: #1e3a5f;
        --arkaplan-hover: #2d4a7a;
        --arkaplan-yuzey: #14274e;
        
        --golge-kucuk: 0 2px 8px rgba(0, 0, 0, 0.4);
        --golge-orta: 0 4px 16px rgba(0, 0, 0, 0.5);
        --golge-buyuk: 0 8px 32px rgba(0, 0, 0, 0.6);
        --golge-cok-buyuk: 0 12px 48px rgba(0, 0, 0, 0.7);
        
        --kenar-yuvarlakligi-kucuk: 8px;
        --kenar-yuvarlakligi-orta: 12px;
        --kenar-yuvarlakligi-buyuk: 16px;
        --kenar-yuvarlakligi-cok-buyuk: 20px;
        
        --gecis-hizli: 200ms cubic-bezier(0.4, 0, 0.2, 1);
        --gecis-normal: 300ms cubic-bezier(0.4, 0, 0.2, 1);
        --gecis-yavas: 500ms cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* === GENEL STİLLER === */
    .stApp {
        background: linear-gradient(135deg, var(--birincil-koyu), var(--ikincil-koyu));
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--metin-birincil);
        min-height: 100vh;
    }
    
    /* === GLASMORFİZM KARTLARI === */
    .cam-kart {
        background: rgba(30, 58, 95, 0.6);
        backdrop-filter: blur(10px);
        border-radius: var(--kenar-yuvarlakligi-buyuk);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: var(--golge-buyuk);
        padding: 1.5rem;
        transition: all var(--gecis-normal);
    }
    
    .cam-kart:hover {
        transform: translateY(-5px);
        box-shadow: var(--golge-cok-buyuk);
        border-color: var(--vurgu-mavi);
    }
    
    /* === TİPOGRAFİ === */
    .pharma-baslik {
        font-size: 3rem;
        background: linear-gradient(135deg, var(--vurgu-mavi), var(--vurgu-cyan), var(--vurgu-teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        animation: gradient-kaydirma 3s ease infinite;
    }
    
    @keyframes gradient-kaydirma {
        0%, 100% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(20deg); }
    }
    
    .pharma-alt-baslik {
        font-size: 1.1rem;
        color: var(--metin-ikincil);
        font-weight: 400;
        max-width: 900px;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    
    .bolum-baslik {
        font-size: 1.8rem;
        color: var(--metin-birincil);
        font-weight: 800;
        margin: 2.5rem 0 1.5rem 0;
        padding-left: 1rem;
        border-left: 5px solid var(--vurgu-mavi);
        background: linear-gradient(90deg, rgba(45, 125, 210, 0.15), transparent);
        padding: 1rem;
        border-radius: var(--kenar-yuvarlakligi-kucuk);
    }
    
    /* === METRİK KARTLARI === */
    .ozel-metrik-kart {
        background: var(--arkaplan-kart);
        padding: 1.5rem;
        border-radius: var(--kenar-yuvarlakligi-buyuk);
        box-shadow: var(--golge-orta);
        border: 1px solid var(--arkaplan-hover);
        transition: all var(--gecis-normal);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .ozel-metrik-kart::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--vurgu-mavi), var(--vurgu-cyan));
    }
    
    .ozel-metrik-kart:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: var(--golge-cok-buyuk);
        border-color: var(--vurgu-mavi);
    }
    
    .ozel-metrik-deger {
        font-size: 2.4rem;
        font-weight: 900;
        margin: 0.5rem 0;
        color: var(--metin-birincil);
        line-height: 1;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .ozel-metrik-etiket {
        font-size: 0.85rem;
        color: var(--metin-ikincil);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* === İÇGÖRÜ KARTLARI === */
    .icgoru-kart {
        background: var(--arkaplan-kart);
        padding: 1.2rem;
        border-radius: var(--kenar-yuvarlakligi-orta);
        box-shadow: var(--golge-kucuk);
        border-left: 5px solid;
        margin: 0.8rem 0;
        transition: all var(--gecis-hizli);
        position: relative;
        overflow: hidden;
    }
    
    .icgoru-kart::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.05), transparent);
        opacity: 0;
        transition: opacity var(--gecis-normal);
    }
    
    .icgoru-kart:hover::before {
        opacity: 1;
    }
    
    .icgoru-kart:hover {
        transform: translateY(-3px);
        box-shadow: var(--golge-orta);
    }
    
    .icgoru-kart.bilgi { border-left-color: var(--vurgu-mavi); }
    .icgoru-kart.basarili { border-left-color: var(--basarili); }
    .icgoru-kart.uyari { border-left-color: var(--uyari); }
    .icgoru-kart.tehlike { border-left-color: var(--tehlike); }
    
    /* === ANİMASYONLAR === */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* === STREAMLIT GEÇERSİZ KILMALARI === */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: var(--metin-birincil) !important;
    }
    
    .stDataFrame, .stTable {
        background: var(--arkaplan-kart) !important;
        border-radius: var(--kenar-yuvarlakligi-orta) !important;
        border: 1px solid var(--arkaplan-hover) !important;
    }
    
    /* === FİLTRE BÖLÜMÜ === */
    .filtre-durumu {
        background: linear-gradient(135deg, rgba(45, 125, 210, 0.2), rgba(42, 202, 234, 0.2));
        padding: 1rem;
        border-radius: var(--kenar-yuvarlakligi-orta);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--basarili);
        box-shadow: var(--golge-orta);
        color: var(--metin-birincil);
        font-size: 0.95rem;
    }
    
    /* === ROZETLER === */
    .rozet {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .rozet-basarili {
        background: rgba(45, 210, 163, 0.2);
        color: var(--basarili);
        border: 1px solid rgba(45, 210, 163, 0.3);
    }
    
    .rozet-uyari {
        background: rgba(242, 201, 76, 0.2);
        color: var(--uyari);
        border: 1px solid rgba(242, 201, 76, 0.3);
    }
    
    .rozet-bilgi {
        background: rgba(45, 125, 210, 0.2);
        color: var(--vurgu-mavi);
        border: 1px solid rgba(45, 125, 210, 0.3);
    }
</style>
"""

st.markdown(PROFESYONEL_CSS, unsafe_allow_html=True)

# ================================================
# 2. VERİ İŞLEME SINIFI
# ================================================

class VeriIsleyici:
    """
    Kurumsal seviye veri işleme motoru ile gelişmiş optimizasyon,
    temizleme ve dönüştürme yetenekleri.
    """
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
    def buyuk_veri_seti_yukle(
        dosya: Any,
        ornek_boyut: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Büyük veri setlerini akıllı bölümleme ve ilerleme takibi ile yükler.
        
        Args:
            dosya: Yüklenen dosya nesnesi
            ornek_boyut: Örnekleme için isteğe bağlı satır limiti
            
        Returns:
            Optimize edilmiş DataFrame veya hata durumunda None
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
                    parcalar = []
                    parca_boyutu = 50000
                    toplam_parca = (ornek_boyut // parca_boyutu) + 1
                    
                    with st.spinner(f"📥 Büyük veri seti yükleniyor..."):
                        ilerleme_cubugu = st.progress(0)
                        durum_metni = st.empty()
                        
                        for i in range(toplam_parca):
                            parca = pd.read_excel(
                                dosya,
                                skiprows=i * parca_boyutu,
                                nrows=parca_boyutu,
                                engine='openpyxl'
                            )
                            
                            if parca.empty:
                                break
                            
                            parcalar.append(parca)
                            
                            yuklenen_satirlar = sum(len(p) for p in parcalar)
                            ilerleme = min(yuklenen_satirlar / ornek_boyut, 1.0)
                            
                            ilerleme_cubugu.progress(ilerleme)
                            durum_metni.text(f"📊 {yuklenen_satirlar:,} satır yüklendi...")
                            
                            if yuklenen_satirlar >= ornek_boyut:
                                break
                        
                        df = pd.concat(parcalar, ignore_index=True)
                        ilerleme_cubugu.progress(1.0)
                        durum_metni.text(f"✅ {len(df):,} satır başarıyla yüklendi")
                        time.sleep(0.5)
                        ilerleme_cubugu.empty()
                        durum_metni.empty()
                else:
                    with st.spinner(f"📥 Tüm veri seti yükleniyor..."):
                        df = pd.read_excel(dosya, engine='openpyxl')
            else:
                st.error("❌ Desteklenmeyen dosya formatı")
                return None
            
            # DataFrame'i optimize et
            df = VeriIsleyici.dataframe_optimize_et(df)
            
            yukleme_suresi = time.time() - baslangic_zamani
            st.success(f"✅ Veri yüklendi: {len(df):,} satır, {len(df.columns)} sütun ({yukleme_suresi:.2f}s)")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.error(f"Detaylar: {traceback.format_exc()}")
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
            
            # Sütun isimlerini temizle
            df.columns = VeriIsleyici.sutun_isimleri_temizle(df.columns)
            
            with st.spinner("Veri seti optimize ediliyor..."):
                
                # Kategorik sütunları optimize et
                for sutun in df.select_dtypes(include=['object']).columns:
                    benzersiz_sayisi = df[sutun].nunique()
                    toplam_satir = len(df)
                    
                    # Kardinalite <%70 ise kategoriye dönüştür
                    if benzersiz_sayisi < toplam_satir * 0.7:
                        df[sutun] = df[sutun].astype('category')
                
                # Güvenli dönüşüm ile numerik sütunları optimize et
                for sutun in df.select_dtypes(include=[np.number]).columns:
                    try:
                        sutun_verisi = df[sutun]
                        
                        # Tümü NaN ise atla
                        if sutun_verisi.isna().all():
                            continue
                        
                        sutun_min = sutun_verisi.min()
                        sutun_max = sutun_verisi.max()
                        
                        # Min/max NaN ise atla
                        if pd.isna(sutun_min) or pd.isna(sutun_max):
                            continue
                        
                        # Tamsayı optimizasyonu
                        if pd.api.types.is_integer_dtype(df[sutun]):
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
                        else:
                            # Ondalık optimizasyonu
                            df[sutun] = df[sutun].astype(np.float32)
                    except Exception:
                        continue
                
                # String temizleme
                for sutun in df.select_dtypes(include=['object', 'category']).columns:
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
    def sutun_isimleri_temizle(sutunlar: List[str]) -> List[str]:
        """
        Yinelenenleri kaldırarak sütun isimlerini temizle ve standardize et.
        
        Args:
            sutunlar: Sütun isimleri listesi
            
        Returns:
            Temizlenmiş, benzersiz sütun isimleri listesi
        """
        temizlenmis = []
        gorulen_isimler = {}
        
        for sutun in sutunlar:
            if isinstance(sutun, str):
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
                sutun = VeriIsleyici._sutun_haritalamalari_uygula(sutun)
                
                sutun = sutun.strip()
            
            # Yinelenenleri kaldırma
            orijinal_sutun = str(sutun)
            if orijinal_sutun in gorulen_isimler:
                gorulen_isimler[orijinal_sutun] += 1
                sutun = f"{orijinal_sutun}_{gorulen_isimler[orijinal_sutun]}"
            else:
                gorulen_isimler[orijinal_sutun] = 0
            
            temizlenmis.append(str(sutun).strip())
        
        return temizlenmis
    
    @staticmethod
    def _sutun_haritalamalari_uygula(sutun: str) -> str:
        """Alan-spesifik sütun ismi haritalamaları uygula."""
        
        # Satış sütunları
        if "MAT Q3 2022 USD MNF" in sutun:
            return "Satış_2022"
        elif "MAT Q3 2023 USD MNF" in sutun:
            return "Satış_2023"
        elif "MAT Q3 2024 USD MNF" in sutun:
            return "Satış_2024"
        
        # Birim sütunları
        elif "MAT Q3 2022 Units" in sutun:
            return "Birim_2022"
        elif "MAT Q3 2023 Units" in sutun:
            return "Birim_2023"
        elif "MAT Q3 2024 Units" in sutun:
            return "Birim_2024"
        
        # Fiyat sütunları
        elif "MAT Q3 2022 Unit Avg Price USD MNF" in sutun:
            return "Ort_Fiyat_2022"
        elif "MAT Q3 2023 Unit Avg Price USD MNF" in sutun:
            return "Ort_Fiyat_2023"
        elif "MAT Q3 2024 Unit Avg Price USD MNF" in sutun:
            return "Ort_Fiyat_2024"
        
        # Standart Birimler
        elif "MAT Q3 2022 Standard Units" in sutun:
            return "Standart_Birim_2022"
        elif "MAT Q3 2023 Standard Units" in sutun:
            return "Standart_Birim_2023"
        elif "MAT Q3 2024 Standard Units" in sutun:
            return "Standart_Birim_2024"
        
        # SB Ortalama Fiyat
        elif "MAT Q3 2022 SU Avg Price USD MNF" in sutun:
            return "SB_Ort_Fiyat_2022"
        elif "MAT Q3 2023 SU Avg Price USD MNF" in sutun:
            return "SB_Ort_Fiyat_2023"
        elif "MAT Q3 2024 SU Avg Price USD MNF" in sutun:
            return "SB_Ort_Fiyat_2024"
        
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
            return "Alt_Bolge"
        
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
        eslesme = re.search(r'\b(20\d{2})\b', sutun_adi)
        if eslesme:
            return int(eslesme.group(1))
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
            # Regex kullanarak satış sütunlarını bul
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            
            if not satis_sutunlari:
                st.warning("⚠️ Satış sütunu bulunamadı")
                return df
            
            # Güvenli bir şekilde yılları çıkar
            yillar = []
            for sutun in satis_sutunlari:
                yil = VeriIsleyici.sutundan_yil_cikar(sutun)
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
                    df[buyume_sutun] = np.where(
                        df[onceki_sutun] != 0,
                        ((df[mevcut_sutun] - df[onceki_sutun]) / df[onceki_sutun]) * 100,
                        np.nan
                    )
            
            # BSBH (CAGR) hesaplaması
            if len(yillar) >= 2:
                ilk_yil = yillar[0]
                son_yil = yillar[-1]
                ilk_sutun = f"Satış_{ilk_yil}"
                son_sutun = f"Satış_{son_yil}"
                
                if ilk_sutun in df.columns and son_sutun in df.columns:
                    donem_sayisi = son_yil - ilk_yil
                    df['BSBH'] = np.where(
                        df[ilk_sutun] > 0,
                        (np.power(df[son_sutun] / df[ilk_sutun], 1/donem_sayisi) - 1) * 100,
                        np.nan
                    )
            
            # Pazar payı
            if yillar:
                son_yil = yillar[-1]
                son_satis_sutun = f"Satış_{son_yil}"
                
                if son_satis_sutun in df.columns:
                    toplam_satis = df[son_satis_sutun].sum()
                    if toplam_satis > 0:
                        df['Pazar_Payi'] = (df[son_satis_sutun] / toplam_satis) * 100
            
            # Ortalama fiyatları hesapla (mevcut değilse)
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            if not fiyat_sutunlari:
                for yil in yillar:
                    satis_sutun = f"Satış_{yil}"
                    birim_sutun = f"Birim_{yil}"
                    
                    if satis_sutun in df.columns and birim_sutun in df.columns:
                        df[f'Ort_Fiyat_{yil}'] = np.where(
                            df[birim_sutun] > 0,
                            df[satis_sutun] / df[birim_sutun],
                            np.nan
                        )
            
            # Fiyat-Hacim oranı
            if yillar:
                son_yil = yillar[-1]
                fiyat_sutun = f"Ort_Fiyat_{son_yil}"
                birim_sutun = f"Birim_{son_yil}"
                
                if fiyat_sutun in df.columns and birim_sutun in df.columns:
                    df['Fiyat_Hacim_Orani'] = df[fiyat_sutun] * df[birim_sutun]
            
            # Performans skoru
            numerik_sutunlar = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numerik_sutunlar) >= 3:
                try:
                    olceklendirici = StandardScaler()
                    numerik_veri = df[numerik_sutunlar].fillna(0)
                    olceklenmis_veri = olceklendirici.fit_transform(numerik_veri)
                    df['Performans_Skoru'] = olceklenmis_veri.mean(axis=1)
                except Exception:
                    pass
            
            # Uluslararası Ürün işleme
            if 'Uluslararasi_Urun' in df.columns:
                df['Uluslararasi_Urun'] = df['Uluslararasi_Urun'].fillna(0).astype(int)
            
            return df
            
        except Exception as e:
            st.warning(f"Analitik hazırlama uyarısı: {str(e)}")
            return df

# ================================================
# 3. GELİŞMİŞ FİLTRE SİSTEMİ
# ================================================

class GelismisFiltreSistemi:
    """
    Akıllı arama yetenekleri ile çok kriterli destek sunan
    gelişmiş filtreleme sistemi.
    """
    
    @staticmethod
    def filtre_yan_cubugu_olustur(df: pd.DataFrame) -> Tuple[str, Dict, bool, bool]:
        """
        Gelişmiş filtreleme yan çubuğu oluştur.
        
        Args:
            df: Girdi DataFrame
            
        Returns:
            (arama_terimi, filtre_yapisi, filtreleri_uygula, filtreleri_temizle) tuple'ı
        """
        with st.sidebar.expander("🎯 GELİŞMİŞ FİLTRELEME", expanded=True):
            st.markdown('<div class="filter-title">🔍 Arama & Filtre</div>', unsafe_allow_html=True)
            
            arama_terimi = st.text_input(
                "🔎 Global Arama",
                placeholder="Molekül, Şirket, Ülke...",
                help="Tüm sütunlarda ara",
                key="global_arama"
            )
            
            filtre_yapisi = {}
            mevcut_sutunlar = df.columns.tolist()
            
            # Ülke filtresi
            if 'Ulke' in mevcut_sutunlar:
                ulkeler = sorted(df['Ulke'].dropna().unique())
                secili_ulkeler = GelismisFiltreSistemi._aranabilir_coklu_secim(
                    "🌍 Ülkeler",
                    ulkeler,
                    key="ulkeler_filtre",
                    tumunu_sec_varsayilan=True
                )
                if secili_ulkeler and "Hepsi" not in secili_ulkeler:
                    filtre_yapisi['Ulke'] = secili_ulkeler
            
            # Şirket filtresi
            if 'Sirket' in mevcut_sutunlar:
                sirketler = sorted(df['Sirket'].dropna().unique())
                secili_sirketler = GelismisFiltreSistemi._aranabilir_coklu_secim(
                    "🏢 Şirketler",
                    sirketler,
                    key="sirketler_filtre",
                    tumunu_sec_varsayilan=True
                )
                if secili_sirketler and "Hepsi" not in secili_sirketler:
                    filtre_yapisi['Sirket'] = secili_sirketler
            
            # Molekül filtresi
            if 'Molekul' in mevcut_sutunlar:
                molekuller = sorted(df['Molekul'].dropna().unique())
                secili_molekuller = GelismisFiltreSistemi._aranabilir_coklu_secim(
                    "🧪 Moleküller",
                    molekuller,
                    key="molekuller_filtre",
                    tumunu_sec_varsayilan=True
                )
                if secili_molekuller and "Hepsi" not in secili_molekuller:
                    filtre_yapisi['Molekul'] = secili_molekuller
            
            st.markdown("---")
            st.markdown('<div class="filter-title">📊 Sayısal Filtreler</div>', unsafe_allow_html=True)
            
            # Satış filtresi
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if satis_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                min_satis = float(df[son_satis_sutun].min())
                max_satis = float(df[son_satis_sutun].max())
                
                satis_araligi = st.slider(
                    f"Satış Filtresi ({son_satis_sutun})",
                    min_value=min_satis,
                    max_value=max_satis,
                    value=(min_satis, max_satis),
                    key="satis_filtre"
                )
                filtre_yapisi['satis_araligi'] = (satis_araligi, son_satis_sutun)
            
            # Büyüme filtresi
            buyume_sutunlari = [sutun for sutun in df.columns if 'Buyume_' in sutun]
            if buyume_sutunlari:
                son_buyume_sutun = buyume_sutunlari[-1]
                min_buyume = float(df[son_buyume_sutun].min())
                max_buyume = float(df[son_buyume_sutun].max())
                
                buyume_araligi = st.slider(
                    f"Büyüme Filtresi ({son_buyume_sutun})",
                    min_value=min_buyume,
                    max_value=max_buyume,
                    value=(min(min_buyume, -50.0), max(max_buyume, 150.0)),
                    key="buyume_filtre"
                )
                filtre_yapisi['buyume_araligi'] = (buyume_araligi, son_buyume_sutun)
            
            st.markdown("---")
            st.markdown('<div class="filter-title">⚙️ Ek Filtreler</div>', unsafe_allow_html=True)
            
            # Uluslararası Ürün filtresi
            if 'Uluslararasi_Urun' in df.columns:
                uluslararasi_filtre = st.selectbox(
                    "Uluslararası Ürün",
                    ["Hepsi", "Sadece Uluslararası", "Sadece Yerel"],
                    key="uluslararasi_filtre"
                )
                if uluslararasi_filtre != "Hepsi":
                    filtre_yapisi['uluslararasi_filtre'] = uluslararasi_filtre
            
            # Sadece pozitif büyüme
            sadece_pozitif = st.checkbox("📈 Sadece Pozitif Büyüme", value=False)
            if sadece_pozitif and buyume_sutunlari:
                filtre_yapisi['pozitif_buyume'] = True
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                filtreleri_uygula = st.button("✅ Uygula", use_container_width=True, key="filtreleri_uygula")
            with col2:
                filtreleri_temizle = st.button("🗑️ Temizle", use_container_width=True, key="filtreleri_temizle")
            
            return arama_terimi, filtre_yapisi, filtreleri_uygula, filtreleri_temizle
    
    @staticmethod
    def _aranabilir_coklu_secim(
        etiket: str,
        secenekler: List[str],
        key: str,
        tumunu_sec_varsayilan: bool = False
    ) -> List[str]:
        """Aranabilir çoklu seçim widget'ı oluştur."""
        
        if not secenekler:
            return []
        
        tum_secenekler = ["Hepsi"] + list(secenekler)
        
        arama_sorgusu = st.text_input(
            f"{etiket} Ara",
            key=f"{key}_arama",
            placeholder="Ara..."
        )
        
        if arama_sorgusu:
            filtrelenmis_secenekler = ["Hepsi"] + [
                secenek for secenek in secenekler
                if arama_sorgusu.lower() in str(secenek).lower()
            ]
        else:
            filtrelenmis_secenekler = tum_secenekler
        
        varsayilan_secim = ["Hepsi"] if tumunu_sec_varsayilan else filtrelenmis_secenekler[:min(5, len(filtrelenmis_secenekler))]
        
        secili = st.multiselect(
            etiket,
            options=filtrelenmis_secenekler,
            default=varsayilan_secim,
            key=key,
            help="'Hepsi' tümünü seçer"
        )
        
        if "Hepsi" in secili and len(secili) > 1:
            secili = [secenek for secenek in secili if secenek != "Hepsi"]
        elif "Hepsi" in secili:
            secili = list(secenekler)
        
        if secili:
            if len(secili) == len(secenekler):
                st.caption(f"✅ TÜMÜ seçildi ({len(secenekler)} öğe)")
            else:
                st.caption(f"✅ {len(secili)} / {len(secenekler)} seçildi")
        
        return secili
    
    @staticmethod
    def filtreleri_uygula(
        df: pd.DataFrame,
        arama_terimi: str,
        filtre_yapisi: Dict
    ) -> pd.DataFrame:
        """
        Tüm yapılandırılmış filtreleri DataFrame'e uygula.
        
        Args:
            df: Girdi DataFrame
            arama_terimi: Global arama terimi
            filtre_yapisi: Filtre yapılandırmaları sözlüğü
            
        Returns:
            Filtrelenmiş DataFrame
        """
        filtrelenmis_df = df.copy()
        
        # Global arama
        if arama_terimi:
            arama_maskesi = pd.Series(False, index=filtrelenmis_df.index)
            for sutun in filtrelenmis_df.columns:
                try:
                    arama_maskesi = arama_maskesi | filtrelenmis_df[sutun].astype(str).str.contains(
                        arama_terimi, case=False, na=False
                    )
                except:
                    continue
            filtrelenmis_df = filtrelenmis_df[arama_maskesi]
        
        # Sütun filtreleri
        for sutun, degerler in filtre_yapisi.items():
            if sutun in filtrelenmis_df.columns and degerler and sutun not in [
                'satis_araligi', 'buyume_araligi', 'pozitif_buyume', 'uluslararasi_filtre'
            ]:
                filtrelenmis_df = filtrelenmis_df[filtrelenmis_df[sutun].isin(degerler)]
        
        # Satış aralığı
        if 'satis_araligi' in filtre_yapisi:
            (min_deger, max_deger), sutun_adi = filtre_yapisi['satis_araligi']
            if sutun_adi in filtrelenmis_df.columns:
                filtrelenmis_df = filtrelenmis_df[
                    (filtrelenmis_df[sutun_adi] >= min_deger) &
                    (filtrelenmis_df[sutun_adi] <= max_deger)
                ]
        
        # Büyüme aralığı
        if 'buyume_araligi' in filtre_yapisi:
            (min_deger, max_deger), sutun_adi = filtre_yapisi['buyume_araligi']
            if sutun_adi in filtrelenmis_df.columns:
                filtrelenmis_df = filtrelenmis_df[
                    (filtrelenmis_df[sutun_adi] >= min_deger) &
                    (filtrelenmis_df[sutun_adi] <= max_deger)
                ]
        
        # Uluslararası filtre
        if 'uluslararasi_filtre' in filtre_yapisi and 'Uluslararasi_Urun' in filtrelenmis_df.columns:
            if filtre_yapisi['uluslararasi_filtre'] == "Sadece Uluslararası":
                filtrelenmis_df = filtrelenmis_df[filtrelenmis_df['Uluslararasi_Urun'] == 1]
            elif filtre_yapisi['uluslararasi_filtre'] == "Sadece Yerel":
                filtrelenmis_df = filtrelenmis_df[filtrelenmis_df['Uluslararasi_Urun'] == 0]
        
        # Pozitif büyüme
        if 'pozitif_buyume' in filtre_yapisi and filtre_yapisi['pozitif_buyume']:
            buyume_sutunlari = [sutun for sutun in filtrelenmis_df.columns if 'Buyume_' in sutun]
            if buyume_sutunlari:
                filtrelenmis_df = filtrelenmis_df[filtrelenmis_df[buyume_sutunlari[-1]] > 0]
        
        return filtrelenmis_df

# ================================================
# 4. ANALİTİK MOTORU
# ================================================

class AnalitikMotoru:
    """
    Kapsamlı pazar istihbaratı, tahminleme ve anomali tespiti
    yetenekleri ile gelişmiş analitik motoru.
    """
    
    @staticmethod
    def kapsamli_metrikleri_hesapla(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Kapsamlı pazar metriklerini hesapla.
        
        Args:
            df: Girdi DataFrame
            
        Returns:
            Hesaplanmış metrikler sözlüğü
        """
        metrikler = {}
        
        try:
            metrikler['Toplam_Satir'] = len(df)
            metrikler['Toplam_Sutun'] = len(df.columns)
            
            # Satış metrikleri
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if satis_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                yil = VeriIsleyici.sutundan_yil_cikar(son_satis_sutun)
                
                metrikler['Son_Satis_Yili'] = yil
                metrikler['Toplam_Pazar_Degeri'] = df[son_satis_sutun].sum()
                metrikler['Urun_Basi_Ort_Satis'] = df[son_satis_sutun].mean()
                metrikler['Medyan_Satis'] = df[son_satis_sutun].median()
                metrikler['Satis_Std'] = df[son_satis_sutun].std()
                metrikler['Satis_Q1'] = df[son_satis_sutun].quantile(0.25)
                metrikler['Satis_Q3'] = df[son_satis_sutun].quantile(0.75)
                metrikler['Satis_CBA'] = metrikler['Satis_Q3'] - metrikler['Satis_Q1']
            
            # Büyüme metrikleri
            buyume_sutunlari = [sutun for sutun in df.columns if 'Buyume_' in sutun]
            if buyume_sutunlari:
                son_buyume_sutun = buyume_sutunlari[-1]
                metrikler['Ort_Buyume_Orani'] = df[son_buyume_sutun].mean()
                metrikler['Medyan_Buyume'] = df[son_buyume_sutun].median()
                metrikler['Pozitif_Buyume_Urunleri'] = (df[son_buyume_sutun] > 0).sum()
                metrikler['Negatif_Buyume_Urunleri'] = (df[son_buyume_sutun] < 0).sum()
                metrikler['Yuksek_Buyume_Urunleri'] = (df[son_buyume_sutun] > 20).sum()
                metrikler['Yuksek_Buyume_Yuzdesi'] = (metrikler['Yuksek_Buyume_Urunleri'] / metrikler['Toplam_Satir']) * 100
            
            # Şirket bazlı metrikler
            if 'Sirket' in df.columns and satis_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                sirket_satisleri = df.groupby('Sirket')[son_satis_sutun].sum().sort_values(ascending=False)
                toplam_satis = sirket_satisleri.sum()
                
                if toplam_satis > 0:
                    pazar_paylari = (sirket_satisleri / toplam_satis * 100)
                    metrikler['HHI_Indeksi'] = (pazar_paylari ** 2).sum()
                    
                    for n in [1, 3, 5, 10]:
                        if len(sirket_satisleri) >= n:
                            metrikler[f'Ilk_{n}_Pay'] = sirket_satisleri.nlargest(n).sum() / toplam_satis * 100
            
            # Molekül metrikleri
            if 'Molekul' in df.columns:
                metrikler['Benzersiz_Molekuller'] = df['Molekul'].nunique()
                if satis_sutunlari:
                    molekul_satisleri = df.groupby('Molekul')[son_satis_sutun].sum()
                    toplam_molekul_satis = molekul_satisleri.sum()
                    if toplam_molekul_satis > 0:
                        metrikler['Ilk_10_Molekul_Payi'] = molekul_satisleri.nlargest(10).sum() / toplam_molekul_satis * 100
            
            # Ülke metrikleri
            if 'Ulke' in df.columns:
                metrikler['Ulke_Kapsami'] = df['Ulke'].nunique()
                if satis_sutunlari:
                    ulke_satisleri = df.groupby('Ulke')[son_satis_sutun].sum()
                    toplam_ulke_satis = ulke_satisleri.sum()
                    if toplam_ulke_satis > 0:
                        metrikler['Ilk_5_Ulke_Payi'] = ulke_satisleri.nlargest(5).sum() / toplam_ulke_satis * 100
            
            # Fiyat metrikleri
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            if fiyat_sutunlari:
                son_fiyat_sutun = fiyat_sutunlari[-1]
                metrikler['Ort_Fiyat'] = df[son_fiyat_sutun].mean()
                metrikler['Fiyat_Varyansi'] = df[son_fiyat_sutun].var()
                metrikler['Fiyat_Q1'] = df[son_fiyat_sutun].quantile(0.25)
                metrikler['Fiyat_Medyan'] = df[son_fiyat_sutun].quantile(0.5)
                metrikler['Fiyat_Q3'] = df[son_fiyat_sutun].quantile(0.75)
            
            # Uluslararası Ürün metrikleri
            if 'Uluslararasi_Urun' in df.columns and satis_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                
                uluslararasi_df = df[df['Uluslararasi_Urun'] == 1]
                yerel_df = df[df['Uluslararasi_Urun'] == 0]
                
                metrikler['Uluslararasi_Urun_Sayisi'] = len(uluslararasi_df)
                metrikler['Yerel_Urun_Sayisi'] = len(yerel_df)
                metrikler['Uluslararasi_Urun_Satis'] = uluslararasi_df[son_satis_sutun].sum()
                metrikler['Yerel_Urun_Satis'] = yerel_df[son_satis_sutun].sum()
                
                toplam_satis = metrikler.get('Toplam_Pazar_Degeri', 0)
                if toplam_satis > 0:
                    metrikler['Uluslararasi_Urun_Payi'] = (metrikler['Uluslararasi_Urun_Satis'] / toplam_satis) * 100
                    metrikler['Yerel_Urun_Payi'] = (metrikler['Yerel_Urun_Satis'] / toplam_satis) * 100
                
                if len(uluslararasi_df) > 0 and buyume_sutunlari:
                    son_buyume_sutun = buyume_sutunlari[-1]
                    metrikler['Uluslararasi_Ort_Buyume'] = uluslararasi_df[son_buyume_sutun].mean()
                    metrikler['Yerel_Ort_Buyume'] = yerel_df[son_buyume_sutun].mean()
                
                if len(uluslararasi_df) > 0 and fiyat_sutunlari:
                    son_fiyat_sutun = fiyat_sutunlari[-1]
                    metrikler['Uluslararasi_Ort_Fiyat'] = uluslararasi_df[son_fiyat_sutun].mean()
                    metrikler['Yerel_Ort_Fiyat'] = yerel_df[son_fiyat_sutun].mean()
            
            return metrikler
            
        except Exception as e:
            st.warning(f"Metrik hesaplama uyarısı: {str(e)}")
            return metrikler
    
    @staticmethod
    def uluslararasi_urun_analizi(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Detaylı Uluslararası Ürün analizi.
        
        Args:
            df: Girdi DataFrame
            
        Returns:
            Analiz DataFrame veya None
        """
        try:
            if 'Uluslararasi_Urun' not in df.columns:
                return None
            
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if not satis_sutunlari:
                return None
            
            son_satis_sutun = satis_sutunlari[-1]
            analiz_verisi = []
            
            # Molekül bazlı analiz
            if 'Molekul' in df.columns:
                for molekul in df['Molekul'].unique():
                    molekul_df = df[df['Molekul'] == molekul]
                    
                    uluslararasi_mi = (molekul_df['Uluslararasi_Urun'] == 1).any()
                    toplam_satis = molekul_df[son_satis_sutun].sum()
                    
                    sirket_sayisi = molekul_df['Sirket'].nunique() if 'Sirket' in molekul_df.columns else 1
                    ulke_sayisi = molekul_df['Ulke'].nunique() if 'Ulke' in molekul_df.columns else 1
                    
                    buyume_sutunlari = [sutun for sutun in df.columns if 'Buyume_' in sutun]
                    ort_buyume = molekul_df[buyume_sutunlari[-1]].mean() if buyume_sutunlari else None
                    
                    fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
                    ort_fiyat = molekul_df[fiyat_sutunlari[-1]].mean() if fiyat_sutunlari else None
                    
                    analiz_verisi.append({
                        'Molekul': molekul,
                        'Uluslararasi': uluslararasi_mi,
                        'Toplam_Satis': toplam_satis,
                        'Sirket_Sayisi': sirket_sayisi,
                        'Ulke_Sayisi': ulke_sayisi,
                        'Urun_Sayisi': len(molekul_df),
                        'Ort_Fiyat': ort_fiyat,
                        'Ort_Buyume': ort_buyume,
                        'Karmaşıklık_Skoru': (sirket_sayisi * 0.6 + ulke_sayisi * 0.4) / 2
                    })
            
            elif 'Sirket' in df.columns:
                # Şirket bazlı analiz
                for sirket in df['Sirket'].unique():
                    sirket_df = df[df['Sirket'] == sirket]
                    
                    uluslararasi_mi = (sirket_df['Uluslararasi_Urun'] == 1).any()
                    toplam_satis = sirket_df[son_satis_sutun].sum()
                    
                    analiz_verisi.append({
                        'Sirket': sirket,
                        'Uluslararasi': uluslararasi_mi,
                        'Toplam_Satis': toplam_satis,
                        'Urun_Sayisi': len(sirket_df),
                        'Uluslararasi_Urun_Sayisi': (sirket_df['Uluslararasi_Urun'] == 1).sum()
                    })
            
            if analiz_verisi:
                analiz_df = pd.DataFrame(analiz_verisi)
                
                # Segmentasyon
                if 'Karmaşıklık_Skoru' in analiz_df.columns:
                    analiz_df['Segment'] = pd.cut(
                        analiz_df['Karmaşıklık_Skoru'],
                        bins=[0, 0.5, 1.5, 3, float('inf')],
                        labels=['Yerel', 'Bölgesel', 'Çok-Uluslu', 'Küresel']
                    )
                
                return analiz_df.sort_values('Toplam_Satis', ascending=False)
            
            return None
            
        except Exception as e:
            st.warning(f"Uluslararası Ürün analiz hatası: {str(e)}")
            return None
    
    @staticmethod
    def anomali_tespiti(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Isolation Forest kullanarak pazar anomali tespiti.
        
        Args:
            df: Girdi DataFrame
            
        Returns:
            Anomali skorları ile DataFrame
        """
        try:
            # Numerik özellikleri seç
            numerik_sutunlar = []
            
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if satis_sutunlari:
                numerik_sutunlar.extend(satis_sutunlari[-2:])
            
            buyume_sutunlari = [sutun for sutun in df.columns if 'Buyume_' in sutun]
            if buyume_sutunlari:
                numerik_sutunlar.append(buyume_sutunlari[-1])
            
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            if fiyat_sutunlari:
                numerik_sutunlar.append(fiyat_sutunlari[-1])
            
            if len(numerik_sutunlar) < 2:
                return None
            
            anomali_verisi = df[numerik_sutunlar].fillna(0)
            
            if len(anomali_verisi) < 10:
                return None
            
            # Isolation Forest
            izolasyon_ormani = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            anomali_skorlari = izolasyon_ormani.fit_predict(anomali_verisi)
            anomali_skorlari_surekli = izolasyon_ormani.score_samples(anomali_verisi)
            
            sonuc_df = df.copy()
            sonuc_df['Anomali'] = anomali_skorlari
            sonuc_df['Anomali_Skoru'] = anomali_skorlari_surekli
            
            # Kategorilere ayır
            sonuc_df['Anomali_Kategorisi'] = pd.cut(
                sonuc_df['Anomali_Skoru'],
                bins=[-np.inf, -0.5, -0.2, 0],
                labels=['Yüksek Risk', 'Orta Risk', 'Normal']
            )
            
            return sonuc_df
            
        except Exception as e:
            st.warning(f"Anomali tespiti hatası: {str(e)}")
            return None
    
    @staticmethod
    def pazar_tahmini(df: pd.DataFrame, donemler: int = 2) -> Optional[pd.DataFrame]:
        """
        Zaman serisi analizi kullanarak gelecek pazar değerlerini tahmin et.
        
        Args:
            df: Girdi DataFrame
            donemler: Tahmin edilecek dönem sayısı
            
        Returns:
            Tahmin DataFrame veya None
        """
        try:
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if len(satis_sutunlari) < 3:
                return None
            
            # Yılları çıkar
            yillar = []
            for sutun in sorted(satis_sutunlari):
                yil = VeriIsleyici.sutundan_yil_cikar(sutun)
                if yil:
                    yillar.append(yil)
            
            yillar = sorted(set(yillar))
            
            if len(yillar) < 3:
                return None
            
            # Yıllara göre satışları topla
            yillik_satis = {}
            for yil in yillar:
                sutun = f"Satış_{yil}"
                if sutun in df.columns:
                    yillik_satis[yil] = df[sutun].sum()
            
            # Zaman serisi oluştur
            zaman_serisi = pd.Series(yillik_satis)
            
            # Exponential Smoothing tahmini
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
            
            # Güven aralıkları (basitleştirilmiş)
            artiklar_std = np.std(uydurulmus_model.fittedvalues - zaman_serisi)
            guven_araligi = 1.96 * artiklar_std
            
            tahmin_df = pd.DataFrame({
                'Yil': tahmin_yillari,
                'Tahmin': tahmin_degerleri.values,
                'Alt_Sinir': tahmin_degerleri.values - guven_araligi,
                'Ust_Sinir': tahmin_degerleri.values + guven_araligi
            })
            
            # Büyüme oranlarını hesapla
            if len(tahmin_df) > 0:
                son_gercek = zaman_serisi.iloc[-1]
                ilk_tahmin = tahmin_df['Tahmin'].iloc[0]
                tahmin_df['Son_Yila_Gore_Buyume'] = ((ilk_tahmin - son_gercek) / son_gercek) * 100
            
            return tahmin_df
            
        except Exception as e:
            st.warning(f"Tahminleme hatası: {str(e)}")
            return None
    
    @staticmethod
    def gelismis_segmentasyon(
        df: pd.DataFrame,
        kume_sayisi: int = 4,
        yontem: str = 'kmeans'
    ) -> Optional[pd.DataFrame]:
        """
        Çoklu algoritmalar ile gelişmiş pazar segmentasyonu.
        
        Args:
            df: Girdi DataFrame
            kume_sayisi: Küme sayısı
            yontem: Kümeleme yöntemi ('kmeans', 'dbscan', 'hierarchical')
            
        Returns:
            Segmentlenmiş DataFrame veya None
        """
        try:
            # Özellikleri seç
            ozellikler = []
            
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if satis_sutunlari:
                ozellikler.extend(satis_sutunlari[-2:])
            
            buyume_sutunlari = [sutun for sutun in df.columns if 'Buyume_' in sutun]
            if buyume_sutunlari:
                ozellikler.append(buyume_sutunlari[-1])
            
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            if fiyat_sutunlari:
                ozellikler.append(fiyat_sutunlari[-1])
            
            if 'Pazar_Payi' in df.columns:
                ozellikler.append('Pazar_Payi')
            
            if len(ozellikler) < 2:
                return None
            
            segmentasyon_verisi = df[ozellikler].fillna(0)
            
            if len(segmentasyon_verisi) < kume_sayisi * 10:
                return None
            
            # Özellikleri ölçeklendir
            olceklendirici = RobustScaler()
            ozellikler_olceklenmis = olceklendirici.fit_transform(segmentasyon_verisi)
            
            # Kümeleme uygula
            if yontem == 'kmeans':
                model = KMeans(
                    n_clusters=kume_sayisi,
                    random_state=42,
                    n_init=10,
                    max_iter=300
                )
            elif yontem == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=10)
            elif yontem == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=kume_sayisi)
            else:
                model = KMeans(n_clusters=kume_sayisi, random_state=42)
            
            kumeler = model.fit_predict(ozellikler_olceklenmis)
            
            sonuc_df = df.copy()
            sonuc_df['Kume'] = kumeler
            
            # Küme isimlendirme
            kume_isimleri = {
                0: 'Büyüyen Ürünler',
                1: 'Olgun Ürünler',
                2: 'İnovatif Ürünler',
                3: 'Riskli Ürünler',
                4: 'Niş Ürünler',
                5: 'Hacim Ürünleri',
                6: 'Premium Ürünler',
                7: 'Ekonomik Ürünler'
            }
            
            sonuc_df['Kume_Adi'] = sonuc_df['Kume'].map(
                lambda x: kume_isimleri.get(x, f'Kume_{x}')
            )
            
            # Küme metriklerini hesapla
            try:
                siluet_skoru = silhouette_score(ozellikler_olceklenmis, kumeler)
                sonuc_df.attrs['siluet_skoru'] = siluet_skoru
            except:
                pass
            
            return sonuc_df
            
        except Exception as e:
            st.warning(f"Segmentasyon hatası: {str(e)}")
            return None
    
    @staticmethod
    def stratejik_icgoruler_uret(df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Veriden stratejik içgörüler üret.
        
        Args:
            df: Girdi DataFrame
            
        Returns:
            İçgörü sözlükleri listesi
        """
        icgoruler = []
        
        try:
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if not satis_sutunlari:
                return icgoruler
            
            son_satis_sutun = satis_sutunlari[-1]
            yil = VeriIsleyici.sutundan_yil_cikar(son_satis_sutun) or "Son Yıl"
            
            # En iyi ürünler
            en_iyi_urunler = df.nlargest(10, son_satis_sutun)
            en_iyi_pay = (en_iyi_urunler[son_satis_sutun].sum() / df[son_satis_sutun].sum() * 100) if df[son_satis_sutun].sum() > 0 else 0
            
            icgoruler.append({
                'tur': 'basarili',
                'baslik': f'🏆 En İyi 10 Ürün - {yil}',
                'aciklama': f"En iyi 10 ürün toplam pazarın %{en_iyi_pay:.1f}'ini oluşturuyor."
            })
            
            # Hızlı büyüyen ürünler
            buyume_sutunlari = [sutun for sutun in df.columns if 'Buyume_' in sutun]
            if buyume_sutunlari:
                son_buyume_sutun = buyume_sutunlari[-1]
                en_hizli_buyume = df.nlargest(10, son_buyume_sutun)
                ort_buyume = en_hizli_buyume[son_buyume_sutun].mean()
                
                icgoruler.append({
                    'tur': 'bilgi',
                    'baslik': f'🚀 En Hızlı Büyüyen 10 Ürün',
                    'aciklama': f"En hızlı büyüyen ürünler ortalama %{ort_buyume:.1f} büyüme gösteriyor."
                })
            
            # Pazar lideri
            if 'Sirket' in df.columns:
                en_iyi_sirketler = df.groupby('Sirket')[son_satis_sutun].sum().nlargest(5)
                en_iyi_sirket = en_iyi_sirketler.index[0]
                en_iyi_sirket_payi = (en_iyi_sirketler.iloc[0] / df[son_satis_sutun].sum()) * 100
                
                icgoruler.append({
                    'tur': 'uyari',
                    'baslik': f'🏢 Pazar Lideri - {yil}',
                    'aciklama': f"{en_iyi_sirket}, %{en_iyi_sirket_payi:.1f} pazar payı ile lider."
                })
            
            # En büyük pazar
            if 'Ulke' in df.columns:
                en_iyi_ulkeler = df.groupby('Ulke')[son_satis_sutun].sum().nlargest(5)
                en_iyi_ulke = en_iyi_ulkeler.index[0]
                en_iyi_ulke_payi = (en_iyi_ulkeler.iloc[0] / df[son_satis_sutun].sum()) * 100
                
                icgoruler.append({
                    'tur': 'cografi',
                    'baslik': f'🌍 En Büyük Pazar - {yil}',
                    'aciklama': f"{en_iyi_ulke}, %{en_iyi_ulke_payi:.1f} pay ile en büyük pazar."
                })
            
            # Fiyat analizi
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            if fiyat_sutunlari:
                ort_fiyat = df[fiyat_sutunlari[-1]].mean()
                fiyat_std = df[fiyat_sutunlari[-1]].std()
                
                icgoruler.append({
                    'tur': 'fiyat',
                    'baslik': f'💰 Fiyat Analizi - {yil}',
                    'aciklama': f"Ortalama fiyat: ${ort_fiyat:.2f} (Std: ${fiyat_std:.2f})"
                })
            
            # Uluslararası Ürün
            if 'Uluslararasi_Urun' in df.columns:
                uluslararasi_df = df[df['Uluslararasi_Urun'] == 1]
                yerel_df = df[df['Uluslararasi_Urun'] == 0]
                
                uluslararasi_sayisi = len(uluslararasi_df)
                uluslararasi_payi = (uluslararasi_df[son_satis_sutun].sum() / df[son_satis_sutun].sum() * 100) if df[son_satis_sutun].sum() > 0 else 0
                
                icgoruler.append({
                    'tur': 'uluslararasi',
                    'baslik': f'🌐 Uluslararası Ürün Analizi',
                    'aciklama': f"{uluslararasi_sayisi} Uluslararası Ürün, pazarın %{uluslararasi_payi:.1f}'ini oluşturuyor."
                })
            
            return icgoruler
            
        except Exception as e:
            st.warning(f"İçgörü üretme uyarısı: {str(e)}")
            return []

# ================================================
# 5. PROFESYONEL GÖRSELLEŞTİRİCİ
# ================================================

class ProfesyonelGorsellestirici:
    """
    Kurumsal seviye grafikler ve interaktif gösterge panelleri ile
    gelişmiş görselleştirme motoru.
    """
    
    @staticmethod
    def gosterge_paneli_metrikleri_olustur(df: pd.DataFrame, metrikler: Dict[str, Any]) -> None:
        """Gösterge paneli metrik kartları oluştur."""
        
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                toplam_satis = metrikler.get('Toplam_Pazar_Degeri', 0)
                satis_yili = metrikler.get('Son_Satis_Yili', '')
                st.markdown(f"""
                <div class="ozel-metrik-kart birincil">
                    <div class="ozel-metrik-etiket">TOPLAM PAZAR DEĞERİ</div>
                    <div class="ozel-metrik-deger">${toplam_satis/1e6:.1f}M</div>
                    <div class="ozel-metrik-trend">
                        <span class="rozet rozet-basarili">{satis_yili}</span>
                        <span>Toplam Pazar</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                ort_buyume = metrikler.get('Ort_Buyume_Orani', 0)
                buyume_sinifi = "basarili" if ort_buyume > 0 else "tehlike"
                st.markdown(f"""
                <div class="ozel-metrik-kart {buyume_sinifi}">
                    <div class="ozel-metrik-etiket">ORTALAMA BÜYÜME</div>
                    <div class="ozel-metrik-deger">{ort_buyume:.1f}%</div>
                    <div class="ozel-metrik-trend">
                        <span class="rozet rozet-bilgi">Yıllık</span>
                        <span>Yıllık Büyüme</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                hhi = metrikler.get('HHI_Indeksi', 0)
                hhi_durum = "tehlike" if hhi > 2500 else "uyari" if hhi > 1500 else "basarili"
                hhi_metin = "Monopolistik" if hhi > 2500 else "Oligopol" if hhi > 1500 else "Rekabetçi"
                st.markdown(f"""
                <div class="ozel-metrik-kart {hhi_durum}">
                    <div class="ozel-metrik-etiket">REKABET YOĞUNLUĞU</div>
                    <div class="ozel-metrik-deger">{hhi:.0f}</div>
                    <div class="ozel-metrik-trend">
                        <span class="rozet rozet-uyari">HHI İndeksi</span>
                        <span>{hhi_metin}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                uluslararasi_pay = metrikler.get('Uluslararasi_Urun_Payi', 0)
                uluslararasi_renk = "basarili" if uluslararasi_pay > 20 else "uyari" if uluslararasi_pay > 10 else "bilgi"
                st.markdown(f"""
                <div class="ozel-metrik-kart {uluslararasi_renk}">
                    <div class="ozel-metrik-etiket">ULUSLARARASI ÜRÜNLER</div>
                    <div class="ozel-metrik-deger">{uluslararasi_pay:.1f}%</div>
                    <div class="ozel-metrik-trend">
                        <span class="rozet rozet-bilgi">Küresel</span>
                        <span>Çoklu Pazar</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # İkinci sıra
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                benzersiz_molekuller = metrikler.get('Benzersiz_Molekuller', 0)
                st.markdown(f"""
                <div class="ozel-metrik-kart">
                    <div class="ozel-metrik-etiket">MOLEKÜL ÇEŞİTLİLİĞİ</div>
                    <div class="ozel-metrik-deger">{benzersiz_molekuller:,}</div>
                    <div class="ozel-metrik-trend">
                        <span class="rozet rozet-basarili">Benzersiz</span>
                        <span>Farklı Moleküller</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                ort_fiyat = metrikler.get('Ort_Fiyat', 0)
                st.markdown(f"""
                <div class="ozel-metrik-kart">
                    <div class="ozel-metrik-etiket">ORTALAMA FİYAT</div>
                    <div class="ozel-metrik-deger">${ort_fiyat:.2f}</div>
                    <div class="ozel-metrik-trend">
                        <span class="rozet rozet-bilgi">Birim Başına</span>
                        <span>Ortalama</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                yuksek_buyume_yuzdesi = metrikler.get('Yuksek_Buyume_Yuzdesi', 0)
                st.markdown(f"""
                <div class="ozel-metrik-kart basarili">
                    <div class="ozel-metrik-etiket">YÜKSEK BÜYÜME</div>
                    <div class="ozel-metrik-deger">{yuksek_buyume_yuzdesi:.1f}%</div>
                    <div class="ozel-metrik-trend">
                        <span class="rozet rozet-basarili">%20+</span>
                        <span>Hızlı Büyüyen</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                ulke_kapsami = metrikler.get('Ulke_Kapsami', 0)
                st.markdown(f"""
                <div class="ozel-metrik-kart">
                    <div class="ozel-metrik-etiket">COĞRAFİ YAYILIM</div>
                    <div class="ozel-metrik-deger">{ulke_kapsami}</div>
                    <div class="ozel-metrik-trend">
                        <span class="rozet rozet-bilgi">Ülke</span>
                        <span>Küresel Kapsam</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"Metrik kart oluşturma hatası: {str(e)}")
    
    @staticmethod
    def satis_trend_grafigi(df: pd.DataFrame) -> Optional[go.Figure]:
        """Satış trend görselleştirmesi oluştur."""
        
        try:
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if len(satis_sutunlari) < 2:
                return None
            
            yillik_veri = []
            for sutun in sorted(satis_sutunlari):
                yil = VeriIsleyici.sutundan_yil_cikar(sutun)
                if yil:
                    yillik_veri.append({
                        'Yil': yil,
                        'Toplam_Satis': df[sutun].sum(),
                        'Ort_Satis': df[sutun].mean(),
                        'Urun_Sayisi': (df[sutun] > 0).sum()
                    })
            
            if len(yillik_veri) < 2:
                return None
            
            yillik_df = pd.DataFrame(yillik_veri)
            
            fig = go.Figure()
            
            # Toplam satış çubuk
            fig.add_trace(go.Bar(
                x=yillik_df['Yil'],
                y=yillik_df['Toplam_Satis'],
                name='Toplam Satış',
                marker_color='#2d7dd2',
                text=[f'${x/1e6:.0f}M' for x in yillik_df['Toplam_Satis']],
                textposition='auto'
            ))
            
            # Ortalama satış çizgisi (ikincil eksen)
            fig.add_trace(go.Scatter(
                x=yillik_df['Yil'],
                y=yillik_df['Ort_Satis'],
                name='Ortalama Satış',
                mode='lines+markers',
                line=dict(color='#2acaea', width=3),
                marker=dict(size=10),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Satış Trend Analizi',
                xaxis_title='Yıl',
                yaxis_title='Toplam Satış (USD)',
                yaxis2=dict(
                    title='Ortalama Satış (USD)',
                    overlaying='y',
                    side='right'
                ),
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Satış trend grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def pazar_payi_analizi(df: pd.DataFrame) -> Optional[go.Figure]:
        """Pazar payı görselleştirmesi oluştur."""
        
        try:
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if not satis_sutunlari or 'Sirket' not in df.columns:
                return None
            
            son_satis_sutun = satis_sutunlari[-1]
            
            sirket_satisleri = df.groupby('Sirket')[son_satis_sutun].sum().sort_values(ascending=False)
            en_iyi_sirketler = sirket_satisleri.nlargest(15)
            diger_satisler = sirket_satisleri.iloc[15:].sum() if len(sirket_satisleri) > 15 else 0
            
            pasta_verisi = en_iyi_sirketler.copy()
            if diger_satisler > 0:
                pasta_verisi['Diğerleri'] = diger_satisler
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Pazar Payı Dağılımı', 'İlk 10 Şirket Satışları'),
                specs=[[{'type': 'domain'}, {'type': 'bar'}]],
                column_widths=[0.4, 0.6]
            )
            
            # Pasta grafik
            fig.add_trace(
                go.Pie(
                    labels=pasta_verisi.index,
                    values=pasta_verisi.values,
                    hole=0.4,
                    marker_colors=px.colors.qualitative.Bold,
                    textinfo='percent+label',
                    textposition='outside'
                ),
                row=1, col=1
            )
            
            # Çubuk grafik
            fig.add_trace(
                go.Bar(
                    x=en_iyi_sirketler.values[:10],
                    y=en_iyi_sirketler.index[:10],
                    orientation='h',
                    marker_color='#2d7dd2',
                    text=[f'${x/1e6:.1f}M' for x in en_iyi_sirketler.values[:10]],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False,
                title_text="Pazar Yoğunlaşma Analizi",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Pazar payı grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def dunya_haritasi_gorsellestirme(df: pd.DataFrame) -> Optional[go.Figure]:
        """Dünya haritası görselleştirmesi oluştur."""
        
        try:
            if 'Ulke' not in df.columns:
                return None
            
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if not satis_sutunlari:
                return None
            
            son_satis_sutun = satis_sutunlari[-1]
            
            ulke_satisleri = df.groupby('Ulke')[son_satis_sutun].sum().reset_index()
            ulke_satisleri.columns = ['Ulke', 'Toplam_Satis']
            
            # Ülke ismi haritalama
            ulke_haritalama = {
                'USA': 'United States',
                'US': 'United States',
                'U.S.A': 'United States',
                'UK': 'United Kingdom',
                'U.K': 'United Kingdom',
                'UAE': 'United Arab Emirates',
                'U.A.E': 'United Arab Emirates',
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
                'Turkey': 'Türkiye',
                'Turkiye': 'Türkiye'
            }
            
            ulke_satisleri['Ulke'] = ulke_satisleri['Ulke'].replace(ulke_haritalama)
            
            fig = px.choropleth(
                ulke_satisleri,
                locations='Ulke',
                locationmode='country names',
                color='Toplam_Satis',
                hover_name='Ulke',
                hover_data={'Toplam_Satis': ':.2f'},
                color_continuous_scale='Viridis',
                title='Global İlaç Pazarı Dağılımı',
                projection='natural earth'
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                geo=dict(
                    bgcolor='rgba(0,0,0,0)',
                    lakecolor='#1e3a5f',
                    landcolor='#2d4a7a',
                    subunitcolor='#64748b'
                ),
                coloraxis_colorbar=dict(
                    title="Toplam Satış (USD)",
                    tickprefix="$"
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Dünya haritası hatası: {str(e)}")
            return None
    
    @staticmethod
    def gunes_patlama_hierarsi_grafigi(df: pd.DataFrame) -> Optional[go.Figure]:
        """Güneş patlaması hiyerarşi görselleştirmesi oluştur."""
        
        try:
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if not satis_sutunlari:
                return None
            
            son_satis_sutun = satis_sutunlari[-1]
            
            # Hiyerarşi verisi hazırla
            hiyerarsi_verisi = []
            
            if 'Sirket' in df.columns and 'Molekul' in df.columns:
                for _, satir in df.iterrows():
                    hiyerarsi_verisi.append({
                        'etiketler': satir['Molekul'],
                        'ebeveynler': satir['Sirket'],
                        'degerler': satir[son_satis_sutun]
                    })
                
                # Şirket seviyesi ekle
                sirket_toplamlari = df.groupby('Sirket')[son_satis_sutun].sum()
                for sirket, toplam in sirket_toplamlari.items():
                    hiyerarsi_verisi.append({
                        'etiketler': sirket,
                        'ebeveynler': '',
                        'degerler': toplam
                    })
                
                hiyerarsi_df = pd.DataFrame(hiyerarsi_verisi)
                
                fig = go.Figure(go.Sunburst(
                    labels=hiyerarsi_df['etiketler'],
                    parents=hiyerarsi_df['ebeveynler'],
                    values=hiyerarsi_df['degerler'],
                    branchvalues="total",
                    marker=dict(
                        colorscale='Viridis',
                        cmid=hiyerarsi_df['degerler'].median()
                    ),
                    hovertemplate='<b>%{label}</b><br>Satış: $%{value:.2f}<br><extra></extra>'
                ))
                
                fig.update_layout(
                    title='Pazar Hiyerarşisi - Şirket > Molekül',
                    height=600,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc'
                )
                
                return fig
            
            return None
            
        except Exception as e:
            st.warning(f"Güneş patlaması grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def radar_karsilastirma_grafigi(df: pd.DataFrame, varliklar: List[str]) -> Optional[go.Figure]:
        """Varlık karşılaştırması için radar grafiği oluştur."""
        
        try:
            if 'Sirket' not in df.columns or len(varliklar) == 0:
                return None
            
            # Karşılaştırma metrikleri
            metrikler = []
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            buyume_sutunlari = [sutun for sutun in df.columns if 'Buyume_' in sutun]
            
            if satis_sutunlari:
                metrikler.append(('Pazar Payı', satis_sutunlari[-1]))
            if buyume_sutunlari:
                metrikler.append(('Büyüme Oranı', buyume_sutunlari[-1]))
            if 'Pazar_Payi' in df.columns:
                metrikler.append(('Pazar Pozisyonu', 'Pazar_Payi'))
            
            if len(metrikler) < 3:
                return None
            
            fig = go.Figure()
            
            for varlik in varliklar[:5]:  # Okunabilirlik için 5 ile sınırla
                varlik_df = df[df['Sirket'] == varlik]
                
                if len(varlik_df) == 0:
                    continue
                
                degerler = []
                for _, metrik_sutun in metrikler:
                    if metrik_sutun in varlik_df.columns:
                        degerler.append(varlik_df[metrik_sutun].mean())
                    else:
                        degerler.append(0)
                
                # Değerleri 0-100 ölçeğine normalize et
                max_degerler = [df[sutun].max() for _, sutun in metrikler]
                normalize_edilmis_degerler = [(v / m * 100) if m > 0 else 0 for v, m in zip(degerler, max_degerler)]
                
                fig.add_trace(go.Scatterpolar(
                    r=normalize_edilmis_degerler,
                    theta=[isim for isim, _ in metrikler],
                    fill='toself',
                    name=varlik
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
                font_color='#f8fafc',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Radar grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def tahmin_gorsellestirme(
        tarihsel_df: pd.DataFrame,
        tahmin_df: Optional[pd.DataFrame]
    ) -> Optional[go.Figure]:
        """Tarihsel veriyi ve tahminleri görselleştir."""
        
        try:
            if tahmin_df is None or len(tahmin_df) == 0:
                return None
            
            satis_sutunlari = [sutun for sutun in tarihsel_df.columns if 'Satış_' in sutun]
            if not satis_sutunlari:
                return None
            
            # Tarihsel veri
            yillar = []
            degerler = []
            for sutun in sorted(satis_sutunlari):
                yil = VeriIsleyici.sutundan_yil_cikar(sutun)
                if yil:
                    yillar.append(yil)
                    degerler.append(tarihsel_df[sutun].sum())
            
            fig = go.Figure()
            
            # Tarihsel
            fig.add_trace(go.Scatter(
                x=yillar,
                y=degerler,
                mode='lines+markers',
                name='Tarihsel',
                line=dict(color='#2d7dd2', width=3),
                marker=dict(size=10)
            ))
            
            # Tahmin
            fig.add_trace(go.Scatter(
                x=tahmin_df['Yil'],
                y=tahmin_df['Tahmin'],
                mode='lines+markers',
                name='Tahmin',
                line=dict(color='#2acaea', width=3, dash='dash'),
                marker=dict(size=10)
            ))
            
            # Güven aralığı
            fig.add_trace(go.Scatter(
                x=list(tahmin_df['Yil']) + list(tahmin_df['Yil'][::-1]),
                y=list(tahmin_df['Ust_Sinir']) + list(tahmin_df['Alt_Sinir'][::-1]),
                fill='toself',
                fillcolor='rgba(42, 202, 234, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='%95 Güven'
            ))
            
            fig.update_layout(
                title='Pazar Tahmini - Güven Aralıkları',
                xaxis_title='Yıl',
                yaxis_title='Toplam Satış (USD)',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Tahmin görselleştirme hatası: {str(e)}")
            return None
    
    @staticmethod
    def anomali_sacilim_grafigi(anomali_df: pd.DataFrame) -> Optional[go.Figure]:
        """Anomali tespiti saçılım grafiği oluştur."""
        
        try:
            if anomali_df is None or 'Anomali_Skoru' not in anomali_df.columns:
                return None
            
            satis_sutunlari = [sutun for sutun in anomali_df.columns if 'Satış_' in sutun]
            buyume_sutunlari = [sutun for sutun in anomali_df.columns if 'Buyume_' in sutun]
            
            if not satis_sutunlari or not buyume_sutunlari:
                return None
            
            son_satis_sutun = satis_sutunlari[-1]
            son_buyume_sutun = buyume_sutunlari[-1]
            
            # Çok büyükse örnekle
            cizim_df = anomali_df if len(anomali_df) <= 5000 else anomali_df.sample(5000, random_state=42)
            
            fig = px.scatter(
                cizim_df,
                x=son_satis_sutun,
                y=son_buyume_sutun,
                color='Anomali_Kategorisi',
                size=abs(cizim_df['Anomali_Skoru']),
                hover_name='Molekul' if 'Molekul' in cizim_df.columns else None,
                title='Anomali Tespiti - Satış vs Büyüme',
                labels={
                    son_satis_sutun: 'Satış (USD)',
                    son_buyume_sutun: 'Büyüme Oranı (%)'
                },
                color_discrete_map={
                    'Yüksek Risk': '#eb5757',
                    'Orta Risk': '#f2c94c',
                    'Normal': '#2dd2a3'
                }
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Anomali grafiği hatası: {str(e)}")
            return None

# ================================================
# 6. RAPOR ÜRETİCİ
# ================================================

class RaporUretici:
    """
    Excel ve PDF dışa aktarım yetenekleri ile gelişmiş rapor üretimi.
    """
    
    @staticmethod
    def excel_raporu_uret(
        df: pd.DataFrame,
        metrikler: Dict[str, Any],
        icgoruler: List[Dict[str, str]],
        dosya_adi: str = "pharma_raporu.xlsx"
    ) -> BytesIO:
        """
        Kapsamlı Excel raporu üret.
        
        Args:
            df: Girdi DataFrame
            metrikler: Hesaplanmış metrikler
            icgoruler: Stratejik içgörüler
            dosya_adi: Çıktı dosya adı
            
        Returns:
            Excel dosyası içeren BytesIO nesnesi
        """
        try:
            cikti = BytesIO()
            
            with pd.ExcelWriter(cikti, engine='xlsxwriter') as yazici:
                calisma_kitabi = yazici.book
                
                # Formatları tanımla
                baslik_format = calisma_kitabi.add_format({
                    'bold': True,
                    'font_color': 'white',
                    'bg_color': '#2d7dd2',
                    'border': 1
                })
                
                sayi_format = calisma_kitabi.add_format({
                    'num_format': '#,##0.00',
                    'border': 1
                })
                
                yuzde_format = calisma_kitabi.add_format({
                    'num_format': '0.00%',
                    'border': 1
                })
                
                # Sayfa 1: Yönetici Özeti
                ozet_veri = pd.DataFrame([
                    ['Toplam Pazar Değeri', f"${metrikler.get('Toplam_Pazar_Degeri', 0)/1e6:.2f}M"],
                    ['Ortalama Büyüme Oranı', f"{metrikler.get('Ort_Buyume_Orani', 0):.2f}%"],
                    ['HHI İndeksi', f"{metrikler.get('HHI_Indeksi', 0):.2f}"],
                    ['Benzersiz Moleküller', metrikler.get('Benzersiz_Molekuller', 0)],
                    ['Ülke Kapsamı', metrikler.get('Ulke_Kapsami', 0)],
                    ['Uluslararası Ürün Payı', f"{metrikler.get('Uluslararasi_Urun_Payi', 0):.2f}%"]
                ], columns=['Metrik', 'Değer'])
                
                ozet_veri.to_excel(yazici, sheet_name='Yönetici Özeti', index=False)
                calisma_sayfasi = yazici.sheets['Yönetici Özeti']
                calisma_sayfasi.set_column('A:A', 30)
                calisma_sayfasi.set_column('B:B', 20)
                
                # Sayfa 2: Detaylı Veri
                df.to_excel(yazici, sheet_name='Detaylı Veri', index=False)
                calisma_sayfasi = yazici.sheets['Detaylı Veri']
                
                for sutun_numarasi, deger in enumerate(df.columns.values):
                    calisma_sayfasi.write(0, sutun_numarasi, deger, baslik_format)
                
                # Sayfa 3: Stratejik İçgörüler
                if icgoruler:
                    icgoru_veri = pd.DataFrame(icgoruler)
                    icgoru_veri.to_excel(yazici, sheet_name='Stratejik İçgörüler', index=False)
                
                # Sayfa 4: En İyi Ürünler
                satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
                if satis_sutunlari:
                    son_satis_sutun = satis_sutunlari[-1]
                    if 'Molekul' in df.columns and 'Sirket' in df.columns:
                        en_iyi_urunler = df[['Molekul', 'Sirket', son_satis_sutun]].nlargest(50, son_satis_sutun)
                    else:
                        en_iyi_urunler = df[[son_satis_sutun]].nlargest(50, son_satis_sutun)
                    en_iyi_urunler.to_excel(yazici, sheet_name='İlk 50 Ürün', index=False)
                
                # Sayfa 5: Şirket Analizi
                if 'Sirket' in df.columns and satis_sutunlari:
                    sirket_analizi = df.groupby('Sirket')[son_satis_sutun].agg(['sum', 'mean', 'count']).round(2)
                    sirket_analizi.columns = ['Toplam Satış', 'Ort Satış', 'Ürün Sayısı']
                    sirket_analizi = sirket_analizi.sort_values('Toplam Satış', ascending=False)
                    sirket_analizi.to_excel(yazici, sheet_name='Şirket Analizi')
            
            cikti.seek(0)
            return cikti
            
        except Exception as e:
            st.error(f"Excel rapor üretme hatası: {str(e)}")
            return BytesIO()
    
    @staticmethod
    def html_raporu_uret(
        df: pd.DataFrame,
        metrikler: Dict[str, Any],
        icgoruler: List[Dict[str, str]]
    ) -> str:
        """HTML raporu üret."""
        
        try:
            html_icerik = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>PharmaIntelligence Pro Rapor</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background: linear-gradient(135deg, #0c1a32, #14274e);
                        color: #f8fafc;
                    }}
                    .baslik {{
                        text-align: center;
                        padding: 30px;
                        background: rgba(30, 58, 95, 0.8);
                        border-radius: 10px;
                        margin-bottom: 30px;
                    }}
                    .metrik-ag {{
                        display: grid;
                        grid-template-columns: repeat(3, 1fr);
                        gap: 20px;
                        margin-bottom: 30px;
                    }}
                    .metrik-kart {{
                        background: rgba(30, 58, 95, 0.6);
                        padding: 20px;
                        border-radius: 10px;
                        border: 1px solid #2d7dd2;
                    }}
                    .metrik-deger {{
                        font-size: 2rem;
                        font-weight: bold;
                        color: #2acaea;
                    }}
                    .icgoruler {{
                        margin-top: 30px;
                    }}
                    .icgoru {{
                        background: rgba(30, 58, 95, 0.6);
                        padding: 15px;
                        margin-bottom: 15px;
                        border-left: 4px solid #2dd2a3;
                        border-radius: 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="baslik">
                    <h1>PharmaIntelligence Pro</h1>
                    <h2>Pazar Analiz Raporu</h2>
                    <p>Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                <div class="metrik-ag">
                    <div class="metrik-kart">
                        <h3>Toplam Pazar Değeri</h3>
                        <div class="metrik-deger">${metrikler.get('Toplam_Pazar_Degeri', 0)/1e6:.1f}M</div>
                    </div>
                    <div class="metrik-kart">
                        <h3>Ortalama Büyüme</h3>
                        <div class="metrik-deger">{metrikler.get('Ort_Buyume_Orani', 0):.1f}%</div>
                    </div>
                    <div class="metrik-kart">
                        <h3>HHI İndeksi</h3>
                        <div class="metrik-deger">{metrikler.get('HHI_Indeksi', 0):.0f}</div>
                    </div>
                </div>
                <div class="icgoruler">
                    <h2>Stratejik İçgörüler</h2>
            """
        
            for icgoru in icgoruler[:10]:
                html_icerik += f"""
                    <div class="icgoru">
                        <h3>{icgoru['baslik']}</h3>
                        <p>{icgoru['aciklama']}</p>
                    </div>
                """
        
            html_icerik += """
                </div>
            </body>
            </html>
            """
        
            return html_icerik
        
        except Exception as e:
            st.error(f"HTML rapor üretme hatası: {str(e)}")
            return ""

# ================================================
# 7. ANA UYGULAMA
# ================================================

def ana():
    """Ana uygulama fonksiyonu."""
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="pharma-baslik">💊 PHARMAINTELLIGENCE PRO</h1>
        <p class="pharma-alt-baslik">
        Yapay zeka destekli tahminleme, anomali tespiti ve kapsamlı stratejik içgörüler 
        ile kurumsal ilaç pazarı analitiği.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Oturum durumunu başlat
    if 'veri' not in st.session_state:
        st.session_state.veri = None
    if 'filtrelenmis_veri' not in st.session_state:
        st.session_state.filtrelenmis_veri = None
    if 'metrikler' not in st.session_state:
        st.session_state.metrikler = None
    if 'icgoruler' not in st.session_state:
        st.session_state.icgoruler = []
    if 'aktif_filtreler' not in st.session_state:
        st.session_state.aktif_filtreler = {}
    if 'uluslararasi_analiz' not in st.session_state:
        st.session_state.uluslararasi_analiz = None
    if 'anomali_verisi' not in st.session_state:
        st.session_state.anomali_verisi = None
    if 'tahmin_verisi' not in st.session_state:
        st.session_state.tahmin_verisi = None

    # Yan çubuk
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">🎛️ KONTROL PANELİ</h2>', unsafe_allow_html=True)
        
        with st.expander("📁 VERİ YÜKLEME", expanded=True):
            yuklenen_dosya = st.file_uploader(
                "Excel/CSV Dosyası Yükle",
                type=['xlsx', 'xls', 'csv'],
                help="1M+ satırı destekler"
            )
            
            if yuklenen_dosya:
                st.info("⚠️ Tüm veri seti yüklenecek")
                st.info(f"Dosya: {yuklenen_dosya.name}")
                
                if st.button("🚀 Veriyi Yükle & Analiz Et", type="primary", use_container_width=True):
                    with st.spinner("Tüm veri seti işleniyor..."):
                        isleyici = VeriIsleyici()
                        
                        veri = isleyici.buyuk_veri_seti_yukle(yuklenen_dosya, ornek_boyut=None)
                        
                        if veri is not None and len(veri) > 0:
                            veri = isleyici.analitik_veri_hazirla(veri)
                            
                            st.session_state.veri = veri
                            st.session_state.filtrelenmis_veri = veri.copy()
                            
                            analitik = AnalitikMotoru()
                            st.session_state.metrikler = analitik.kapsamli_metrikleri_hesapla(veri)
                            st.session_state.icgoruler = analitik.stratejik_icgoruler_uret(veri)
                            st.session_state.uluslararasi_analiz = analitik.uluslararasi_urun_analizi(veri)
                            
                            st.success(f"✅ {len(veri):,} satır başarıyla yüklendi!")
                            st.rerun()
        
        # Filtreler
        if st.session_state.veri is not None:
            veri = st.session_state.veri
            
            filtre_sistemi = GelismisFiltreSistemi()
            arama_terimi, filtre_yapisi, filtreleri_uygula, filtreleri_temizle = filtre_sistemi.filtre_yan_cubugu_olustur(veri)
            
            if filtreleri_uygula:
                with st.spinner("Filtreler uygulanıyor..."):
                    filtrelenmis_veri = filtre_sistemi.filtreleri_uygula(veri, arama_terimi, filtre_yapisi)
                    st.session_state.filtrelenmis_veri = filtrelenmis_veri
                    st.session_state.aktif_filtreler = filtre_yapisi
                    
                    analitik = AnalitikMotoru()
                    st.session_state.metrikler = analitik.kapsamli_metrikleri_hesapla(filtrelenmis_veri)
                    st.session_state.icgoruler = analitik.stratejik_icgoruler_uret(filtrelenmis_veri)
                    st.session_state.uluslararasi_analiz = analitik.uluslararasi_urun_analizi(filtrelenmis_veri)
                    
                    st.success(f"✅ Filtreler uygulandı: {len(filtrelenmis_veri):,} satır")
                    st.rerun()
            
            if filtreleri_temizle:
                st.session_state.filtrelenmis_veri = st.session_state.veri.copy()
                st.session_state.aktif_filtreler = {}
                st.session_state.metrikler = AnalitikMotoru().kapsamli_metrikleri_hesapla(st.session_state.veri)
                st.session_state.icgoruler = AnalitikMotoru().stratejik_icgoruler_uret(st.session_state.veri)
                st.session_state.uluslararasi_analiz = AnalitikMotoru().uluslararasi_urun_analizi(st.session_state.veri)
                st.success("✅ Filtreler temizlendi")
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #64748b;">
        <strong>PharmaIntelligence Pro</strong><br>
        v6.0 | Yapay Zeka Destekli Analitik<br>
        © 2024 Tüm hakları saklıdır
        </div>
        """, unsafe_allow_html=True)

    # Ana içerik
    if st.session_state.veri is None:
        hosgeldin_ekrani_goster()
        return

    veri = st.session_state.filtrelenmis_veri
    metrikler = st.session_state.metrikler
    icgoruler = st.session_state.icgoruler
    uluslararasi_analiz = st.session_state.uluslararasi_analiz

    # Filtre durumu
    if st.session_state.aktif_filtreler:
        filtre_bilgisi = f"🎯 **Aktif Filtreler:** "
        filtre_ogeleri = []
        
        for anahtar, deger in st.session_state.aktif_filtreler.items():
            if anahtar in ['Ulke', 'Sirket', 'Molekul']:
                if isinstance(deger, list):
                    if len(deger) > 3:
                        filtre_ogeleri.append(f"{anahtar}: {len(deger)} seçenek")
                    else:
                        filtre_ogeleri.append(f"{anahtar}: {', '.join(deger[:3])}")
            elif anahtar == 'satis_araligi':
                (min_deger, max_deger), sutun_adi = deger
                filtre_ogeleri.append(f"Satış: ${min_deger:,.0f}-${max_deger:,.0f}")
            elif anahtar == 'buyume_araligi':
                (min_deger, max_deger), sutun_adi = deger
                filtre_ogeleri.append(f"Büyüme: {min_deger:.1f}%-{max_deger:.1f}%")
            elif anahtar == 'pozitif_buyume':
                filtre_ogeleri.append("Pozitif Büyüme")
            elif anahtar == 'uluslararasi_filtre':
                filtre_ogeleri.append(deger)
        
        filtre_bilgisi += " | ".join(filtre_ogeleri)
        filtre_bilgisi += f" | **Gösterilen:** {len(veri):,} / {len(st.session_state.veri):,} satır"
        
        st.markdown(f'<div class="filtre-durumu">{filtre_bilgisi}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("❌ Tüm Filtreleri Temizle", use_container_width=True):
                st.session_state.filtrelenmis_veri = st.session_state.veri.copy()
                st.session_state.aktif_filtreler = {}
                st.session_state.metrikler = AnalitikMotoru().kapsamli_metrikleri_hesapla(st.session_state.veri)
                st.session_state.icgoruler = AnalitikMotoru().stratejik_icgoruler_uret(st.session_state.veri)
                st.session_state.uluslararasi_analiz = AnalitikMotoru().uluslararasi_urun_analizi(st.session_state.veri)
                st.success("✅ Tüm filtreler temizlendi")
                st.rerun()
    else:
        st.info(f"🎯 Aktif filtre yok | Gösterilen: {len(veri):,} satır")

    # Sekmeler
    sekme1, sekme2, sekme3, sekme4, sekme5, sekme6, sekme7, sekme8 = st.tabs([
        "📊 ÖZET",
        "📈 PAZAR ANALİZİ",
        "💰 FİYAT ANALİZİ",
        "🏆 REKABET",
        "🌍 ULUSLARARASI",
        "🔮 TAHMİNLEME",
        "⚠️ ANOMALİ TESPİTİ",
        "📑 RAPORLAMA"
    ])

    with sekme1:
        ozet_sekmesi_goster(veri, metrikler, icgoruler)

    with sekme2:
        pazar_analizi_sekmesi_goster(veri)

    with sekme3:
        fiyat_analizi_sekmesi_goster(veri)

    with sekme4:
        rekabet_sekmesi_goster(veri, metrikler)

    with sekme5:
        uluslararasi_sekmesi_goster(veri, uluslararasi_analiz, metrikler)

    with sekme6:
        tahminleme_sekmesi_goster(veri)

    with sekme7:
        anomali_sekmesi_goster(veri)

    with sekme8:
        raporlama_sekmesi_goster(veri, metrikler, icgoruler, uluslararasi_analiz)

# ================================================
# 8. SEKME FONKSİYONLARI
# ================================================

def hosgeldin_ekrani_goster():
    """Hoşgeldin ekranını göster."""
    st.markdown("""
    <div class="hosgeldin-container">
        <div class="hosgeldin-icon">💊</div>
        <h2 style="color: #f1f5f9; margin-bottom: 1rem;">PharmaIntelligence Pro'ya Hoş Geldiniz</h2>
        <p style="color: #cbd5e1; margin-bottom: 2rem; line-height: 1.6;">
        İlaç pazarı verinizi yükleyerek yapay zeka destekli tahminleme, 
        anomali tespiti ve stratejik içgörüler gibi güçlü analitiklere erişin.
        </p>
    </div>
    """, unsafe_allow_html=True)

def ozet_sekmesi_goster(df: pd.DataFrame, metrikler: Dict, icgoruler: List[Dict]):
    """Özet sekmesini göster."""
    st.markdown('<h2 class="bolum-baslik">Özet & Performans Göstergeleri</h2>', unsafe_allow_html=True)

    gorsellestirici = ProfesyonelGorsellestirici()
    gorsellestirici.gosterge_paneli_metrikleri_olustur(df, metrikler)

    st.markdown('<h3 class="alt-bolum-baslik">🔍 Stratejik İçgörüler</h3>', unsafe_allow_html=True)

    if icgoruler:
        icgoru_sutunlari = st.columns(2)
        
        for indeks, icgoru in enumerate(icgoruler[:6]):
            with icgoru_sutunlari[indeks % 2]:
                ikon_haritasi = {
                    'uyari': '⚠️',
                    'basarili': '✅',
                    'bilgi': 'ℹ️',
                    'cografi': '🌍',
                    'fiyat': '💰',
                    'uluslararasi': '🌐'
                }
                ikon = ikon_haritasi.get(icgoru['tur'], '💡')
                
                st.markdown(f"""
                <div class="icgoru-kart {icgoru['tur']}">
                    <div class="icgoru-icon">{ikon}</div>
                    <div class="icgoru-baslik">{icgoru['baslik']}</div>
                    <div class="icgoru-icerik">{icgoru['aciklama']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<h3 class="alt-bolum-baslik">📋 Veri Önizleme</h3>', unsafe_allow_html=True)

    onizleme_col1, onizleme_col2 = st.columns([1, 3])

    with onizleme_col1:
        satir_sayisi = st.slider("Gösterilecek Satır Sayısı", 10, 5000, 100, 10, key="satir_onizleme")
        
        mevcut_sutunlar = df.columns.tolist()
        varsayilan_sutunlar = []
        
        oncelikli_sutunlar = ['Molekul', 'Sirket', 'Ulke', 'Satış_2024', 'Buyume_2023_2024']
        for sutun in oncelikli_sutunlar:
            if sutun in mevcut_sutunlar:
                varsayilan_sutunlar.append(sutun)
                if len(varsayilan_sutunlar) >= 5:
                    break
        
        if len(varsayilan_sutunlar) < 5:
            varsayilan_sutunlar.extend([sutun for sutun in mevcut_sutunlar[:5] if sutun not in varsayilan_sutunlar])
        
        gosterilecek_sutunlar = st.multiselect(
            "Gösterilecek Sütunlar",
            options=mevcut_sutunlar,
            default=varsayilan_sutunlar[:min(5, len(varsayilan_sutunlar))],
            key="sutun_onizleme"
        )

    with onizleme_col2:
        if gosterilecek_sutunlar:
            st.dataframe(
                df[gosterilecek_sutunlar].head(satir_sayisi),
                use_container_width=True,
                height=400
            )
        else:
            st.dataframe(
                df.head(satir_sayisi),
                use_container_width=True,
                height=400
            )

def pazar_analizi_sekmesi_goster(df: pd.DataFrame):
    """Pazar analizi sekmesini göster."""
    st.markdown('<h2 class="bolum-baslik">Pazar Analizi & Trendler</h2>', unsafe_allow_html=True)

    gorsellestirici = ProfesyonelGorsellestirici()

    st.markdown('<h3 class="alt-bolum-baslik">📈 Satış Trendleri</h3>', unsafe_allow_html=True)
    trend_grafigi = gorsellestirici.satis_trend_grafigi(df)
    if trend_grafigi:
        st.plotly_chart(trend_grafigi, use_container_width=True, config={'displayModeBar': True})

    st.markdown('<h3 class="alt-bolum-baslik">🏆 Pazar Payı Analizi</h3>', unsafe_allow_html=True)
    pay_grafigi = gorsellestirici.pazar_payi_analizi(df)
    if pay_grafigi:
        st.plotly_chart(pay_grafigi, use_container_width=True, config={'displayModeBar': True})

    st.markdown('<h3 class="alt-bolum-baslik">🌐 Coğrafi Dağılım</h3>', unsafe_allow_html=True)
    dunya_haritasi = gorsellestirici.dunya_haritasi_gorsellestirme(df)
    if dunya_haritasi:
        st.plotly_chart(dunya_haritasi, use_container_width=True, config={'displayModeBar': True})

def fiyat_analizi_sekmesi_goster(df: pd.DataFrame):
    """Fiyat analizi sekmesini göster."""
    st.markdown('<h2 class="bolum-baslik">Fiyat Analizi & Optimizasyon</h2>', unsafe_allow_html=True)

    fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]

    if not fiyat_sutunlari:
        st.info("Fiyat analizi için veri setinde ortalama fiyat sütunları gereklidir.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💰 Fiyat Dağılımı")
        son_fiyat_sutun = fiyat_sutunlari[-1]
        
        fig = px.histogram(
            df,
            x=son_fiyat_sutun,
            nbins=50,
            title='Fiyat Dağılımı',
            labels={son_fiyat_sutun: 'Fiyat (USD)'}
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Fiyat Segmentasyonu")
        
        fiyat_verisi = df[son_fiyat_sutun].dropna()
        if len(fiyat_verisi) > 0:
            segmentler = pd.cut(
                fiyat_verisi,
                bins=[0, 10, 50, 100, 500, float('inf')],
                labels=['Ekonomik (<$10)', 'Standart ($10-$50)', 'Premium ($50-$100)',
                       'Süper Premium ($100-$500)', 'Lüks (>$500)']
            )
            
            segment_sayilari = segmentler.value_counts()
            
            fig = px.bar(
                x=segment_sayilari.index,
                y=segment_sayilari.values,
                title='Fiyat Segmentlerine Göre Ürünler',
                labels={'x': 'Segment', 'y': 'Ürün Sayısı'}
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            st.plotly_chart(fig, use_container_width=True)

def rekabet_sekmesi_goster(df: pd.DataFrame, metrikler: Dict):
    """Rekabet analizi sekmesini göster."""
    st.markdown('<h2 class="bolum-baslik">Rekabet Analizi & Pazar Yapısı</h2>', unsafe_allow_html=True)

    st.markdown('<h3 class="alt-bolum-baslik">📊 Rekabet Metrikleri</h3>', unsafe_allow_html=True)

    rekabet_sutunlari = st.columns(4)

    with rekabet_sutunlari[0]:
        hhi = metrikler.get('HHI_Indeksi', 0)
        hhi_durum = "Monopolistik" if hhi > 2500 else "Oligopol" if hhi > 1800 else "Rekabetçi"
        st.metric("HHI İndeksi", f"{hhi:.0f}", hhi_durum)

    with rekabet_sutunlari[1]:
        ilk3 = metrikler.get('Ilk_3_Pay', 0)
        yogunlasma = "Yüksek" if ilk3 > 50 else "Orta" if ilk3 > 30 else "Düşük"
        st.metric("İlk 3 Pay", f"{ilk3:.1f}%", yogunlasma)

    with rekabet_sutunlari[2]:
        ilk5 = metrikler.get('Ilk_5_Pay', 0)
        st.metric("İlk 5 Pay", f"{ilk5:.1f}%")

    with rekabet_sutunlari[3]:
        ilk10_molekul = metrikler.get('Ilk_10_Molekul_Payi', 0)
        st.metric("İlk 10 Molekül Payı", f"{ilk10_molekul:.1f}%")

    gorsellestirici = ProfesyonelGorsellestirici()

    st.markdown('<h3 class="alt-bolum-baslik">🎯 Pazar Hiyerarşisi</h3>', unsafe_allow_html=True)
    gunes_patlama = gorsellestirici.gunes_patlama_hierarsi_grafigi(df)
    if gunes_patlama:
        st.plotly_chart(gunes_patlama, use_container_width=True, config={'displayModeBar': True})

    if 'Sirket' in df.columns:
        st.markdown('<h3 class="alt-bolum-baslik">📊 Şirket Karşılaştırması</h3>', unsafe_allow_html=True)
        
        sirketler = df['Sirket'].value_counts().nlargest(10).index.tolist()
        secili_sirketler = st.multiselect(
            "Karşılaştırılacak şirketleri seçin (max 5)",
            sirketler,
            default=sirketler[:min(3, len(sirketler))]
        )
        
        if len(secili_sirketler) > 0:
            radar = gorsellestirici.radar_karsilastirma_grafigi(df, secili_sirketler)
            if radar:
                st.plotly_chart(radar, use_container_width=True, config={'displayModeBar': True})

def uluslararasi_sekmesi_goster(df: pd.DataFrame, analiz_df: Optional[pd.DataFrame], metrikler: Dict):
    """Uluslararası ürün sekmesini göster."""
    st.markdown('<h2 class="bolum-baslik">🌍 Uluslararası Ürün Analizi</h2>', unsafe_allow_html=True)

    if analiz_df is None:
        st.warning("Uluslararası Ürün analiz verisi mevcut değil.")
        return

    st.markdown('<h3 class="alt-bolum-baslik">📊 Uluslararası Özet</h3>', unsafe_allow_html=True)

    uluslararasi_sutunlari = st.columns(4)

    with uluslararasi_sutunlari[0]:
        uluslararasi_sayisi = metrikler.get('Uluslararasi_Urun_Sayisi', 0)
        toplam_sayisi = metrikler.get('Toplam_Satir', 0)
        uluslararasi_yuzdesi = (uluslararasi_sayisi / toplam_sayisi * 100) if toplam_sayisi > 0 else 0
        st.metric("Uluslararası Ürünler", f"{uluslararasi_sayisi:,}", f"%{uluslararasi_yuzdesi:.1f}")

    with uluslararasi_sutunlari[1]:
        uluslararasi_pay = metrikler.get('Uluslararasi_Urun_Payi', 0)
        st.metric("Pazar Payı", f"%{uluslararasi_pay:.1f}")

    with uluslararasi_sutunlari[2]:
        if 'Ulke_Sayisi' in analiz_df.columns:
            ort_ulke_sayisi = analiz_df['Ulke_Sayisi'].mean()
            st.metric("Ort. Ülke Sayısı", f"{ort_ulke_sayisi:.1f}")

    with uluslararasi_sutunlari[3]:
        if 'Sirket_Sayisi' in analiz_df.columns:
            ort_sirket_sayisi = analiz_df['Sirket_Sayisi'].mean()
            st.metric("Ort. Şirket Sayısı", f"{ort_sirket_sayisi:.1f}")

    st.markdown('<h3 class="alt-bolum-baslik">📋 Uluslararası Ürün Detayları</h3>', unsafe_allow_html=True)

    if len(analiz_df) > 0:
        gosterilecek_sutunlar = []
        
        for sutun in ['Molekul', 'Sirket', 'Uluslararasi', 'Toplam_Satis', 'Sirket_Sayisi',
                      'Ulke_Sayisi', 'Ort_Fiyat', 'Ort_Buyume', 'Segment']:
            if sutun in analiz_df.columns:
                gosterilecek_sutunlar.append(sutun)
        
        gosterim_df = analiz_df[gosterilecek_sutunlar].copy()
        
        if 'Toplam_Satis' in gosterim_df.columns:
            gosterim_df['Toplam_Satis'] = gosterim_df['Toplam_Satis'].apply(
                lambda x: f"${x/1e6:.2f}M" if pd.notnull(x) and x > 0 else "N/A"
            )
        
        if 'Ort_Buyume' in gosterim_df.columns:
            gosterim_df['Ort_Buyume'] = gosterim_df['Ort_Buyume'].apply(
                lambda x: f"%{x:.1f}" if pd.notnull(x) else "N/A"
            )
        
        if 'Ort_Fiyat' in gosterim_df.columns:
            gosterim_df['Ort_Fiyat'] = gosterim_df['Ort_Fiyat'].apply(
                lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A"
            )
        
        st.dataframe(
            gosterim_df,
            use_container_width=True,
            height=400
        )

def tahminleme_sekmesi_goster(df: pd.DataFrame):
    """Tahminleme sekmesini göster."""
    st.markdown('<h2 class="bolum-baslik">🔮 Pazar Tahminleme</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="icgoru-kart bilgi">
        <div class="icgoru-baslik">📊 Tahminleme Metodolojisi</div>
        <div class="icgoru-icerik">
        Tarihsel trendlere dayanarak gelecek pazar değerlerini tahmin etmek için 
        Exponential Smoothing kullanılmaktadır. Güven aralıkları olası sonuç 
        aralığını göstermektedir.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        tahmin_donemleri = st.slider("Tahmin Dönemleri (Yıl)", 1, 5, 2)
        
        if st.button("🔮 Tahmin Oluştur", type="primary", use_container_width=True):
            with st.spinner("Tahmin oluşturuluyor..."):
                analitik = AnalitikMotoru()
                tahmin_df = analitik.pazar_tahmini(df, donemler=tahmin_donemleri)
                
                if tahmin_df is not None:
                    st.session_state.tahmin_verisi = tahmin_df
                    st.success("✅ Tahmin oluşturuldu!")
                else:
                    st.error("Tahmin oluşturulamadı. En az 3 yıl tarihsel veri gereklidir.")

    with col2:
        if 'tahmin_verisi' in st.session_state and st.session_state.tahmin_verisi is not None:
            gorsellestirici = ProfesyonelGorsellestirici()
            tahmin_grafigi = gorsellestirici.tahmin_gorsellestirme(df, st.session_state.tahmin_verisi)
            
            if tahmin_grafigi:
                st.plotly_chart(tahmin_grafigi, use_container_width=True, config={'displayModeBar': True})

    if 'tahmin_verisi' in st.session_state and st.session_state.tahmin_verisi is not None:
        st.markdown('<h3 class="alt-bolum-baslik">📊 Tahmin Detayları</h3>', unsafe_allow_html=True)
        
        tahmin_gosterim = st.session_state.tahmin_verisi.copy()
        tahmin_gosterim['Tahmin'] = tahmin_gosterim['Tahmin'].apply(lambda x: f"${x/1e6:.2f}M")
        tahmin_gosterim['Alt_Sinir'] = tahmin_gosterim['Alt_Sinir'].apply(lambda x: f"${x/1e6:.2f}M")
        tahmin_gosterim['Ust_Sinir'] = tahmin_gosterim['Ust_Sinir'].apply(lambda x: f"${x/1e6:.2f}M")
        
        if 'Son_Yila_Gore_Buyume' in tahmin_gosterim.columns:
            tahmin_gosterim['Son_Yila_Gore_Buyume'] = tahmin_gosterim['Son_Yila_Gore_Buyume'].apply(
                lambda x: f"%{x:.1f}"
            )
        
        st.dataframe(tahmin_gosterim, use_container_width=True)

def anomali_sekmesi_goster(df: pd.DataFrame):
    """Anomali tespiti sekmesini göster."""
    st.markdown('<h2 class="bolum-baslik">⚠️ Anomali Tespiti & İzleme</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="icgoru-kart uyari">
        <div class="icgoru-baslik">🔍 Anomali Tespiti</div>
        <div class="icgoru-icerik">
        Pazardaki aykırı değerleri ve olağandışı kalıpları belirlemek için 
        Isolation Forest algoritması kullanılmaktadır. Yüksek anomali skorlu 
        ürünler özel ilgi gerektirebilir.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🔍 Anomalileri Tespit Et", type="primary", use_container_width=True):
            with st.spinner("Pazar anomalileri analiz ediliyor..."):
                analitik = AnalitikMotoru()
                anomali_df = analitik.anomali_tespiti(df)
                
                if anomali_df is not None:
                    st.session_state.anomali_verisi = anomali_df
                    st.success("✅ Anomali tespiti tamamlandı!")
                else:
                    st.error("Anomali tespiti yapılamadı.")

    with col2:
        if 'anomali_verisi' in st.session_state and st.session_state.anomali_verisi is not None:
            gorsellestirici = ProfesyonelGorsellestirici()
            anomali_grafigi = gorsellestirici.anomali_sacilim_grafigi(st.session_state.anomali_verisi)
            
            if anomali_grafigi:
                st.plotly_chart(anomali_grafigi, use_container_width=True, config={'displayModeBar': True})

    if 'anomali_verisi' in st.session_state and st.session_state.anomali_verisi is not None:
        anomali_df = st.session_state.anomali_verisi
        
        st.markdown('<h3 class="alt-bolum-baslik">⚠️ Yüksek-Riskli Ürünler</h3>', unsafe_allow_html=True)
        
        if 'Anomali_Kategorisi' in anomali_df.columns:
            yuksek_risk = anomali_df[anomali_df['Anomali_Kategorisi'] == 'Yüksek Risk']
            
            if len(yuksek_risk) > 0:
                gosterilecek_sutunlar = ['Molekul', 'Sirket', 'Anomali_Skoru'] if 'Molekul' in yuksek_risk.columns and 'Sirket' in yuksek_risk.columns else ['Anomali_Skoru']
                
                satis_sutunlari = [sutun for sutun in yuksek_risk.columns if 'Satış_' in sutun]
                if satis_sutunlari:
                    gosterilecek_sutunlar.append(satis_sutunlari[-1])
                
                st.dataframe(
                    yuksek_risk[gosterilecek_sutunlar].sort_values('Anomali_Skoru').head(20),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("Yüksek-riskli anomali tespit edilmedi.")

def raporlama_sekmesi_goster(df: pd.DataFrame, metrikler: Dict, icgoruler: List[Dict], analiz_df: Optional[pd.DataFrame]):
    """Raporlama sekmesini göster."""
    st.markdown('<h2 class="bolum-baslik">📑 Raporlama & Dışa Aktarım</h2>', unsafe_allow_html=True)

    st.markdown('<h3 class="alt-bolum-baslik">📊 Rapor Türleri</h3>', unsafe_allow_html=True)

    rapor_secenekleri = ['Excel Kapsamlı Rapor', 'HTML İnteraktif Rapor', 'CSV Ham Veri']
    if REPORTLAB_MEVCUT:
        rapor_secenekleri.append('PDF Rapor')
    
    rapor_turu = st.radio(
        "Rapor Türü Seçin",
        rapor_secenekleri,
        horizontal=True
    )

    st.markdown('<h3 class="alt-bolum-baslik">🛠️ Rapor Oluştur</h3>', unsafe_allow_html=True)

    rapor_sutunlari = st.columns(4)

    with rapor_sutunlari[0]:
        if st.button("📈 Excel Raporu", use_container_width=True):
            with st.spinner("Excel raporu oluşturuluyor..."):
                uretici = RaporUretici()
                excel_verisi = uretici.excel_raporu_uret(df, metrikler, icgoruler)
                
                zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                st.download_button(
                    label="⬇️ Excel'i İndir",
                    data=excel_verisi,
                    file_name=f"pharma_rapor_{zaman_damgasi}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

    with rapor_sutunlari[1]:
        if st.button("🌐 HTML Raporu", use_container_width=True):
            with st.spinner("HTML raporu oluşturuluyor..."):
                uretici = RaporUretici()
                html_verisi = uretici.html_raporu_uret(df, metrikler, icgoruler)
                
                zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                st.download_button(
                    label="⬇️ HTML'i İndir",
                    data=html_verisi.encode('utf-8'),
                    file_name=f"pharma_rapor_{zaman_damgasi}.html",
                    mime="text/html",
                    use_container_width=True
                )

    with rapor_sutunlari[2]:
        if st.button("💾 CSV Dışa Aktar", use_container_width=True):
            zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_verisi = df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ CSV'yi İndir",
                data=csv_verisi,
                file_name=f"pharma_veri_{zaman_damgasi}.csv",
                mime="text/csv",
                use_container_width=True
            )

    with rapor_sutunlari[3]:
        if st.button("🔄 Analizi Sıfırla", use_container_width=True):
            for anahtar in list(st.session_state.keys()):
                if anahtar != 'app_started':
                    del st.session_state[anahtar]
            st.rerun()

    st.markdown('<h3 class="alt-bolum-baslik">📈 Hızlı İstatistikler</h3>', unsafe_allow_html=True)

    istatistik_sutunlari = st.columns(4)

    with istatistik_sutunlari[0]:
        st.metric("Toplam Satır", f"{len(df):,}")

    with istatistik_sutunlari[1]:
        st.metric("Toplam Sütun", len(df.columns))

    with istatistik_sutunlari[2]:
        hafiza_kullanimi = df.memory_usage(deep=True).sum()/1024**2
        st.metric("Hafıza Kullanımı", f"{hafiza_kullanimi:.1f} MB")

    with istatistik_sutunlari[3]:
        uluslararasi_sayisi = metrikler.get('Uluslararasi_Urun_Sayisi', 0)
        st.metric("Uluslararası Ürünler", uluslararasi_sayisi)

# ================================================
# 9. UYGULAMA GİRİŞ NOKTASI
# ================================================

if __name__ == "__main__":
    try:
        gc.enable()
        st.session_state.setdefault('app_started', True)
        ana()
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        st.error("Detaylı hata bilgisi:")
        st.code(traceback.format_exc())
        
        if st.button("🔄 Uygulamayı Yeniden Yükle", use_container_width=True):
            for anahtar in list(st.session_state.keys()):
                del st.session_state[anahtar]
            st.rerun()


