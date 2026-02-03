import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import zscore, percentileofscore
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFİGÜRASYON ====================
st.set_page_config(
    page_title="İleri Seviye Finansal Analiz Platformu",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# ==================== ÖZEL CSS ====================
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    .sidebar .sidebar-content { background-color: #1e293b; }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    
    .metric-title {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 5px;
    }
    
    .metric-change {
        font-size: 13px;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    
    .positive { background-color: #064e3b; color: #34d399; }
    .negative { background-color: #7f1d1d; color: #f87171; }
    .neutral { background-color: #374151; color: #9ca3af; }
    
    .section-header {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        padding: 18px;
        border-radius: 10px;
        margin: 25px 0;
        color: white;
        font-weight: 800;
        font-size: 18px;
        box-shadow: 0 4px 6px rgba(30, 64, 175, 0.3);
    }
    
    .insight-box {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
        border: 1px solid #334155;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .insight-title {
        color: #60a5fa;
        font-weight: 700;
        margin-bottom: 8px;
        font-size: 15px;
    }
    
    .insight-content {
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.5;
    }
    
    .warning-alert {
        background: linear-gradient(135deg, #7c2d12 0%, #9a3412 100%);
        border-left: 5px solid #f97316;
        padding: 18px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .success-alert {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border-left: 5px solid #10b981;
        padding: 18px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .info-alert {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        border-left: 5px solid #60a5fa;
        padding: 18px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .tab-container {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 25px;
        margin-top: 20px;
        border: 1px solid #334155;
    }
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    .stSelectbox div[data-baseweb="select"],
    .stMultiSelect div[data-baseweb="select"] {
        background-color: #0f172a;
        border-color: #334155;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(30, 64, 175, 0.4);
    }
    
    .score-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        font-weight: 800;
        font-size: 14px;
        margin-right: 10px;
    }
    
    .score-excellent { background-color: #065f46; color: #34d399; }
    .score-good { background-color: #2563eb; color: #60a5fa; }
    .score-fair { background-color: #ca8a04; color: #fbbf24; }
    .score-poor { background-color: #dc2626; color: #f87171; }
    
    .risk-indicator {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .risk-high { background-color: #ef4444; box-shadow: 0 0 8px #ef4444; }
    .risk-medium { background-color: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
    .risk-low { background-color: #10b981; box-shadow: 0 0 4px #10b981; }
    
    .growth-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
    }
    
    .growth-high { background-color: #065f46; color: #34d399; }
    .growth-moderate { background-color: #ca8a04; color: #fbbf24; }
    .growth-low { background-color: #7f1d1d; color: #f87171; }
</style>
""", unsafe_allow_html=True)

# ==================== VERİ ÜRETİM FONKSİYONLARI ====================
@st.cache_resource
def generate_financial_data():
    """Büyük ölçekli finansal veri seti oluştur"""
    np.random.seed(42)
    
    # Temel yapılar
    companies = [
        'Teknoloji AŞ', 'Finans Bank', 'Enerji Holding', 'İnşaat Grup',
        'Sağlık Şirketi', 'Perakende Zinciri', 'Otomotiv Üreticisi',
        'Telekom Şirketi', 'Gıda Üreticisi', 'Kimya Sanayi'
    ]
    
    sectors = {
        'Teknoloji': ['Yazılım', 'Donanım', 'Bulut', 'Siber Güvenlik'],
        'Finans': ['Bankacılık', 'Sigorta', 'Yatırım', 'FinTech'],
        'Enerji': ['Petrol', 'Doğalgaz', 'Yenilenebilir', 'Kömür'],
        'İnşaat': ['Konut', 'Ticari', 'Altyapı', 'Endüstriyel'],
        'Sağlık': ['İlaç', 'Medikal Cihaz', 'Hastane', 'Biyoteknoloji']
    }
    
    regions = ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya', 'Kocaeli']
    
    # 500,000 satır veri oluştur
    n_rows = 500000
    data = []
    
    for i in range(n_rows):
        # Şirket ve sektör ataması
        company = np.random.choice(companies)
        sector_main = [k for k, v in sectors.items() if any(company.lower() in x.lower() for x in [k])]
        sector_main = sector_main[0] if sector_main else np.random.choice(list(sectors.keys()))
        sector_sub = np.random.choice(sectors[sector_main])
        
        # Temel finansal metrikler
        revenue_base = np.random.lognormal(12, 1.5)
        profit_margin = np.random.beta(5, 2) * 0.3  # 0-30% arası
        growth_rate = np.random.normal(0.15, 0.1)  # Ortalama %15 büyüme
        
        # Firma büyüklüğüne göre ayarlamalar
        company_multiplier = {
            'Teknoloji AŞ': 1.8,
            'Finans Bank': 2.2,
            'Enerji Holding': 2.5,
            'İnşaat Grup': 1.5,
            'Sağlık Şirketi': 1.7,
            'Perakende Zinciri': 1.3,
            'Otomotiv Üreticisi': 2.0,
            'Telekom Şirketi': 1.9,
            'Gıda Üreticisi': 1.2,
            'Kimya Sanayi': 1.4
        }.get(company, 1.0)
        
        # Mevsimsellik etkisi
        quarter = (i % 4) + 1
        seasonal_factor = 1 + (quarter - 2.5) * 0.1  # Q2 ve Q3'te daha yüksek
        
        # Bölgesel etki
        region_factor = 1 + (regions.index(np.random.choice(regions)) * 0.05)
        
        # Temel finansal veriler
        revenue_q1 = revenue_base * company_multiplier * seasonal_factor * region_factor * np.random.uniform(0.9, 1.1)
        revenue_q2 = revenue_q1 * (1 + growth_rate * np.random.uniform(0.8, 1.2))
        revenue_q3 = revenue_q2 * (1 + growth_rate * np.random.uniform(0.8, 1.2))
        revenue_q4 = revenue_q3 * (1 + growth_rate * np.random.uniform(0.8, 1.2))
        
        # Karlılık metrikleri
        gross_profit_q1 = revenue_q1 * profit_margin * np.random.uniform(0.9, 1.1)
        gross_profit_q2 = revenue_q2 * profit_margin * np.random.uniform(0.9, 1.1)
        gross_profit_q3 = revenue_q3 * profit_margin * np.random.uniform(0.9, 1.1)
        gross_profit_q4 = revenue_q4 * profit_margin * np.random.uniform(0.9, 1.1)
        
        # Operasyonel metrikler
        operating_expense_ratio = np.random.beta(3, 3)  # 0-1 arası
        ebitda_margin = profit_margin * (1 - operating_expense_ratio) * np.random.uniform(0.7, 0.9)
        
        # Verimlilik metrikleri
        asset_turnover = np.random.lognormal(0, 0.3)
        inventory_turnover = np.random.lognormal(2.5, 0.4)
        
        # Risk metrikleri
        debt_ratio = np.random.beta(2, 3)
        current_ratio = np.random.lognormal(0.7, 0.2)
        
        # Piyasa metrikleri
        pe_ratio = np.random.lognormal(2.5, 0.5)
        market_cap = revenue_q4 * pe_ratio * np.random.uniform(0.8, 1.2)
        
        # Satır verisi oluştur
        row = {
            # Tanımlayıcı bilgiler
            'Şirket_ID': f'CMP{str(i%1000).zfill(4)}',
            'Şirket_Adı': company,
            'Ana_Sektör': sector_main,
            'Alt_Sektör': sector_sub,
            'Bölge': np.random.choice(regions),
            'Rapor_Dönemi': f'2024-Q{quarter}',
            
            # Gelir tablosu metrikleri
            'Ciro': revenue_q4,
            'Ciro_Q1': revenue_q1,
            'Ciro_Q2': revenue_q2,
            'Ciro_Q3': revenue_q3,
            'Ciro_Büyüme': growth_rate * 100,
            'Ciro_YoY_Büyüme': np.random.normal(18, 8),
            'Ciro_QoQ_Büyüme': np.random.normal(5, 3),
            
            # Karlılık metrikleri
            'Brüt_Kar': gross_profit_q4,
            'Brüt_Kar_Marjı': (gross_profit_q4 / revenue_q4 * 100) if revenue_q4 > 0 else 0,
            'EBITDA': revenue_q4 * ebitda_margin,
            'EBITDA_Marjı': ebitda_margin * 100,
            'Net_Kar': gross_profit_q4 * (1 - operating_expense_ratio) * np.random.uniform(0.6, 0.8),
            'Net_Kar_Marjı': (gross_profit_q4 * (1 - operating_expense_ratio) * np.random.uniform(0.6, 0.8) / revenue_q4 * 100) if revenue_q4 > 0 else 0,
            
            # Operasyonel metrikler
            'Operasyonel_Gider_Oranı': operating_expense_ratio * 100,
            'Varlık_Devir_Hızı': asset_turnover,
            'Stok_Devir_Hızı': inventory_turnover,
            'Çalışma_Sermayesi': revenue_q4 * np.random.uniform(0.1, 0.3),
            'CFO': gross_profit_q4 * np.random.uniform(0.7, 0.9),
            'CFI': -revenue_q4 * np.random.uniform(0.05, 0.15),
            'CFF': revenue_q4 * np.random.uniform(-0.1, 0.1),
            
            # Finansal yapı metrikleri
            'Toplam_Varlıklar': revenue_q4 / asset_turnover,
            'Toplam_Borç': (revenue_q4 / asset_turnover) * debt_ratio,
            'Özkaynak': (revenue_q4 / asset_turnover) * (1 - debt_ratio),
            'Borç_Oranı': debt_ratio * 100,
            'Cari_Oran': current_ratio,
            'Faiz_Karşılama': np.random.lognormal(1.5, 0.5),
            
            # Piyasa metrikleri
            'Piyasa_Değeri': market_cap,
            'F/K_Oranı': pe_ratio,
            'PD/DD_Oranı': np.random.lognormal(1.2, 0.3),
            'Beta_Katsayısı': np.random.beta(2, 2) * 2,
            'Getiri_Oranı': np.random.normal(12, 8),
            'Getiri_Volatilitesi': np.random.lognormal(2, 0.3),
            
            # Ek metrikler
            'Çalışan_Sayısı': int(revenue_q4 / np.random.lognormal(100, 20)),
            'ARGE_Harcaması': revenue_q4 * np.random.beta(2, 8) * 100,
            'Pazar_Payı': np.random.beta(2, 5) * 100,
            'Müşteri_Memnuniyeti': np.random.beta(7, 3) * 100,
            'Çevresel_Skor': np.random.beta(6, 4) * 100,
            'Sosyal_Skor': np.random.beta(7, 3) * 100,
            'Yönetişim_Skoru': np.random.beta(6, 4) * 100,
            
            # Risk skorları
            'Likidite_Risk_Skoru': np.random.randint(1, 100),
            'Kredi_Risk_Skoru': np.random.randint(1, 100),
            'Piyasa_Risk_Skoru': np.random.randint(1, 100),
            'Operasyonel_Risk_Skoru': np.random.randint(1, 100),
            
            # Trendler
            'Ciro_Trend': np.random.choice(['Yükselen', 'Düşen', 'Sabit', 'Dalgalı']),
            'Kar_Trend': np.random.choice(['Yükselen', 'Düşen', 'Sabit', 'Dalgalı']),
            'Pazar_Trend': np.random.choice(['Genişleyen', 'Daralan', 'Olgun', 'Yenilikçi'])
        }
        
        data.append(row)
        
        # Performans optimizasyonu için ara belleği temizle
        if i % 50000 == 0 and i > 0:
            gc.collect()
    
    df = pd.DataFrame(data)
    
    # Veri kalitesi iyileştirmeleri
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Bazı değerleri eksik yap (gerçekçilik için)
    for col in numeric_cols:
        if np.random.random() < 0.05:  # %5 eksik veri
            mask = np.random.random(len(df)) < 0.1
            df.loc[mask, col] = np.nan
    
    # Ayırıcı formatı ekle (virgül/ nokta karışımı)
    for col in ['Ciro', 'Brüt_Kar', 'EBITDA', 'Net_Kar', 'Piyasa_Değeri']:
        df[col] = df[col].apply(
            lambda x: f"{x:,.2f}".replace('.', ',').replace(',', 'X').replace('.', ',').replace('X', '.') 
            if np.random.random() < 0.4 else x
        )
    
    return df

def convert_numeric_columns(df):
    """Sayısal kolonları uygun formata çevir"""
    numeric_cols = [
        'Ciro', 'Ciro_Q1', 'Ciro_Q2', 'Ciro_Q3', 'Ciro_Büyüme', 'Ciro_YoY_Büyüme', 'Ciro_QoQ_Büyüme',
        'Brüt_Kar', 'Brüt_Kar_Marjı', 'EBITDA', 'EBITDA_Marjı', 'Net_Kar', 'Net_Kar_Marjı',
        'Operasyonel_Gider_Oranı', 'Varlık_Devir_Hızı', 'Stok_Devir_Hızı', 'Çalışma_Sermayesi',
        'CFO', 'CFI', 'CFF', 'Toplam_Varlıklar', 'Toplam_Borç', 'Özkaynak', 'Borç_Oranı',
        'Cari_Oran', 'Faiz_Karşılama', 'Piyasa_Değeri', 'F/K_Oranı', 'PD/DD_Oranı',
        'Beta_Katsayısı', 'Getiri_Oranı', 'Getiri_Volatilitesi', 'ARGE_Harcaması',
        'Pazar_Payı', 'Müşteri_Memnuniyeti', 'Çevresel_Skor', 'Sosyal_Skor', 'Yönetişim_Skoru'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

@st.cache_data
def load_data():
    """Veriyi yükle ve ön işle"""
    with st.spinner('📊 500,000+ satır finansal veri yükleniyor...'):
        df = generate_financial_data()
        df = convert_numeric_columns(df)
    return df

# ==================== ANALİTİK FONKSİYONLAR ====================
def calculate_financial_ratios(df):
    """Finansal oranları hesapla"""
    ratios = {}
    
    # Temel oranlar
    ratios['cari_oran'] = df['Cari_Oran'].mean()
    ratios['borc_oran'] = df['Borç_Oranı'].mean()
    ratios['fk_oran'] = df['F/K_Oranı'].mean()
    ratios['roa'] = (df['Net_Kar'] / df['Toplam_Varlıklar']).mean() * 100
    ratios['roe'] = (df['Net_Kar'] / df['Özkaynak']).mean() * 100
    ratios['ros'] = df['Net_Kar_Marjı'].mean()
    
    # Büyüme oranları
    ratios['ciro_buyume'] = df['Ciro_Büyüme'].mean()
    ratios['kar_buyume'] = ((df['Net_Kar'].mean() - df['Net_Kar'].median()) / df['Net_Kar'].median() * 100 
                           if df['Net_Kar'].median() > 0 else 0)
    
    # Verimlilik oranları
    ratios['varlik_devir'] = df['Varlık_Devir_Hızı'].mean()
    ratios['stok_devir'] = df['Stok_Devir_Hızı'].mean()
    
    # Risk ölçümleri
    ratios['volatilite'] = df['Getiri_Volatilitesi'].mean()
    ratios['beta'] = df['Beta_Katsayısı'].mean()
    
    return ratios

def perform_dupont_analysis(df):
    """DuPont analizi yap"""
    results = {}
    
    # DuPont bileşenleri
    results['net_kar_marji'] = df['Net_Kar_Marjı'].mean()
    results['varlik_devir_hizi'] = df['Varlık_Devir_Hızı'].mean()
    results['ozkaynak_carpani'] = (df['Toplam_Varlıklar'] / df['Özkaynak']).mean()
    
    # ROE hesaplama
    results['roe_dupont'] = results['net_kar_marji'] * results['varlik_devir_hizi'] * results['ozkaynak_carpani']
    
    # Sektörel karşılaştırma
    sector_roe = df.groupby('Ana_Sektör').apply(
        lambda x: (x['Net_Kar'] / x['Özkaynak']).mean() * 100
    ).to_dict()
    
    results['sector_roe'] = sector_roe
    
    return results

def calculate_altman_z_score(df):
    """Altman Z-Score hesapla"""
    results = []
    
    for _, row in df.iterrows():
        try:
            # Altman Z-Score formülü
            X1 = row['Çalışma_Sermayesi'] / row['Toplam_Varlıklar'] if row['Toplam_Varlıklar'] > 0 else 0
            X2 = row['Net_Kar'] / row['Toplam_Varlıklar'] if row['Toplam_Varlıklar'] > 0 else 0
            X3 = row['EBITDA'] / row['Toplam_Varlıklar'] if row['Toplam_Varlıklar'] > 0 else 0
            X4 = row['Özkaynak'] / row['Toplam_Borç'] if row['Toplam_Borç'] > 0 else 0
            X5 = row['Ciro'] / row['Toplam_Varlıklar'] if row['Toplam_Varlıklar'] > 0 else 0
            
            Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
            
            # Risk kategorisi
            if Z > 2.99:
                risk = 'Düşük Risk'
            elif Z > 1.81:
                risk = 'Gri Bölge'
            else:
                risk = 'Yüksek Risk'
            
            results.append({
                'Şirket': row['Şirket_Adı'],
                'Z_Score': Z,
                'Risk_Kategorisi': risk,
                'X1': X1,
                'X2': X2,
                'X3': X3,
                'X4': X4,
                'X5': X5
            })
        except:
            continue
    
    return pd.DataFrame(results)

def perform_regression_analysis(df, target='Net_Kar'):
    """Regresyon analizi yap"""
    # Özellik seçimi
    features = ['Ciro', 'Brüt_Kar_Marjı', 'Varlık_Devir_Hızı', 'Borç_Oranı', 'ARGE_Harcaması']
    
    # Eksik verileri temizle
    analysis_df = df[features + [target]].dropna()
    
    if len(analysis_df) < 10:
        return None
    
    # Korelasyon analizi
    correlation = analysis_df.corr()[target].sort_values(ascending=False)
    
    # Basit lineer regresyon (her özellik için)
    from scipy import stats
    regression_results = {}
    
    for feature in features:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            analysis_df[feature], analysis_df[target]
        )
        regression_results[feature] = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'importance': abs(r_value)
        }
    
    return {
        'correlation': correlation,
        'regression': regression_results,
        'sample_size': len(analysis_df)
    }

def calculate_risk_metrics(df):
    """Risk metriklerini hesapla"""
    risk_data = {}
    
    # Volatilite tabanlı risk
    risk_data['price_volatility'] = df['Getiri_Volatilitesi'].mean()
    risk_data['beta_risk'] = df['Beta_Katsayısı'].mean()
    
    # Finansal risk
    risk_data['leverage_risk'] = df['Borç_Oranı'].mean()
    risk_data['liquidity_risk'] = (df[df['Cari_Oran'] < 1].shape[0] / len(df)) * 100
    
    # Kredi riski
    altman_scores = calculate_altman_z_score(df)
    if not altman_scores.empty:
        risk_data['bankruptcy_risk'] = (altman_scores[altman_scores['Risk_Kategorisi'] == 'Yüksek Risk'].shape[0] / 
                                       len(altman_scores)) * 100
    
    # Konsantrasyon riski
    top_3_market_share = df.groupby('Ana_Sektör')['Pazar_Payı'].sum().nlargest(3).sum()
    total_market_share = df['Pazar_Payı'].sum()
    risk_data['concentration_risk'] = (top_3_market_share / total_market_share * 100) if total_market_share > 0 else 0
    
    return risk_data

def generate_advanced_insights(df):
    """İleri seviye içgörüler oluştur"""
    insights = []
    
    # 1. Büyüme kalitesi analizi
    high_growth = df[df['Ciro_Büyüme'] > 20]
    profitable_growth = high_growth[high_growth['Net_Kar_Marjı'] > 10]
    
    growth_quality = (len(profitable_growth) / len(high_growth) * 100) if len(high_growth) > 0 else 0
    insights.append({
        'title': 'Büyüme Kalitesi Analizi',
        'content': f'{growth_quality:.1f}% yüksek büyüme gösteren şirket aynı zamanda karlı',
        'metric': growth_quality,
        'threshold': 50
    })
    
    # 2. Verimlilik trendi
    sector_efficiency = df.groupby('Ana_Sektör')['Varlık_Devir_Hızı'].mean().sort_values(ascending=False)
    top_sector = sector_efficiency.index[0] if len(sector_efficiency) > 0 else 'N/A'
    insights.append({
        'title': 'En Verimli Sektör',
        'content': f'{top_sector} sektörü en yüksek varlık devir hızına sahip',
        'metric': sector_efficiency.iloc[0] if len(sector_efficiency) > 0 else 0
    })
    
    # 3. Risk-Getiri optimizasyonu
    df['risk_adjusted_return'] = df['Getiri_Oranı'] / df['Getiri_Volatilitesi']
    top_risk_adjusted = df.nlargest(5, 'risk_adjusted_return')[['Şirket_Adı', 'risk_adjusted_return']]
    
    insights.append({
        'title': 'Risk-Getiri Optimizasyonu',
        'content': 'En iyi risk-getiri oranına sahip 5 şirket tespit edildi',
        'data': top_risk_adjusted.to_dict('records')
    })
    
    # 4. ESG performansı
    esg_score = df[['Çevresel_Skor', 'Sosyal_Skor', 'Yönetişim_Skoru']].mean().mean()
    insights.append({
        'title': 'ESG Performansı',
        'content': f'Ortalama ESG skoru: {esg_score:.1f}/100',
        'metric': esg_score
    })
    
    # 5. İnovasyon yatırımları
    high_rd = df[df['ARGE_Harcaması'] > df['ARGE_Harcaması'].quantile(0.75)]
    rd_growth_correlation = high_rd['Ciro_Büyüme'].corr(high_rd['ARGE_Harcaması'])
    
    insights.append({
        'title': 'ARGE Yatırımı & Büyüme',
        'content': f'Yüksek ARGE harcaması ile büyüme korelasyonu: {rd_growth_correlation:.3f}',
        'metric': rd_growth_correlation
    })
    
    return insights

# ==================== GÖRSELLEŞTİRME FONKSİYONLARI ====================
def create_financial_health_chart(df):
    """Finansal sağlık radar chart'ı oluştur"""
    # Ortalama değerleri hesapla
    metrics = {
        'Karlılık': df['Net_Kar_Marjı'].mean(),
        'Likidite': df['Cari_Oran'].mean() * 10,  # Ölçeklendirme
        'Verimlilik': df['Varlık_Devir_Hızı'].mean() * 10,
        'Büyüme': df['Ciro_Büyüme'].mean(),
        'Risk Yönetimi': 100 - df['Borç_Oranı'].mean(),
        'ESG': df[['Çevresel_Skor', 'Sosyal_Skor', 'Yönetişim_Skoru']].mean().mean()
    }
    
    fig = go.Figure(data=go.Scatterpolar(
        r=list(metrics.values()),
        theta=list(metrics.keys()),
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='rgb(59, 130, 246)', width=2),
        hoverinfo='text',
        text=[f'{k}: {v:.1f}' for k, v in metrics.items()]
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title='Finansal Sağlık Radar Chart',
        height=500,
        paper_bgcolor='#1e293b',
        plot_bgcolor='#1e293b',
        font_color='white'
    )
    
    return fig

def create_correlation_heatmap(df):
    """Korelasyon heatmap oluştur"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    important_cols = [col for col in numeric_cols if col in [
        'Ciro', 'Net_Kar', 'Brüt_Kar_Marjı', 'EBITDA_Marjı', 'Varlık_Devir_Hızı',
        'Borç_Oranı', 'Cari_Oran', 'F/K_Oranı', 'Getiri_Oranı', 'ARGE_Harcaması'
    ]]
    
    if len(important_cols) < 3:
        return None
    
    corr_matrix = df[important_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Finansal Metrikler Korelasyon Matrisi',
        height=600,
        paper_bgcolor='#1e293b',
        plot_bgcolor='#1e293b',
        font_color='white'
    )
    
    return fig

def create_growth_comparison_chart(df):
    """Büyüme karşılaştırma chart'ı oluştur"""
    sector_growth = df.groupby('Ana_Sektör').agg({
        'Ciro_Büyüme': 'mean',
        'Net_Kar_Marjı': 'mean',
        'Şirket_Adı': 'count'
    }).reset_index()
    
    sector_growth.columns = ['Sektör', 'Ortalama_Büyüme', 'Ortalama_Karlılık', 'Şirket_Sayısı']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sector_growth['Sektör'],
        y=sector_growth['Ortalama_Büyüme'],
        name='Büyüme (%)',
        marker_color='#3b82f6',
        text=sector_growth['Ortalama_Büyüme'].round(1),
        textposition='auto'
    ))
    
    fig.add_trace(go.Scatter(
        x=sector_growth['Sektör'],
        y=sector_growth['Ortalama_Karlılık'],
        name='Karlılık (%)',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='#10b981', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title='Sektörel Büyüme & Karlılık Karşılaştırması',
        yaxis=dict(title='Büyüme (%)'),
        yaxis2=dict(
            title='Karlılık (%)',
            overlaying='y',
            side='right'
        ),
        height=500,
        paper_bgcolor='#1e293b',
        plot_bgcolor='#1e293b',
        font_color='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_risk_return_scatter(df):
    """Risk-Getiri dağılım grafiği oluştur"""
    fig = px.scatter(
        df,
        x='Getiri_Volatilitesi',
        y='Getiri_Oranı',
        size='Piyasa_Değeri',
        color='Ana_Sektör',
        hover_name='Şirket_Adı',
        hover_data=['Net_Kar_Marjı', 'F/K_Oranı'],
        title='Risk-Getiri Dağılımı',
        labels={
            'Getiri_Volatilitesi': 'Risk (Volatilite)',
            'Getiri_Oranı': 'Getiri (%)'
        }
    )
    
    # Efektif sınır çizgisi ekle
    fig.add_trace(go.Scatter(
        x=[df['Getiri_Volatilitesi'].min(), df['Getiri_Volatilitesi'].max()],
        y=[df['Getiri_Oranı'].min(), df['Getiri_Oranı'].max()],
        mode='lines',
        name='Efektif Sınır',
        line=dict(color='#f59e0b', dash='dash', width=2),
        showlegend=True
    ))
    
    fig.update_layout(
        height=600,
        paper_bgcolor='#1e293b',
        plot_bgcolor='#1e293b',
        font_color='white'
    )
    
    return fig

def create_valuation_analysis(df):
    """Değerleme analizi grafiği oluştur"""
    # Sektörel F/K oranları
    sector_pe = df.groupby('Ana_Sektör')['F/K_Oranı'].agg(['mean', 'median', 'std']).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sektörel F/K Dağılımı', 'F/K vs Karlılık', 'Piyasa Değeri Dağılımı', 'Değerleme İndikatörleri'),
        specs=[[{'type': 'bar'}, {'type': 'scatter'}],
               [{'type': 'histogram'}, {'type': 'box'}]]
    )
    
    # 1. Sektörel F/K
    fig.add_trace(
        go.Bar(
            x=sector_pe['Ana_Sektör'],
            y=sector_pe['mean'],
            error_y=dict(type='data', array=sector_pe['std']),
            name='Ortalama F/K',
            marker_color='#3b82f6'
        ),
        row=1, col=1
    )
    
    # 2. F/K vs Karlılık
    fig.add_trace(
        go.Scatter(
            x=df['F/K_Oranı'],
            y=df['Net_Kar_Marjı'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['Ciro_Büyüme'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Büyüme (%)")
            ),
            name='F/K vs Karlılık',
            hovertext=df['Şirket_Adı']
        ),
        row=1, col=2
    )
    
    # 3. Piyasa değeri dağılımı
    fig.add_trace(
        go.Histogram(
            x=df['Piyasa_Değeri'],
            nbinsx=50,
            name='Piyasa Değeri',
            marker_color='#10b981'
        ),
        row=2, col=1
    )
    
    # 4. Değerleme indikatörleri
    fig.add_trace(
        go.Box(
            y=df['PD/DD_Oranı'],
            name='PD/DD',
            boxmean=True,
            marker_color='#8b5cf6'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        paper_bgcolor='#1e293b',
        plot_bgcolor='#1e293b',
        font_color='white'
    )
    
    return fig

# ==================== UI KOMPONENTLERİ ====================
def create_metric_card(title, value, change=None, change_label=None, size='medium'):
    """Metrik kartı oluştur"""
    if size == 'large':
        value_size = '32px'
        title_size = '16px'
    else:
        value_size = '26px'
        title_size = '14px'
    
    if change is not None:
        change_class = "positive" if change > 0 else "negative" if change < 0 else "neutral"
        change_display = f'''
        <div class="metric-change {change_class}">
            {change:+.1f}% {change_label if change_label else ""}
        </div>
        '''
    else:
        change_display = ''
    
    return f'''
    <div class="metric-card">
        <div class="metric-title" style="font-size: {title_size};">{title}</div>
        <div class="metric-value" style="font-size: {value_size};">{value}</div>
        {change_display}
    </div>
    '''

def display_insight_box(title, content, metric=None, threshold=None):
    """İçgörü kutusu göster"""
    if metric is not None and threshold is not None:
        if metric > threshold:
            icon = "✅"
            color = "#10b981"
        else:
            icon = "⚠️"
            color = "#f59e0b"
    else:
        icon = "💡"
        color = "#60a5fa"
    
    return f'''
    <div class="insight-box">
        <div class="insight-title">{icon} {title}</div>
        <div class="insight-content">{content}</div>
        {f'<div style="margin-top: 10px; color: {color}; font-weight: 600;">Metrik: {metric:.1f}</div>' if metric is not None else ''}
    </div>
    '''

def create_risk_indicator(risk_level, label):
    """Risk göstergesi oluştur"""
    if risk_level == 'high':
        risk_class = 'risk-high'
        text_color = '#ef4444'
    elif risk_level == 'medium':
        risk_class = 'risk-medium'
        text_color = '#f59e0b'
    else:
        risk_class = 'risk-low'
        text_color = '#10b981'
    
    return f'''
    <div style="display: flex; align-items: center; margin: 5px 0;">
        <span class="risk-indicator {risk_class}"></span>
        <span style="color: {text_color}; font-weight: 600;">{label}</span>
    </div>
    '''

# ==================== ANA UYGULAMA ====================
def main():
    st.title("📈 İleri Seviye Finansal Analiz Platformu")
    st.markdown("### 500,000+ Şirket Verisi | Gerçek Zamanlı Analitik | Profesyonel İçgörüler")
    
    # Veriyi yükle
    df = load_data()
    
    # Sidebar filtreleri
    st.sidebar.markdown("## 🔍 Filtre Kontrolleri")
    
    # Sektör filtresi
    sectors = ['Tümü'] + sorted(df['Ana_Sektör'].unique().tolist())
    selected_sector = st.sidebar.selectbox("Ana Sektör", sectors)
    
    # Bölge filtresi
    regions = ['Tümü'] + sorted(df['Bölge'].unique().tolist())
    selected_region = st.sidebar.selectbox("Bölge", regions)
    
    # Büyüme filtresi
    growth_filter = st.sidebar.slider(
        "Minimum Büyüme Oranı (%)",
        min_value=-50, max_value=100, value=0, step=5
    )
    
    # Karlılık filtresi
    profitability_filter = st.sidebar.slider(
        "Minimum Net Kar Marjı (%)",
        min_value=-20, max_value=50, value=0, step=2
    )
    
    # Risk filtresi
    risk_filter = st.sidebar.selectbox(
        "Risk Seviyesi",
        ['Tümü', 'Düşük Risk', 'Orta Risk', 'Yüksek Risk']
    )
    
    # Veriyi filtrele
    filtered_df = df.copy()
    
    if selected_sector != 'Tümü':
        filtered_df = filtered_df[filtered_df['Ana_Sektör'] == selected_sector]
    
    if selected_region != 'Tümü':
        filtered_df = filtered_df[filtered_df['Bölge'] == selected_region]
    
    filtered_df = filtered_df[filtered_df['Ciro_Büyüme'] >= growth_filter]
    filtered_df = filtered_df[filtered_df['Net_Kar_Marjı'] >= profitability_filter]
    
    # Sidebar istatistikleri
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Filtre İstatistikleri")
    
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        st.metric("Şirket Sayısı", f"{filtered_df['Şirket_Adı'].nunique():,}")
    with col_s2:
        st.metric("Ort. Büyüme", f"{filtered_df['Ciro_Büyüme'].mean():.1f}%")
    
    st.sidebar.metric("Toplam Ciro", f"${filtered_df['Ciro'].sum():,.0f}")
    
    # Reset butonu
    if st.sidebar.button("🔄 Filtreleri Sıfırla"):
        st.rerun()
    
    # Ana içerik
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Genel Bakış", 
        "📈 Performans", 
        "⚖️ Risk Analizi",
        "💰 Değerleme",
        "🔍 Detaylı Analiz",
        "📊 Veri Keşfi"
    ])
    
    with tab1:
        st.markdown('<div class="section-header">📊 GENEL BAKIŞ & TEMEL METRİKLER</div>', unsafe_allow_html=True)
        
        # Üst metrik satırı
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = filtered_df['Ciro'].sum()
            revenue_growth = filtered_df['Ciro_Büyüme'].mean()
            st.markdown(create_metric_card(
                "Toplam Ciro", 
                f"${total_revenue:,.0f}", 
                revenue_growth, 
                "Ort. Büyüme"
            ), unsafe_allow_html=True)
        
        with col2:
            avg_profit_margin = filtered_df['Net_Kar_Marjı'].mean()
            profit_growth = ((filtered_df['Net_Kar'].mean() - df['Net_Kar'].mean()) / df['Net_Kar'].mean() * 100 
                           if df['Net_Kar'].mean() > 0 else 0)
            st.markdown(create_metric_card(
                "Ort. Net Kar Marjı", 
                f"{avg_profit_margin:.1f}%", 
                profit_growth, 
                "vs Genel"
            ), unsafe_allow_html=True)
        
        with col3:
            market_cap = filtered_df['Piyasa_Değeri'].sum()
            pe_ratio = filtered_df['F/K_Oranı'].mean()
            st.markdown(create_metric_card(
                "Toplam Piyasa Değeri", 
                f"${market_cap:,.0f}", 
                None, 
                f"Ort. F/K: {pe_ratio:.1f}"
            ), unsafe_allow_html=True)
        
        with col4:
            efficiency = filtered_df['Varlık_Devir_Hızı'].mean()
            debt_ratio = filtered_df['Borç_Oranı'].mean()
            st.markdown(create_metric_card(
                "Finansal Sağlık", 
                f"{((efficiency*10) + (100-debt_ratio))/2:.0f}/100", 
                None, 
                f"Borç: {debt_ratio:.1f}%"
            ), unsafe_allow_html=True)
        
        # İkinci metrik satırı
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            roe = (filtered_df['Net_Kar'] / filtered_df['Özkaynak']).mean() * 100
            st.markdown(create_metric_card(
                "Ort. Özkaynak Karlılığı (ROE)", 
                f"{roe:.1f}%", 
                None
            ), unsafe_allow_html=True)
        
        with col6:
            current_ratio = filtered_df['Cari_Oran'].mean()
            st.markdown(create_metric_card(
                "Ort. Cari Oran", 
                f"{current_ratio:.2f}", 
                None
            ), unsafe_allow_html=True)
        
        with col7:
            volatility = filtered_df['Getiri_Volatilitesi'].mean()
            st.markdown(create_metric_card(
                "Ort. Volatilite", 
                f"{volatility:.1f}%", 
                None
            ), unsafe_allow_html=True)
        
        with col8:
            esg_score = filtered_df[['Çevresel_Skor', 'Sosyal_Skor', 'Yönetişim_Skoru']].mean().mean()
            st.markdown(create_metric_card(
                "ESG Skoru", 
                f"{esg_score:.0f}/100", 
                None
            ), unsafe_allow_html=True)
        
        # Grafikler
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.plotly_chart(create_financial_health_chart(filtered_df), use_container_width=True)
        
        with col_chart2:
            st.plotly_chart(create_growth_comparison_chart(filtered_df), use_container_width=True)
        
        # Hızlı içgörüler
        st.markdown('<div class="section-header">🚀 HIZLI İÇGÖRÜLER</div>', unsafe_allow_html=True)
        
        insights = generate_advanced_insights(filtered_df)
        
        for insight in insights[:4]:
            st.markdown(display_insight_box(
                insight['title'],
                insight['content'],
                insight.get('metric'),
                insight.get('threshold')
            ), unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="section-header">📈 PERFORMANS ANALİZİ</div>', unsafe_allow_html=True)
        
        # DuPont analizi
        dupont_results = perform_dupont_analysis(filtered_df)
        
        col_dup1, col_dup2, col_dup3, col_dup4 = st.columns(4)
        
        with col_dup1:
            st.markdown(create_metric_card(
                "Net Kar Marjı", 
                f"{dupont_results['net_kar_marji']:.1f}%", 
                None
            ), unsafe_allow_html=True)
        
        with col_dup2:
            st.markdown(create_metric_card(
                "Varlık Devir Hızı", 
                f"{dupont_results['varlik_devir_hizi']:.2f}", 
                None
            ), unsafe_allow_html=True)
        
        with col_dup3:
            st.markdown(create_metric_card(
                "Özkaynak Çarpanı", 
                f"{dupont_results['ozkaynak_carpani']:.2f}", 
                None
            ), unsafe_allow_html=True)
        
        with col_dup4:
            st.markdown(create_metric_card(
                "ROE (DuPont)", 
                f"{dupont_results['roe_dupont']:.1f}%", 
                None
            ), unsafe_allow_html=True)
        
        # Performans grafikleri
        st.plotly_chart(create_correlation_heatmap(filtered_df), use_container_width=True)
        
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            # Büyüme dağılımı
            fig_growth_dist = px.histogram(
                filtered_df, 
                x='Ciro_Büyüme',
                nbins=30,
                title='Büyüme Oranı Dağılımı',
                labels={'Ciro_Büyüme': 'Büyüme (%)'},
                color_discrete_sequence=['#3b82f6']
            )
            fig_growth_dist.update_layout(
                height=400,
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b',
                font_color='white'
            )
            st.plotly_chart(fig_growth_dist, use_container_width=True)
        
        with col_perf2:
            # Karlılık dağılımı
            fig_profit_dist = px.box(
                filtered_df,
                y='Net_Kar_Marjı',
                x='Ana_Sektör',
                title='Sektörel Karlılık Dağılımı',
                color='Ana_Sektör'
            )
            fig_profit_dist.update_layout(
                height=400,
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b',
                font_color='white'
            )
            st.plotly_chart(fig_profit_dist, use_container_width=True)
        
        # Performans sıralaması
        st.markdown("### 🏆 Performans Sıralaması")
        
        top_performers = filtered_df.nlargest(10, 'Ciro_Büyüme')[
            ['Şirket_Adı', 'Ana_Sektör', 'Ciro_Büyüme', 'Net_Kar_Marjı', 'ROE', 'Piyasa_Değeri']
        ].copy()
        
        top_performers['ROE'] = (top_performers['Net_Kar_Marjı'] * 
                                filtered_df['Varlık_Devir_Hızı'].mean() * 
                                filtered_df['Özkaynak'].mean() / filtered_df['Toplam_Varlıklar'].mean())
        
        st.dataframe(
            top_performers.style.format({
                'Ciro_Büyüme': '{:.1f}%',
                'Net_Kar_Marjı': '{:.1f}%',
                'ROE': '{:.1f}%',
                'Piyasa_Değeri': '${:,.0f}'
            }).background_gradient(subset=['Ciro_Büyüme'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    with tab3:
        st.markdown('<div class="section-header">⚖️ RİSK ANALİZİ & ERKEN UYARI SİSTEMİ</div>', unsafe_allow_html=True)
        
        # Risk metrikleri
        risk_metrics = calculate_risk_metrics(filtered_df)
        
        col_risk1, col_risk2, col_risk3, col_risk4 = st.columns(4)
        
        with col_risk1:
            st.markdown(create_metric_card(
                "Ort. Beta Katsayısı", 
                f"{risk_metrics.get('beta_risk', 0):.2f}", 
                None,
                "Piyasa Risk Ölçüsü"
            ), unsafe_allow_html=True)
        
        with col_risk2:
            st.markdown(create_metric_card(
                "Likidite Riski", 
                f"{risk_metrics.get('liquidity_risk', 0):.1f}%", 
                None,
                "Cari Oran < 1"
            ), unsafe_allow_html=True)
        
        with col_risk3:
            st.markdown(create_metric_card(
                "Konsantrasyon Riski", 
                f"{risk_metrics.get('concentration_risk', 0):.1f}%", 
                None,
                "Top 3 Sektör Payı"
            ), unsafe_allow_html=True)
        
        with col_risk4:
            st.markdown(create_metric_card(
                "İflas Riski", 
                f"{risk_metrics.get('bankruptcy_risk', 0):.1f}%", 
                None,
                "Yüksek Risk Z-Score"
            ), unsafe_allow_html=True)
        
        # Altman Z-Score analizi
        st.markdown("### 📊 Altman Z-Score Analizi")
        altman_results = calculate_altman_z_score(filtered_df)
        
        if not altman_results.empty:
            col_z1, col_z2 = st.columns(2)
            
            with col_z1:
                # Risk dağılımı
                risk_dist = altman_results['Risk_Kategorisi'].value_counts()
                fig_risk = px.pie(
                    values=risk_dist.values,
                    names=risk_dist.index,
                    title='Z-Score Risk Dağılımı',
                    color_discrete_sequence=['#ef4444', '#f59e0b', '#10b981']
                )
                fig_risk.update_layout(
                    height=400,
                    paper_bgcolor='#1e293b',
                    plot_bgcolor='#1e293b',
                    font_color='white'
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with col_z2:
                # Z-Score dağılımı
                fig_zscore = px.histogram(
                    altman_results,
                    x='Z_Score',
                    nbins=30,
                    title='Z-Score Dağılımı',
                    labels={'Z_Score': 'Altman Z-Score'},
                    color_discrete_sequence=['#8b5cf6']
                )
                # Risk bölgelerini işaretle
                fig_zscore.add_vline(x=1.81, line_dash="dash", line_color="yellow")
                fig_zscore.add_vline(x=2.99, line_dash="dash", line_color="green")
                fig_zscore.add_annotation(x=1.4, y=0, text="Yüksek Risk", showarrow=False, font=dict(color="red"))
                fig_zscore.add_annotation(x=2.4, y=0, text="Gri Bölge", showarrow=False, font=dict(color="yellow"))
                fig_zscore.add_annotation(x=3.5, y=0, text="Düşük Risk", showarrow=False, font=dict(color="green"))
                
                fig_zscore.update_layout(
                    height=400,
                    paper_bgcolor='#1e293b',
                    plot_bgcolor='#1e293b',
                    font_color='white'
                )
                st.plotly_chart(fig_zscore, use_container_width=True)
        
        # Risk-Getiri analizi
        st.plotly_chart(create_risk_return_scatter(filtered_df), use_container_width=True)
        
        # Risk uyarıları
        st.markdown("### ⚠️ Risk Uyarıları")
        
        warning_col1, warning_col2 = st.columns(2)
        
        with warning_col1:
            # Yüksek borç riski
            high_debt = filtered_df[filtered_df['Borç_Oranı'] > 70]
            if len(high_debt) > 0:
                st.markdown(f'''
                <div class="warning-alert">
                    <h4>🚨 Yüksek Borç Riski</h4>
                    <p>{len(high_debt)} şirket borç oranı %70'in üzerinde</p>
                    <p>Ortalama borç oranı: {high_debt['Borç_Oranı'].mean():.1f}%</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Düşük likidite
            low_liquidity = filtered_df[filtered_df['Cari_Oran'] < 1]
            if len(low_liquidity) > 0:
                st.markdown(f'''
                <div class="warning-alert">
                    <h4>💧 Likidite Riski</h4>
                    <p>{len(low_liquidity)} şirket cari oranı 1'in altında</p>
                    <p>Ortalama cari oran: {low_liquidity['Cari_Oran'].mean():.2f}</p>
                </div>
                ''', unsafe_allow_html=True)
        
        with warning_col2:
            # Yüksek volatilite
            high_vol = filtered_df[filtered_df['Getiri_Volatilitesi'] > filtered_df['Getiri_Volatilitesi'].quantile(0.9)]
            if len(high_vol) > 0:
                st.markdown(f'''
                <div class="warning-alert">
                    <h4>📉 Yüksek Volatilite</h4>
                    <p>{len(high_vol)} şirket en yüksek %10 volatilite diliminde</p>
                    <p>Ortalama volatilite: {high_vol['Getiri_Volatilitesi'].mean():.1f}%</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Karlılık düşüşü
            declining_profit = filtered_df[filtered_df['Net_Kar_Marjı'] < 0]
            if len(declining_profit) > 0:
                st.markdown(f'''
                <div class="warning-alert">
                    <h4>📉 Zarar Eden Şirketler</h4>
                    <p>{len(declining_profit)} şirket net zarar durumunda</p>
                    <p>Ortalama zarar marjı: {declining_profit['Net_Kar_Marjı'].mean():.1f}%</p>
                </div>
                ''', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="section-header">💰 DEĞERLEME ANALİZİ & FİYATLAMA</div>', unsafe_allow_html=True)
        
        # Değerleme grafikleri
        st.plotly_chart(create_valuation_analysis(filtered_df), use_container_width=True)
        
        # Değerleme metrikleri
        col_val1, col_val2, col_val3, col_val4 = st.columns(4)
        
        with col_val1:
            median_pe = filtered_df['F/K_Oranı'].median()
            st.markdown(create_metric_card(
                "Medyan F/K Oranı", 
                f"{median_pe:.1f}", 
                None
            ), unsafe_allow_html=True)
        
        with col_val2:
            avg_pd_dd = filtered_df['PD/DD_Oranı'].mean()
            st.markdown(create_metric_card(
                "Ort. PD/DD Oranı", 
                f"{avg_pd_dd:.2f}", 
                None
            ), unsafe_allow_html=True)
        
        with col_val3:
            market_cap_to_revenue = filtered_df['Piyasa_Değeri'].sum() / filtered_df['Ciro'].sum()
            st.markdown(create_metric_card(
                "Piyasa Değeri/Ciro", 
                f"{market_cap_to_revenue:.2f}", 
                None
            ), unsafe_allow_html=True)
        
        with col_val4:
            peg_ratio = median_pe / filtered_df['Ciro_Büyüme'].mean() if filtered_df['Ciro_Büyüme'].mean() > 0 else 0
            st.markdown(create_metric_card(
                "PEG Oranı", 
                f"{peg_ratio:.2f}", 
                None
            ), unsafe_allow_html=True)
        
        # Değer yatırımı fırsatları
        st.markdown("### 💎 Değer Yatırımı Fırsatları")
        
        # Değer kriterleri
        value_stocks = filtered_df[
            (filtered_df['F/K_Oranı'] < filtered_df['F/K_Oranı'].median()) &
            (filtered_df['PD/DD_Oranı'] < 1) &
            (filtered_df['Net_Kar_Marjı'] > 5) &
            (filtered_df['Ciro_Büyüme'] > 5) &
            (filtered_df['Borç_Oranı'] < 50)
        ].copy()
        
        if len(value_stocks) > 0:
            value_stocks['Değer_Skoru'] = (
                (1 / value_stocks['F/K_Oranı']) * 30 +
                (1 / value_stocks['PD/DD_Oranı']) * 30 +
                value_stocks['Net_Kar_Marjı'] * 20 +
                value_stocks['Ciro_Büyüme'] * 10 +
                (100 - value_stocks['Borç_Oranı']) * 10
            ) / 100
            
            top_value = value_stocks.nlargest(10, 'Değer_Skoru')[
                ['Şirket_Adı', 'Ana_Sektör', 'F/K_Oranı', 'PD/DD_Oranı', 
                 'Net_Kar_Marjı', 'Ciro_Büyüme', 'Değer_Skoru']
            ]
            
            st.dataframe(
                top_value.style.format({
                    'F/K_Oranı': '{:.1f}',
                    'PD/DD_Oranı': '{:.2f}',
                    'Net_Kar_Marjı': '{:.1f}%',
                    'Ciro_Büyüme': '{:.1f}%',
                    'Değer_Skoru': '{:.2f}'
                }).background_gradient(subset=['Değer_Skoru'], cmap='RdYlGn'),
                use_container_width=True
            )
        else:
            st.info("Mevcut filtrelerle değer yatırımı fırsatı bulunamadı")
        
        # Büyüme yatırımı fırsatları
        st.markdown("### 🚀 Büyüme Yatırımı Fırsatları")
        
        growth_stocks = filtered_df[
            (filtered_df['Ciro_Büyüme'] > 20) &
            (filtered_df['Net_Kar_Marjı'] > 10) &
            (filtered_df['ARGE_Harcaması'] > filtered_df['ARGE_Harcaması'].median())
        ].copy()
        
        if len(growth_stocks) > 0:
            growth_stocks['Büyüme_Skoru'] = (
                growth_stocks['Ciro_Büyüme'] * 40 +
                growth_stocks['Net_Kar_Marjı'] * 30 +
                (growth_stocks['ARGE_Harcaması'] / growth_stocks['ARGE_Harcaması'].max() * 100) * 20 +
                (100 - growth_stocks['Borç_Oranı']) * 10
            ) / 100
            
            top_growth = growth_stocks.nlargest(10, 'Büyüme_Skoru')[
                ['Şirket_Adı', 'Ana_Sektör', 'Ciro_Büyüme', 'Net_Kar_Marjı', 
                 'ARGE_Harcaması', 'Borç_Oranı', 'Büyüme_Skoru']
            ]
            
            st.dataframe(
                top_growth.style.format({
                    'Ciro_Büyüme': '{:.1f}%',
                    'Net_Kar_Marjı': '{:.1f}%',
                    'ARGE_Harcaması': '${:,.0f}',
                    'Borç_Oranı': '{:.1f}%',
                    'Büyüme_Skoru': '{:.2f}'
                }).background_gradient(subset=['Büyüme_Skoru'], cmap='RdYlGn'),
                use_container_width=True
            )
        else:
            st.info("Mevcut filtrelerle büyüme yatırımı fırsatı bulunamadı")
    
    with tab5:
        st.markdown('<div class="section-header">🔍 DETAYLI ANALİZ & İLERİ METRİKLER</div>', unsafe_allow_html=True)
        
        # Regresyon analizi
        st.markdown("### 📊 Regresyon Analizi")
        
        regression_results = perform_regression_analysis(filtered_df)
        
        if regression_results:
            col_reg1, col_reg2 = st.columns(2)
            
            with col_reg1:
                st.markdown("#### Korelasyon Katsayıları")
                correlation_df = pd.DataFrame({
                    'Metrik': regression_results['correlation'].index,
                    'Korelasyon': regression_results['correlation'].values
                })
                st.dataframe(
                    correlation_df.style.format({'Korelasyon': '{:.3f}'})
                    .background_gradient(subset=['Korelasyon'], cmap='RdYlBu'),
                    use_container_width=True
                )
            
            with col_reg2:
                st.markdown("#### Regresyon Önem Derecesi")
                importance_df = pd.DataFrame([
                    {
                        'Metrik': k,
                        'R²': v['r_squared'],
                        'p-Değeri': v['p_value'],
                        'Önem': v['importance']
                    }
                    for k, v in regression_results['regression'].items()
                ])
                st.dataframe(
                    importance_df.style.format({
                        'R²': '{:.3f}',
                        'p-Değeri': '{:.4f}',
                        'Önem': '{:.3f}'
                    }).background_gradient(subset=['Önem'], cmap='RdYlGn'),
                    use_container_width=True
                )
        
        # Çok değişkenli analiz
        st.markdown("### 🎯 Çok Değişkenli Analiz")
        
        col_multi1, col_multi2 = st.columns(2)
        
        with col_multi1:
            # Cluster analizi için hazırlık
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            
            cluster_features = filtered_df[[
                'Ciro_Büyüme', 'Net_Kar_Marjı', 'Borç_Oranı', 
                'Varlık_Devir_Hızı', 'Getiri_Volatilitesi'
            ]].dropna()
            
            if len(cluster_features) > 10:
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(cluster_features)
                
                # Optimal cluster sayısı (Elbow method)
                wcss = []
                for i in range(1, 11):
                    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
                    kmeans.fit(scaled_features)
                    wcss.append(kmeans.inertia_)
                
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=list(range(1, 11)),
                    y=wcss,
                    mode='lines+markers',
                    line=dict(color='#3b82f6', width=2),
                    marker=dict(size=8)
                ))
                
                fig_elbow.update_layout(
                    title='Elbow Method - Optimal Cluster Sayısı',
                    xaxis_title='Cluster Sayısı',
                    yaxis_title='WCSS',
                    height=400,
                    paper_bgcolor='#1e293b',
                    plot_bgcolor='#1e293b',
                    font_color='white'
                )
                st.plotly_chart(fig_elbow, use_container_width=True)
        
        with col_multi2:
            # Senaryo analizi
            st.markdown("#### 📈 Senaryo Analizi")
            
            base_growth = filtered_df['Ciro_Büyüme'].mean()
            base_margin = filtered_df['Net_Kar_Marjı'].mean()
            
            scenarios = {
                'İyimser': {'growth': base_growth * 1.5, 'margin': base_margin * 1.2},
                'Baz': {'growth': base_growth, 'margin': base_margin},
                'Kötümser': {'growth': base_growth * 0.5, 'margin': base_margin * 0.8}
            }
            
            scenario_data = []
            for name, values in scenarios.items():
                scenario_data.append({
                    'Senaryo': name,
                    'Büyüme': values['growth'],
                    'Karlılık': values['margin'],
                    'Beklenen Getiri': values['growth'] * values['margin'] / 100
                })
            
            scenario_df = pd.DataFrame(scenario_data)
            st.dataframe(
                scenario_df.style.format({
                    'Büyüme': '{:.1f}%',
                    'Karlılık': '{:.1f}%',
                    'Beklenen Getiri': '{:.3f}'
                }),
                use_container_width=True
            )
        
        # Zaman serisi analizi
        st.markdown("### ⏳ Zaman Serisi Analizi")
        
        if 'Ciro_Q1' in filtered_df.columns and 'Ciro_Q4' in filtered_df.columns:
            quarterly_data = filtered_df[['Ciro_Q1', 'Ciro_Q2', 'Ciro_Q3', 'Ciro']].mean()
            quarterly_data.index = ['Q1', 'Q2', 'Q3', 'Q4']
            
            fig_quarterly = go.Figure()
            fig_quarterly.add_trace(go.Scatter(
                x=quarterly_data.index,
                y=quarterly_data.values,
                mode='lines+markers',
                line=dict(color='#10b981', width=3),
                marker=dict(size=10),
                name='Ortalama Ciro'
            ))
            
            fig_quarterly.update_layout(
                title='Çeyreklik Ciro Trendi',
                xaxis_title='Çeyrek',
                yaxis_title='Ciro ($)',
                height=400,
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b',
                font_color='white'
            )
            
            st.plotly_chart(fig_quarterly, use_container_width=True)
    
    with tab6:
        st.markdown('<div class="section-header">📊 VERİ KEŞFİ & HAM VERİ</div>', unsafe_allow_html=True)
        
        # Veri keşfi kontrolleri
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            show_rows = st.selectbox("Gösterilecek Satır Sayısı", [100, 500, 1000, 5000], index=0)
        
        with col_exp2:
            sort_column = st.selectbox(
                "Sıralama Sütunu",
                ['Ciro', 'Net_Kar_Marjı', 'Ciro_Büyüme', 'Piyasa_Değeri', 'F/K_Oranı'],
                index=0
            )
        
        with col_exp3:
            sort_direction = st.selectbox("Sıralama Yönü", ['Azalan', 'Artan'], index=0)
        
        # Veri görüntüleme
        display_data = filtered_df.sort_values(
            sort_column, 
            ascending=(sort_direction == 'Artan')
        ).head(show_rows)
        
        st.dataframe(display_data, use_container_width=True)
        
        # Veri indirme
        st.markdown("---")
        st.markdown("### 📥 Veri İndirme")
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Filtrelenmiş Veriyi İndir (CSV)",
                data=csv,
                file_name="filtrelenmis_finansal_veri.csv",
                mime="text/csv"
            )
        
        with col_dl2:
            sample_size = st.slider("Örneklem Boyutu", 1000, 10000, 5000, 1000)
            sample_data = filtered_df.sample(min(sample_size, len(filtered_df))).to_csv(index=False)
            st.download_button(
                label=f"📥 {sample_size} Satır Örnek Veri (CSV)",
                data=sample_data,
                file_name=f"ornek_finansal_veri_{sample_size}.csv",
                mime="text/csv"
            )
        
        # Veri kalitesi raporu
        st.markdown("---")
        st.markdown("### 🔍 Veri Kalitesi Raporu")
        
        quality_col1, quality_col2, quality_col3 = st.columns(3)
        
        with quality_col1:
            missing_data = filtered_df.isnull().sum().sum()
            total_cells = filtered_df.size
            missing_percentage = (missing_data / total_cells) * 100
            
            st.metric(
                "Eksik Veri Oranı",
                f"{missing_percentage:.2f}%",
                delta=f"{missing_data:,} hücre"
            )
        
        with quality_col2:
            duplicate_rows = filtered_df.duplicated().sum()
            st.metric(
                "Kopya Satırlar",
                f"{duplicate_rows:,}",
                delta=f"{(duplicate_rows/len(filtered_df)*100):.2f}%"
            )
        
        with quality_col3:
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).shape[1]
            total_cols = filtered_df.shape[1]
            st.metric(
                "Sayısal Sütunlar",
                f"{numeric_cols}/{total_cols}",
                delta=f"{(numeric_cols/total_cols*100):.1f}%"
            )
        
        # Veri istatistikleri
        st.markdown("#### 📈 Sayısal Veri İstatistikleri")
        
        numeric_stats = filtered_df.select_dtypes(include=[np.number]).describe().T
        numeric_stats = numeric_stats[['mean', 'std', 'min', '50%', 'max']]
        numeric_stats.columns = ['Ortalama', 'Standart Sapma', 'Minimum', 'Medyan', 'Maksimum']
        
        st.dataframe(
            numeric_stats.style.format('{:,.2f}'),
            use_container_width=True
        )

if __name__ == "__main__":
    import gc
    gc.collect()
    main()
