"""
Excel Data Processing Web Application
Streamlit ile oluşturulmuş profesyonel Excel veri işleme uygulaması
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import io
import json
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(
    page_title="Excel Data Processor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #10B981;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

class DataProcessor:
    """Veri işleme sınıfı"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.cleaning_report = {}
        self.analysis_results = {}
        
    def clean_data(self):
        """Veri temizleme işlemleri"""
        st.info("Veri temizleniyor...")
        
        # Sütun isimlerini temizle
        self.df.columns = [self._clean_column_name(col) for col in self.df.columns]
        
        # Eksik değerleri işle
        missing_before = self.df.isnull().sum().sum()
        self.df = self._handle_missing_values(self.df)
        missing_after = self.df.isnull().sum().sum()
        
        # Veri tiplerini düzelt
        self.df = self._fix_data_types(self.df)
        
        # Tekrar eden satırları kaldır
        duplicates_before = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        duplicates_after = self.df.duplicated().sum()
        
        self.cleaning_report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'missing_values': {
                'before': missing_before,
                'after': missing_after,
                'reduction': missing_before - missing_after
            },
            'duplicates': {
                'before': duplicates_before,
                'after': duplicates_after,
                'removed': duplicates_before - duplicates_after
            }
        }
        
        return self.df
    
    def _clean_column_name(self, col):
        """Sütun ismini temizle"""
        col = str(col).strip()
        col = col.replace('\n', ' ').replace('\r', ' ')
        col = col.replace(' ', '_').replace('.', '_').replace('-', '_')
        col = ''.join(c for c in col if c.isalnum() or c == '_')
        return col.lower()
    
    def _handle_missing_values(self, df):
        """Eksik değerleri işle"""
        # Sayısal kolonlar için medyan ile doldur
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Kategorik kolonlar için mod ile doldur
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
        
        return df
    
    def _fix_data_types(self, df):
        """Veri tiplerini düzelt"""
        # Tarih kolonlarını bul ve dönüştür
        date_patterns = ['date', 'time', 'year', 'month', 'day']
        for col in df.columns:
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in date_patterns):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        # Sayısal kolonları float'a çevir
        numeric_patterns = ['price', 'cost', 'amount', 'value', 'unit', 'usd', 'mnf']
        for col in df.columns:
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in numeric_patterns):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        return df
    
    def analyze_data(self):
        """Veri analizi yap"""
        st.info("Veri analizi yapılıyor...")
        
        self.analysis_results = {
            'descriptive_stats': self._get_descriptive_stats(),
            'correlation_matrix': self._get_correlation_matrix(),
            'top_correlations': self._get_top_correlations(),
            'data_quality': self._get_data_quality_report()
        }
        
        return self.analysis_results
    
    def _get_descriptive_stats(self):
        """Tanımlayıcı istatistikler"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            stats = self.df[numeric_cols].describe().T
            stats['missing'] = self.df[numeric_cols].isnull().sum()
            stats['missing_percent'] = (stats['missing'] / len(self.df) * 100).round(2)
            stats['skewness'] = self.df[numeric_cols].skew()
            stats['kurtosis'] = self.df[numeric_cols].kurtosis()
            
            return stats.round(2).to_dict('index')
        
        return {}
    
    def _get_correlation_matrix(self):
        """Korelasyon matrisi"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr().round(3)
            return corr_matrix.to_dict()
        
        return {}
    
    def _get_top_correlations(self):
        """En yüksek korelasyonlar"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.5:  # Sadece güçlü korelasyonlar
                        correlations.append({
                            'variable1': corr_matrix.columns[i],
                            'variable2': corr_matrix.columns[j],
                            'correlation': round(corr, 3)
                        })
            
            # Korelasyona göre sırala
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            return correlations[:10]  # İlk 10'u döndür
        
        return []
    
    def _get_data_quality_report(self):
        """Veri kalite raporu"""
        report = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values_total': self.df.isnull().sum().sum(),
            'missing_values_percent': round((self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100), 2),
            'duplicate_rows': self.df.duplicated().sum(),
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object']).columns),
            'date_columns': len(self.df.select_dtypes(include=['datetime64']).columns)
        }
        return report

def create_download_link(df, filename="processed_data.csv"):
    """İndirme linki oluştur"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {filename} indir</a>'
    return href

def main():
    """Ana uygulama"""
    
    # Başlık
    st.markdown('<h1 class="main-header">📊 Excel Data Processor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #6B7280; margin-bottom: 2rem;'>
    Profesyonel Excel veri işleme, analiz ve görselleştirme uygulaması
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Ayarlar")
        
        # Dosya yükleme
        uploaded_file = st.file_uploader(
            "Excel dosyası yükle",
            type=['xlsx', 'xls', 'csv'],
            help="Excel veya CSV dosyası yükleyin"
        )
        
        if uploaded_file:
            st.success(f"✓ {uploaded_file.name} yüklendi")
        
        st.markdown("---")
        
        # İşlem seçenekleri
        st.markdown("### 🔧 İşlemler")
        auto_clean = st.checkbox("Otomatik temizleme", value=True)
        show_analysis = st.checkbox("Analiz göster", value=True)
        create_visualizations = st.checkbox("Görselleştirme oluştur", value=True)
        
        st.markdown("---")
        
        # Hakkında
        st.markdown("### ℹ️ Hakkında")
        st.markdown("""
        Bu uygulama ile:
        - Excel/CSV dosyalarını yükleyin
        - Veriyi otomatik temizleyin
        - İstatistiksel analiz yapın
        - Görselleştirmeler oluşturun
        - Sonuçları indirin
        """)
    
    # Ana içerik
    if uploaded_file is not None:
        try:
            # Dosyayı yükle
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # İlk görünüm
            with st.expander("👁️ Ham Veri Önizleme", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Satır Sayısı", len(df))
                with col2:
                    st.metric("Sütun Sayısı", len(df.columns))
                with col3:
                    missing = df.isnull().sum().sum()
                    st.metric("Eksik Değer", missing)
                
                # Veri önizleme
                st.dataframe(df.head(), use_container_width=True)
                
                # Sütun bilgileri
                col_info = pd.DataFrame({
                    'Sütun': df.columns,
                    'Tip': df.dtypes.values,
                    'Benzersiz Değer': df.nunique().values,
                    'Eksik Değer': df.isnull().sum().values
                })
                st.dataframe(col_info, use_container_width=True)
            
            # Veri işleme
            if auto_clean:
                processor = DataProcessor(df)
                processor.original_shape = df.shape
                
                # Temizleme butonu
                if st.button("🚀 Veriyi İşle ve Temizle", type="primary"):
                    with st.spinner("Veri işleniyor..."):
                        df_clean = processor.clean_data()
                        analysis_results = processor.analyze_data()
                    
                    # Temizleme raporu
                    st.markdown('<div class="sub-header">🧹 Temizleme Raporu</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Önceki Satır", processor.cleaning_report['original_shape'][0])
                        st.metric("Sonraki Satır", processor.cleaning_report['cleaned_shape'][0])
                    with col2:
                        st.metric("Önceki Sütun", processor.cleaning_report['original_shape'][1])
                        st.metric("Sonraki Sütun", processor.cleaning_report['cleaned_shape'][1])
                    with col3:
                        st.metric("Eksik Değer (Önce)", processor.cleaning_report['missing_values']['before'])
                        st.metric("Eksik Değer (Sonra)", processor.cleaning_report['missing_values']['after'])
                    with col4:
                        st.metric("Silinen Tekrarlar", processor.cleaning_report['duplicates']['removed'])
                    
                    # Temizlenmiş veri önizleme
                    with st.expander("✅ Temizlenmiş Veri", expanded=True):
                        st.dataframe(df_clean.head(), use_container_width=True)
                        
                        # İndirme linki
                        st.markdown("### 📥 İndirme Seçenekleri")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(create_download_link(df_clean, "cleaned_data.csv"), unsafe_allow_html=True)
                        with col2:
                            # Excel olarak indirme
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df_clean.to_excel(writer, index=False, sheet_name='Cleaned_Data')
                            excel_data = output.getvalue()
                            b64 = base64.b64encode(excel_data).decode()
                            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="cleaned_data.xlsx">📊 Excel dosyası indir</a>'
                            st.markdown(href, unsafe_allow_html=True)
                    
                    # Analiz sonuçları
                    if show_analysis:
                        st.markdown('<div class="sub-header">📈 Analiz Sonuçları</div>', unsafe_allow_html=True)
                        
                        # Veri kalite raporu
                        if 'data_quality' in analysis_results:
                            st.markdown("##### 📊 Veri Kalite Raporu")
                            quality = analysis_results['data_quality']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Toplam Satır", quality['total_rows'])
                                st.metric("Kategorik Sütun", quality['categorical_columns'])
                            with col2:
                                st.metric("Toplam Sütun", quality['total_columns'])
                                st.metric("Sayısal Sütun", quality['numeric_columns'])
                            with col3:
                                st.metric("Eksik Değer", quality['missing_values_total'])
                                st.metric("Tekrar Eden Satır", quality['duplicate_rows'])
                            with col4:
                                st.metric("Eksik Değer %", f"{quality['missing_values_percent']}%")
                                st.metric("Tarih Sütunu", quality['date_columns'])
                        
                        # Tanımlayıcı istatistikler
                        if 'descriptive_stats' in analysis_results and analysis_results['descriptive_stats']:
                            st.markdown("##### 📋 Tanımlayıcı İstatistikler")
                            
                            # Sayısal kolon seçimi
                            numeric_cols = list(analysis_results['descriptive_stats'].keys())
                            selected_cols = st.multiselect(
                                "İstatistik görmek istediğiniz sütunları seçin:",
                                numeric_cols,
                                default=numeric_cols[:min(5, len(numeric_cols))]
                            )
                            
                            if selected_cols:
                                stats_data = {col: analysis_results['descriptive_stats'][col] 
                                            for col in selected_cols}
                                stats_df = pd.DataFrame(stats_data).T
                                st.dataframe(stats_df, use_container_width=True)
                        
                        # Korelasyon analizi
                        if 'top_correlations' in analysis_results and analysis_results['top_correlations']:
                            st.markdown("##### 🔗 Korelasyon Analizi")
                            
                            # En yüksek korelasyonlar
                            if analysis_results['top_correlations']:
                                st.markdown("**En Yüksek Korelasyonlar:**")
                                corr_data = []
                                for corr in analysis_results['top_correlations'][:10]:
                                    corr_data.append({
                                        'Değişken 1': corr['variable1'],
                                        'Değişken 2': corr['variable2'],
                                        'Korelasyon': corr['correlation'],
                                        'Güç': 'Güçlü' if abs(corr['correlation']) > 0.7 else 'Orta'
                                    })
                                
                                corr_df = pd.DataFrame(corr_data)
                                st.dataframe(corr_df, use_container_width=True)
                            
                            # Korelasyon heatmap
                            if 'correlation_matrix' in analysis_results and analysis_results['correlation_matrix']:
                                st.markdown("**Korelasyon Matrisi:**")
                                
                                # Sayısal kolonları seç
                                numeric_df = df_clean.select_dtypes(include=[np.number])
                                if len(numeric_df.columns) > 1:
                                    # Sadece ilk 10 sütunu göster (performans için)
                                    display_cols = numeric_df.columns[:min(10, len(numeric_df.columns))]
                                    corr_matrix = numeric_df[display_cols].corr()
                                    
                                    # Heatmap oluştur
                                    fig = px.imshow(
                                        corr_matrix,
                                        text_auto='.2f',
                                        aspect='auto',
                                        color_continuous_scale='RdBu',
                                        title='Korelasyon Heatmap'
                                    )
                                    fig.update_layout(height=600)
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Görselleştirmeler
                    if create_visualizations and len(df_clean) > 0:
                        st.markdown('<div class="sub-header">📊 Görselleştirmeler</div>', unsafe_allow_html=True)
                        
                        # Görselleştirme seçenekleri
                        viz_type = st.selectbox(
                            "Görselleştirme Türü Seçin:",
                            ["Dağılım Grafiği", "Bar Grafiği", "Scatter Plot", "Histogram", "Box Plot"]
                        )
                        
                        # Kolon seçimi
                        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
                        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
                        
                        if viz_type == "Dağılım Grafiği" and len(numeric_cols) >= 1:
                            x_col = st.selectbox("X Ekseni:", numeric_cols)
                            color_col = st.selectbox("Renk Kategorisi (opsiyonel):", [None] + categorical_cols)
                            
                            if st.button("Grafik Oluştur"):
                                fig = px.histogram(
                                    df_clean,
                                    x=x_col,
                                    color=color_col if color_col else None,
                                    title=f"{x_col} Dağılımı",
                                    nbins=30,
                                    opacity=0.7
                                )
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == "Bar Grafiği" and len(categorical_cols) >= 1:
                            cat_col = st.selectbox("Kategorik Sütun:", categorical_cols)
                            if st.button("Grafik Oluştur"):
                                value_counts = df_clean[cat_col].value_counts().head(20)
                                fig = px.bar(
                                    x=value_counts.index,
                                    y=value_counts.values,
                                    title=f"{cat_col} Dağılımı",
                                    labels={'x': cat_col, 'y': 'Sayı'}
                                )
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
                            col1, col2 = st.columns(2)
                            with col1:
                                x_col = st.selectbox("X Ekseni:", numeric_cols)
                            with col2:
                                y_col = st.selectbox("Y Ekseni:", numeric_cols)
                            
                            color_col = st.selectbox("Renk Kategorisi (opsiyonel):", [None] + categorical_cols)
                            
                            if st.button("Grafik Oluştur"):
                                fig = px.scatter(
                                    df_clean,
                                    x=x_col,
                                    y=y_col,
                                    color=color_col if color_col else None,
                                    title=f"{x_col} vs {y_col}",
                                    opacity=0.6
                                )
                                fig.update_layout(height=600)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == "Histogram" and len(numeric_cols) >= 1:
                            num_col = st.selectbox("Sayısal Sütun:", numeric_cols)
                            color_col = st.selectbox("Renk Kategorisi (opsiyonel):", [None] + categorical_cols)
                            
                            if st.button("Grafik Oluştur"):
                                fig = px.histogram(
                                    df_clean,
                                    x=num_col,
                                    color=color_col if color_col else None,
                                    title=f"{num_col} Histogramı",
                                    nbins=30,
                                    marginal="box"
                                )
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == "Box Plot" and len(numeric_cols) >= 1:
                            num_col = st.selectbox("Sayısal Sütun:", numeric_cols)
                            cat_col = st.selectbox("Kategorik Sütun (opsiyonel):", [None] + categorical_cols)
                            
                            if st.button("Grafik Oluştur"):
                                if cat_col:
                                    fig = px.box(
                                        df_clean,
                                        x=cat_col,
                                        y=num_col,
                                        title=f"{num_col} Dağılımı ({cat_col}'a göre)"
                                    )
                                else:
                                    fig = px.box(
                                        df_clean,
                                        y=num_col,
                                        title=f"{num_col} Dağılımı"
                                    )
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Otomatik görselleştirmeler
                        with st.expander("🎨 Otomatik Görselleştirmeler"):
                            st.markdown("**Sayısal Değişkenlerin Dağılımı:**")
                            
                            if len(numeric_cols) > 0:
                                # İlk 4 sayısal değişkenin dağılımı
                                display_cols = numeric_cols[:min(4, len(numeric_cols))]
                                fig = make_subplots(
                                    rows=2, 
                                    cols=2,
                                    subplot_titles=display_cols
                                )
                                
                                for idx, col in enumerate(display_cols):
                                    row = (idx // 2) + 1
                                    col_num = (idx % 2) + 1
                                    
                                    fig.add_trace(
                                        go.Histogram(
                                            x=df_clean[col].dropna(),
                                            name=col,
                                            nbinsx=30
                                        ),
                                        row=row, col=col_num
                                    )
                                
                                fig.update_layout(height=600, showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Kategorik değişkenler
                            if len(categorical_cols) > 0:
                                st.markdown("**Kategorik Değişkenlerin Dağılımı:**")
                                
                                # İlk 3 kategorik değişken
                                display_cats = categorical_cols[:min(3, len(categorical_cols))]
                                
                                for cat_col in display_cats:
                                    if df_clean[cat_col].nunique() <= 20:  # Çok fazla kategori yoksa
                                        value_counts = df_clean[cat_col].value_counts().head(10)
                                        fig = px.bar(
                                            x=value_counts.index,
                                            y=value_counts.values,
                                            title=f"{cat_col} (Top 10)",
                                            labels={'x': cat_col, 'y': 'Sayı'}
                                        )
                                        fig.update_layout(height=400)
                                        st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("Veriyi işlemek için yukarıdaki butona tıklayın.")
        
        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
            st.info("Lütfen Excel dosyasının formatını kontrol edin.")
    
    else:
        # Hoş geldiniz ekranı
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/3767/3767094.png", width=200)
        
        st.markdown("""
        <div class="metric-card">
        <h3>👋 Hoş Geldiniz!</h3>
        <p>Başlamak için lütfen sol taraftan bir Excel veya CSV dosyası yükleyin.</p>
        <p><strong>Desteklenen formatlar:</strong></p>
        <ul>
            <li>Excel (.xlsx, .xls)</li>
            <li>CSV (.csv)</li>
        </ul>
        <p><strong>Özellikler:</strong></p>
        <ul>
            <li>Otomatik veri temizleme</li>
            <li>Eksik değer işleme</li>
            <li>İstatistiksel analiz</li>
            <li>İnteraktif görselleştirmeler</li>
            <li>Veri indirme</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Örnek veri yükleme
        st.markdown("---")
        st.markdown("### 🚀 Hemen Başlayın")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📁 Örnek Veri Yükle (Demo)", type="secondary"):
                # Örnek veri oluştur
                sample_data = {
                    'Product': ['A', 'B', 'C', 'D', 'E'] * 4,
                    'Category': ['X', 'X', 'Y', 'Y', 'Z'] * 4,
                    'Sales': np.random.randint(100, 1000, 20),
                    'Price': np.random.uniform(10, 100, 20),
                    'Date': pd.date_range('2023-01-01', periods=20, freq='D'),
                    'Region': ['North', 'South'] * 10
                }
                df_sample = pd.DataFrame(sample_data)
                
                # Geçici olarak session state'e kaydet
                st.session_state['sample_data'] = df_sample
                st.rerun()
        
        with col2:
            if st.button("ℹ️ Kullanım Kılavuzu", type="secondary"):
                st.info("""
                **Kullanım Adımları:**
                1. Sol taraftan Excel/CSV dosyası yükleyin
                2. İstenilen işlemleri seçin
                3. "Veriyi İşle ve Temizle" butonuna tıklayın
                4. Sonuçları görüntüleyin ve indirin
                
                **İpuçları:**
                - Büyük dosyalar için işlem biraz zaman alabilir
                - Görselleştirmeler için sayısal veri gereklidir
                - Eksik değerler otomatik olarak doldurulur
                """)

if __name__ == "__main__":
    main()
