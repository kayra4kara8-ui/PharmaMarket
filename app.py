# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Pharma Commercial Analytics", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    
    numeric_cols = [
        'MAT Q3 2022 USD MNF', 'MAT Q3 2022 Standard Units', 'MAT Q3 2022 Units',
        'MAT Q3 2022 SU Avg Price USD MNF', 'MAT Q3 2022 Unit Avg Price USD MNF',
        'MAT Q3 2023 USD MNF', 'MAT Q3 2023 Standard Units', 'MAT Q3 2023 Units',
        'MAT Q3 2023 SU Avg Price USD MNF', 'MAT Q3 2023 Unit Avg Price USD MNF',
        'MAT Q3 2024 USD MNF', 'MAT Q3 2024 Standard Units', 'MAT Q3 2024 Units',
        'MAT Q3 2024 SU Avg Price USD MNF', 'MAT Q3 2024 Unit Avg Price USD MNF'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.fillna(0, inplace=True)
    
    return df

def apply_filters(df, country_filter, corp_filter, mol_filter, sector_filter, panel_filter, year_filter, specialty_filter):
    filtered = df.copy()
    
    if country_filter and len(country_filter) > 0:
        filtered = filtered[filtered['Country'].isin(country_filter)]
    
    if corp_filter and corp_filter != 'All':
        filtered = filtered[filtered['Corporation'] == corp_filter]
    
    if mol_filter and mol_filter != 'All':
        filtered = filtered[filtered['Molecule'] == mol_filter]
    
    if sector_filter and sector_filter != 'All':
        filtered = filtered[filtered['Sector'] == sector_filter]
    
    if panel_filter and panel_filter != 'All':
        filtered = filtered[filtered['Panel'] == panel_filter]
    
    if specialty_filter and specialty_filter != 'All':
        filtered = filtered[filtered['Specialty Product'] == specialty_filter]
    
    return filtered

def format_number(num):
    if num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

def format_units(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.0f}"

def calculate_growth(old_val, new_val):
    if old_val == 0:
        return 0
    return ((new_val - old_val) / old_val) * 100

def get_insights(df, year_filter):
    insights = []
    
    if '2022' in year_filter or len(year_filter) == 0:
        usd_2022 = df['MAT Q3 2022 USD MNF'].sum()
        units_2022 = df['MAT Q3 2022 Units'].sum()
        su_2022 = df['MAT Q3 2022 Standard Units'].sum()
    else:
        usd_2022 = 0
        units_2022 = 0
        su_2022 = 0
    
    if '2023' in year_filter or len(year_filter) == 0:
        usd_2023 = df['MAT Q3 2023 USD MNF'].sum()
        units_2023 = df['MAT Q3 2023 Units'].sum()
        su_2023 = df['MAT Q3 2023 Standard Units'].sum()
    else:
        usd_2023 = 0
        units_2023 = 0
        su_2023 = 0
    
    if '2024' in year_filter or len(year_filter) == 0:
        usd_2024 = df['MAT Q3 2024 USD MNF'].sum()
        units_2024 = df['MAT Q3 2024 Units'].sum()
        su_2024 = df['MAT Q3 2024 Standard Units'].sum()
    else:
        usd_2024 = 0
        units_2024 = 0
        su_2024 = 0
    
    if usd_2022 > 0 and usd_2023 > 0:
        growth_22_23 = calculate_growth(usd_2022, usd_2023)
        insights.append(f"2022'den 2023'e satışlar %{growth_22_23:.1f} değişim gösterdi.")
    
    if usd_2023 > 0 and usd_2024 > 0:
        growth_23_24 = calculate_growth(usd_2023, usd_2024)
        insights.append(f"2023'ten 2024'e satışlar %{growth_23_24:.1f} değişim gösterdi.")
    
    if usd_2022 > 0 and usd_2024 > 0:
        cagr = ((usd_2024 / usd_2022) ** (1/2) - 1) * 100
        insights.append(f"2022-2024 arası bileşik yıllık büyüme oranı %{cagr:.1f}.")
    
    top_countries = df.groupby('Country')[['MAT Q3 2024 USD MNF']].sum().sort_values('MAT Q3 2024 USD MNF', ascending=False).head(3)
    if len(top_countries) > 0:
        top_country = top_countries.index[0]
        top_value = top_countries.iloc[0, 0]
        insights.append(f"En yüksek satış {top_country} ülkesinde: {format_number(top_value)}.")
    
    top_molecules = df.groupby('Molecule')[['MAT Q3 2024 USD MNF']].sum().sort_values('MAT Q3 2024 USD MNF', ascending=False).head(3)
    if len(top_molecules) > 0:
        top_mol = top_molecules.index[0]
        top_mol_value = top_molecules.iloc[0, 0]
        insights.append(f"En yüksek satışlı molekül {top_mol}: {format_number(top_mol_value)}.")
    
    top_corps = df.groupby('Corporation')[['MAT Q3 2024 USD MNF']].sum().sort_values('MAT Q3 2024 USD MNF', ascending=False).head(3)
    if len(top_corps) > 0:
        top_corp = top_corps.index[0]
        top_corp_value = top_corps.iloc[0, 0]
        total_market = df['MAT Q3 2024 USD MNF'].sum()
        if total_market > 0:
            market_share = (top_corp_value / total_market) * 100
            insights.append(f"{top_corp} şirketinin pazar payı %{market_share:.1f}.")
    
    specialty_sales = df[df['Specialty Product'] == 'Yes']['MAT Q3 2024 USD MNF'].sum()
    total_sales = df['MAT Q3 2024 USD MNF'].sum()
    if total_sales > 0:
        specialty_pct = (specialty_sales / total_sales) * 100
        insights.append(f"Specialty ürünler toplam satışların %{specialty_pct:.1f}'ini oluşturuyor.")
    
    avg_price_2022 = df['MAT Q3 2022 SU Avg Price USD MNF'].replace(0, np.nan).mean()
    avg_price_2024 = df['MAT Q3 2024 SU Avg Price USD MNF'].replace(0, np.nan).mean()
    if not np.isnan(avg_price_2022) and not np.isnan(avg_price_2024) and avg_price_2022 > 0:
        price_change = calculate_growth(avg_price_2022, avg_price_2024)
        insights.append(f"Ortalama SU fiyatı 2022-2024 arasında %{price_change:.1f} değişti.")
    
    if units_2023 > 0 and units_2024 > 0:
        volume_growth = calculate_growth(units_2023, units_2024)
        insights.append(f"Hacim (Units) 2023-2024 arasında %{volume_growth:.1f} büyüdü.")
    
    return insights

st.title("🏥 Pharma Commercial Analytics Platform")
st.markdown("---")

uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=['xlsx', 'xls'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    st.sidebar.header("🔍 Global Filtreler")
    
    countries = ['All'] + sorted(df['Country'].dropna().unique().tolist())
    country_filter = st.sidebar.multiselect("Country", options=sorted(df['Country'].dropna().unique().tolist()), default=[])
    
    corporations = ['All'] + sorted(df['Corporation'].dropna().unique().tolist())
    corp_filter = st.sidebar.selectbox("Corporation", options=corporations)
    
    molecules = ['All'] + sorted(df['Molecule'].dropna().unique().tolist())
    mol_filter = st.sidebar.selectbox("Molecule", options=molecules)
    
    sectors = ['All'] + sorted(df['Sector'].dropna().unique().tolist())
    sector_filter = st.sidebar.selectbox("Sector", options=sectors)
    
    panels = ['All'] + sorted(df['Panel'].dropna().unique().tolist())
    panel_filter = st.sidebar.selectbox("Panel", options=panels)
    
    year_filter = st.sidebar.multiselect("Yıl", options=['2022', '2023', '2024'], default=['2022', '2023', '2024'])
    
    specialty_options = ['All'] + sorted(df['Specialty Product'].dropna().unique().tolist())
    specialty_filter = st.sidebar.selectbox("Specialty Product", options=specialty_options)
    
    filtered_df = apply_filters(df, country_filter, corp_filter, mol_filter, sector_filter, panel_filter, year_filter, specialty_filter)
    
    tabs = st.tabs([
        "📊 Yönetici Özeti",
        "🌍 Ülke & Bölge",
        "🧬 Molekül Analizi",
        "🏢 Corporation & Rekabet",
        "💊 Specialty vs Non-Specialty",
        "💰 Fiyat & Mix",
        "📦 Pack/Strength/Size",
        "🤖 Otomatik İçgörü"
    ])
    
    with tabs[0]:
        st.header("Yönetici Özeti")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if '2022' in year_filter:
            usd_2022 = filtered_df['MAT Q3 2022 USD MNF'].sum()
            units_2022 = filtered_df['MAT Q3 2022 Units'].sum()
            su_2022 = filtered_df['MAT Q3 2022 Standard Units'].sum()
        else:
            usd_2022 = 0
            units_2022 = 0
            su_2022 = 0
        
        if '2023' in year_filter:
            usd_2023 = filtered_df['MAT Q3 2023 USD MNF'].sum()
            units_2023 = filtered_df['MAT Q3 2023 Units'].sum()
            su_2023 = filtered_df['MAT Q3 2023 Standard Units'].sum()
        else:
            usd_2023 = 0
            units_2023 = 0
            su_2023 = 0
        
        if '2024' in year_filter:
            usd_2024 = filtered_df['MAT Q3 2024 USD MNF'].sum()
            units_2024 = filtered_df['MAT Q3 2024 Units'].sum()
            su_2024 = filtered_df['MAT Q3 2024 Standard Units'].sum()
        else:
            usd_2024 = 0
            units_2024 = 0
            su_2024 = 0
        
        with col1:
            st.metric("2022 USD MNF", format_number(usd_2022))
            st.metric("2022 Units", format_units(units_2022))
        
        with col2:
            growth_22_23 = calculate_growth(usd_2022, usd_2023) if usd_2022 > 0 else 0
            st.metric("2023 USD MNF", format_number(usd_2023), f"{growth_22_23:+.1f}%")
            st.metric("2023 Units", format_units(units_2023))
        
        with col3:
            growth_23_24 = calculate_growth(usd_2023, usd_2024) if usd_2023 > 0 else 0
            st.metric("2024 USD MNF", format_number(usd_2024), f"{growth_23_24:+.1f}%")
            st.metric("2024 Units", format_units(units_2024))
        
        with col4:
            st.metric("2022 Standard Units", format_units(su_2022))
            st.metric("2023 Standard Units", format_units(su_2023))
            st.metric("2024 Standard Units", format_units(su_2024))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            years = []
            usd_values = []
            if '2022' in year_filter:
                years.append('2022')
                usd_values.append(usd_2022)
            if '2023' in year_filter:
                years.append('2023')
                usd_values.append(usd_2023)
            if '2024' in year_filter:
                years.append('2024')
                usd_values.append(usd_2024)
            
            if len(years) > 0:
                fig_usd = go.Figure()
                fig_usd.add_trace(go.Bar(
                    x=years,
                    y=usd_values,
                    marker_color='#1f77b4',
                    text=[format_number(v) for v in usd_values],
                    textposition='outside'
                ))
                fig_usd.update_layout(
                    title="USD MNF Trendi",
                    xaxis_title="Yıl",
                    yaxis_title="USD MNF",
                    height=400
                )
                st.plotly_chart(fig_usd, use_container_width=True)
        
        with col2:
            units_values = []
            su_values = []
            if '2022' in year_filter:
                units_values.append(units_2022)
                su_values.append(su_2022)
            if '2023' in year_filter:
                units_values.append(units_2023)
                su_values.append(su_2023)
            if '2024' in year_filter:
                units_values.append(units_2024)
                su_values.append(su_2024)
            
            if len(years) > 0:
                fig_volume = go.Figure()
                fig_volume.add_trace(go.Scatter(
                    x=years,
                    y=units_values,
                    mode='lines+markers',
                    name='Units',
                    line=dict(color='#ff7f0e', width=3)
                ))
                fig_volume.add_trace(go.Scatter(
                    x=years,
                    y=su_values,
                    mode='lines+markers',
                    name='Standard Units',
                    line=dict(color='#2ca02c', width=3)
                ))
                fig_volume.update_layout(
                    title="Hacim Trendi",
                    xaxis_title="Yıl",
                    yaxis_title="Birimler",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_volume, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_su_price_2022 = filtered_df['MAT Q3 2022 SU Avg Price USD MNF'].replace(0, np.nan).mean()
            avg_su_price_2023 = filtered_df['MAT Q3 2023 SU Avg Price USD MNF'].replace(0, np.nan).mean()
            avg_su_price_2024 = filtered_df['MAT Q3 2024 SU Avg Price USD MNF'].replace(0, np.nan).mean()
            
            price_years = []
            price_values = []
            if '2022' in year_filter and not np.isnan(avg_su_price_2022):
                price_years.append('2022')
                price_values.append(avg_su_price_2022)
            if '2023' in year_filter and not np.isnan(avg_su_price_2023):
                price_years.append('2023')
                price_values.append(avg_su_price_2023)
            if '2024' in year_filter and not np.isnan(avg_su_price_2024):
                price_years.append('2024')
                price_values.append(avg_su_price_2024)
            
            if len(price_years) > 0:
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(
                    x=price_years,
                    y=price_values,
                    mode='lines+markers',
                    marker=dict(size=12, color='#d62728'),
                    line=dict(width=3, color='#d62728')
                ))
                fig_price.update_layout(
                    title="SU Avg Price Trendi",
                    xaxis_title="Yıl",
                    yaxis_title="USD",
                    height=400
                )
                st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            avg_unit_price_2022 = filtered_df['MAT Q3 2022 Unit Avg Price USD MNF'].replace(0, np.nan).mean()
            avg_unit_price_2023 = filtered_df['MAT Q3 2023 Unit Avg Price USD MNF'].replace(0, np.nan).mean()
            avg_unit_price_2024 = filtered_df['MAT Q3 2024 Unit Avg Price USD MNF'].replace(0, np.nan).mean()
            
            unit_price_years = []
            unit_price_values = []
            if '2022' in year_filter and not np.isnan(avg_unit_price_2022):
                unit_price_years.append('2022')
                unit_price_values.append(avg_unit_price_2022)
            if '2023' in year_filter and not np.isnan(avg_unit_price_2023):
                unit_price_years.append('2023')
                unit_price_values.append(avg_unit_price_2023)
            if '2024' in year_filter and not np.isnan(avg_unit_price_2024):
                unit_price_years.append('2024')
                unit_price_values.append(avg_unit_price_2024)
            
            if len(unit_price_years) > 0:
                fig_unit_price = go.Figure()
                fig_unit_price.add_trace(go.Scatter(
                    x=unit_price_years,
                    y=unit_price_values,
                    mode='lines+markers',
                    marker=dict(size=12, color='#9467bd'),
                    line=dict(width=3, color='#9467bd')
                ))
                fig_unit_price.update_layout(
                    title="Unit Avg Price Trendi",
                    xaxis_title="Yıl",
                    yaxis_title="USD",
                    height=400
                )
                st.plotly_chart(fig_unit_price, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💡 Otomatik İçgörüler")
        insights = get_insights(filtered_df, year_filter)
        for idx, insight in enumerate(insights, 1):
            st.info(f"{idx}. {insight}")
    
    with tabs[1]:
        st.header("Ülke & Bölge Analizi")
        
        analysis_type = st.radio("Analiz Türü", ["Ülke", "Bölge", "Alt Bölge"], horizontal=True)
        
        if analysis_type == "Ülke":
            group_col = 'Country'
        elif analysis_type == "Bölge":
            group_col = 'Region'
        else:
            group_col = 'Sub-Region'
        
        metric_choice = st.selectbox("Metrik Seçin", ["USD MNF", "Units", "Standard Units", "SU Avg Price", "Unit Avg Price"])
        
        year_choice = st.selectbox("Yıl Seçin", ['2022', '2023', '2024'], index=2)
        
        if metric_choice == "USD MNF":
            metric_col = f'MAT Q3 {year_choice} USD MNF'
        elif metric_choice == "Units":
            metric_col = f'MAT Q3 {year_choice} Units'
        elif metric_choice == "Standard Units":
            metric_col = f'MAT Q3 {year_choice} Standard Units'
        elif metric_choice == "SU Avg Price":
            metric_col = f'MAT Q3 {year_choice} SU Avg Price USD MNF'
        else:
            metric_col = f'MAT Q3 {year_choice} Unit Avg Price USD MNF'
        
        if metric_choice in ["SU Avg Price", "Unit Avg Price"]:
            country_data = filtered_df.groupby(group_col)[[metric_col]].mean().reset_index()
        else:
            country_data = filtered_df.groupby(group_col)[[metric_col]].sum().reset_index()
        
        country_data = country_data.sort_values(metric_col, ascending=False).head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(
                country_data,
                x=group_col,
                y=metric_col,
                title=f"Top 20 {analysis_type} - {metric_choice} ({year_choice})",
                color=metric_col,
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                country_data.head(10),
                values=metric_col,
                names=group_col,
                title=f"Top 10 {analysis_type} Dağılımı - {metric_choice} ({year_choice})"
            )
            fig_pie.update_layout(height=500)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Karşılaştırmalı Analiz")
        
        comparison_data = filtered_df.groupby(group_col).agg({
            f'MAT Q3 2022 {metric_choice}': 'sum' if metric_choice in ["USD MNF", "Units", "Standard Units"] else 'mean',
            f'MAT Q3 2023 {metric_choice}': 'sum' if metric_choice in ["USD MNF", "Units", "Standard Units"] else 'mean',
            f'MAT Q3 2024 {metric_choice}': 'sum' if metric_choice in ["USD MNF", "Units", "Standard Units"] else 'mean'
        }).reset_index()
        
        comparison_data = comparison_data.sort_values(f'MAT Q3 2024 {metric_choice}', ascending=False).head(15)
        
        fig_comparison = go.Figure()
        
        if '2022' in year_filter:
            fig_comparison.add_trace(go.Bar(
                x=comparison_data[group_col],
                y=comparison_data[f'MAT Q3 2022 {metric_choice}'],
                name='2022',
                marker_color='#1f77b4'
            ))
        
        if '2023' in year_filter:
            fig_comparison.add_trace(go.Bar(
                x=comparison_data[group_col],
                y=comparison_data[f'MAT Q3 2023 {metric_choice}'],
                name='2023',
                marker_color='#ff7f0e'
            ))
        
        if '2024' in year_filter:
            fig_comparison.add_trace(go.Bar(
                x=comparison_data[group_col],
                y=comparison_data[f'MAT Q3 2024 {metric_choice}'],
                name='2024',
                marker_color='#2ca02c'
            ))
        
        fig_comparison.update_layout(
            title=f"{analysis_type} Bazlı Yıllık Karşılaştırma - {metric_choice}",
            xaxis_title=analysis_type,
            yaxis_title=metric_choice,
            barmode='group',
            height=500,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Isı Haritası")
        
        heatmap_data = filtered_df.groupby(group_col).agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum'
        }).head(15)
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=['2022', '2023', '2024'],
            y=heatmap_data.index,
            colorscale='Viridis',
            text=heatmap_data.values,
            texttemplate='%{text:.2s}',
            textfont={"size": 10}
        ))
        
        fig_heatmap.update_layout(
            title=f"{analysis_type} USD MNF Isı Haritası",
            xaxis_title="Yıl",
            yaxis_title=analysis_type,
            height=600
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tabs[2]:
        st.header("Molekül Analizi")
        
        mol_metric = st.selectbox("Molekül Metriği", ["USD MNF", "Units", "Standard Units", "SU Avg Price", "Unit Avg Price"])
        mol_year = st.selectbox("Molekül Yılı", ['2022', '2023', '2024'], index=2, key='mol_year')
        
        if mol_metric == "USD MNF":
            mol_col = f'MAT Q3 {mol_year} USD MNF'
            agg_func = 'sum'
        elif mol_metric == "Units":
            mol_col = f'MAT Q3 {mol_year} Units'
            agg_func = 'sum'
        elif mol_metric == "Standard Units":
            mol_col = f'MAT Q3 {mol_year} Standard Units'
            agg_func = 'sum'
        elif mol_metric == "SU Avg Price":
            mol_col = f'MAT Q3 {mol_year} SU Avg Price USD MNF'
            agg_func = 'mean'
        else:
            mol_col = f'MAT Q3 {mol_year} Unit Avg Price USD MNF'
            agg_func = 'mean'
        
        if agg_func == 'sum':
            molecule_data = filtered_df.groupby('Molecule')[[mol_col]].sum().reset_index()
        else:
            molecule_data = filtered_df.groupby('Molecule')[[mol_col]].mean().reset_index()
        
        molecule_data = molecule_data.sort_values(mol_col, ascending=False).head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_mol_bar = px.bar(
                molecule_data,
                x='Molecule',
                y=mol_col,
                title=f"Top 20 Molekül - {mol_metric} ({mol_year})",
                color=mol_col,
                color_continuous_scale='Reds'
            )
            fig_mol_bar.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig_mol_bar, use_container_width=True)
        
        with col2:
            fig_mol_treemap = px.treemap(
                molecule_data.head(15),
                path=['Molecule'],
                values=mol_col,
                title=f"Molekül Treemap - {mol_metric} ({mol_year})"
            )
            fig_mol_treemap.update_layout(height=500)
            st.plotly_chart(fig_mol_treemap, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Molekül Trend Analizi")
        
        mol_trend_data = filtered_df.groupby('Molecule').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum'
        }).reset_index()
        
        mol_trend_data = mol_trend_data.sort_values('MAT Q3 2024 USD MNF', ascending=False).head(10)
        
        fig_mol_trend = go.Figure()
        
        for idx, row in mol_trend_data.iterrows():
            years = []
            values = []
            if '2022' in year_filter:
                years.append('2022')
                values.append(row['MAT Q3 2022 USD MNF'])
            if '2023' in year_filter:
                years.append('2023')
                values.append(row['MAT Q3 2023 USD MNF'])
            if '2024' in year_filter:
                years.append('2024')
                values.append(row['MAT Q3 2024 USD MNF'])
            
            fig_mol_trend.add_trace(go.Scatter(
                x=years,
                y=values,
                mode='lines+markers',
                name=row['Molecule'],
                line=dict(width=2)
            ))
        
        fig_mol_trend.update_layout(
            title="Top 10 Molekül USD MNF Trendi",
            xaxis_title="Yıl",
            yaxis_title="USD MNF",
            height=500,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        st.plotly_chart(fig_mol_trend, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Molekül Fiyat Analizi")
        
        mol_price_data = filtered_df.groupby('Molecule').agg({
            'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2023 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2024 SU Avg Price USD MNF': 'mean'
        }).reset_index()
        
        mol_price_data = mol_price_data.sort_values('MAT Q3 2024 SU Avg Price USD MNF', ascending=False).head(15)
        
        fig_mol_price = go.Figure()
        
        if '2022' in year_filter:
            fig_mol_price.add_trace(go.Bar(
                x=mol_price_data['Molecule'],
                y=mol_price_data['MAT Q3 2022 SU Avg Price USD MNF'],
                name='2022',
                marker_color='#8c564b'
            ))
        
        if '2023' in year_filter:
            fig_mol_price.add_trace(go.Bar(
                x=mol_price_data['Molecule'],
                y=mol_price_data['MAT Q3 2023 SU Avg Price USD MNF'],
                name='2023',
                marker_color='#e377c2'
            ))
        
        if '2024' in year_filter:
            fig_mol_price.add_trace(go.Bar(
                x=mol_price_data['Molecule'],
                y=mol_price_data['MAT Q3 2024 SU Avg Price USD MNF'],
                name='2024',
                marker_color='#7f7f7f'
            ))
        
        fig_mol_price.update_layout(
            title="Molekül SU Avg Price Karşılaştırma",
            xaxis_title="Molekül",
            yaxis_title="SU Avg Price USD",
            barmode='group',
            height=500,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_mol_price, use_container_width=True)
    
    with tabs[3]:
        st.header("Corporation & Rekabet Analizi")
        
        st.subheader("Pazar Payı Analizi")
        
        corp_year = st.selectbox("Corporation Yılı", ['2022', '2023', '2024'], index=2, key='corp_year')
        
        corp_data = filtered_df.groupby('Corporation')[[f'MAT Q3 {corp_year} USD MNF']].sum().reset_index()
        corp_data = corp_data.sort_values(f'MAT Q3 {corp_year} USD MNF', ascending=False).head(15)
        
        total_market = corp_data[f'MAT Q3 {corp_year} USD MNF'].sum()
        corp_data['Market Share %'] = (corp_data[f'MAT Q3 {corp_year} USD MNF'] / total_market * 100).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_corp_bar = px.bar(
                corp_data,
                x='Corporation',
                y=f'MAT Q3 {corp_year} USD MNF',
                title=f"Corporation USD MNF ({corp_year})",
                color='Market Share %',
                color_continuous_scale='Greens',
                text='Market Share %'
            )
            fig_corp_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_corp_bar.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig_corp_bar, use_container_width=True)
        
        with col2:
            fig_corp_pie = px.pie(
                corp_data.head(10),
                values=f'MAT Q3 {corp_year} USD MNF',
                names='Corporation',
                title=f"Top 10 Corporation Pazar Payı ({corp_year})"
            )
            fig_corp_pie.update_layout(height=500)
            st.plotly_chart(fig_corp_pie, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Corporation Büyüme Analizi")
        
        corp_growth_data = filtered_df.groupby('Corporation').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum'
        }).reset_index()
        
        corp_growth_data['Growth 22-23 %'] = corp_growth_data.apply(
            lambda x: calculate_growth(x['MAT Q3 2022 USD MNF'], x['MAT Q3 2023 USD MNF']), axis=1
        )
        corp_growth_data['Growth 23-24 %'] = corp_growth_data.apply(
            lambda x: calculate_growth(x['MAT Q3 2023 USD MNF'], x['MAT Q3 2024 USD MNF']), axis=1
        )
        
        corp_growth_data = corp_growth_data.sort_values('MAT Q3 2024 USD MNF', ascending=False).head(15)
        
        fig_corp_growth = go.Figure()
        
        fig_corp_growth.add_trace(go.Bar(
            x=corp_growth_data['Corporation'],
            y=corp_growth_data['Growth 22-23 %'],
            name='2022-2023',
            marker_color='#17becf'
        ))
        
        fig_corp_growth.add_trace(go.Bar(
            x=corp_growth_data['Corporation'],
            y=corp_growth_data['Growth 23-24 %'],
            name='2023-2024',
            marker_color='#bcbd22'
        ))
        
        fig_corp_growth.update_layout(
            title="Corporation Büyüme Oranları",
            xaxis_title="Corporation",
            yaxis_title="Büyüme %",
            barmode='group',
            height=500,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_corp_growth, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Molekül Bazlı Rekabet Analizi")
        
        selected_molecule = st.selectbox(
            "Molekül Seçin",
            options=sorted(filtered_df['Molecule'].dropna().unique())
        )
        
        if selected_molecule:
            mol_corp_data = filtered_df[filtered_df['Molecule'] == selected_molecule].groupby('Corporation').agg({
                f'MAT Q3 {corp_year} USD MNF': 'sum',
                f'MAT Q3 {corp_year} SU Avg Price USD MNF': 'mean',
                f'MAT Q3 {corp_year} Unit Avg Price USD MNF': 'mean'
            }).reset_index()
            
            mol_corp_data = mol_corp_data.sort_values(f'MAT Q3 {corp_year} USD MNF', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_mol_corp_sales = px.bar(
                    mol_corp_data,
                    x='Corporation',
                    y=f'MAT Q3 {corp_year} USD MNF',
                    title=f"{selected_molecule} - Corporation Satışları ({corp_year})",
                    color=f'MAT Q3 {corp_year} USD MNF',
                    color_continuous_scale='Oranges'
                )
                fig_mol_corp_sales.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_mol_corp_sales, use_container_width=True)
            
            with col2:
                fig_mol_corp_price = px.bar(
                    mol_corp_data,
                    x='Corporation',
                    y=f'MAT Q3 {corp_year} SU Avg Price USD MNF',
                    title=f"{selected_molecule} - Corporation SU Avg Price ({corp_year})",
                    color=f'MAT Q3 {corp_year} SU Avg Price USD MNF',
                    color_continuous_scale='Purples'
                )
                fig_mol_corp_price.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_mol_corp_price, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Pazar Payı Değişimi")
        
        corp_share_22 = filtered_df.groupby('Corporation')[['MAT Q3 2022 USD MNF']].sum()
        corp_share_24 = filtered_df.groupby('Corporation')[['MAT Q3 2024 USD MNF']].sum()
        
        total_22 = corp_share_22['MAT Q3 2022 USD MNF'].sum()
        total_24 = corp_share_24['MAT Q3 2024 USD MNF'].sum()
        
        corp_share_change = pd.DataFrame({
            'Corporation': corp_share_22.index,
            'Share 2022 %': (corp_share_22['MAT Q3 2022 USD MNF'] / total_22 * 100).values if total_22 > 0 else 0,
            'Share 2024 %': (corp_share_24['MAT Q3 2024 USD MNF'] / total_24 * 100).values if total_24 > 0 else 0
        })
        
        corp_share_change['Share Change'] = corp_share_change['Share 2024 %'] - corp_share_change['Share 2022 %']
        corp_share_change = corp_share_change.sort_values('Share 2024 %', ascending=False).head(15)
        
        fig_share_change = go.Figure()
        
        fig_share_change.add_trace(go.Scatter(
            x=corp_share_change['Corporation'],
            y=corp_share_change['Share 2022 %'],
            mode='markers',
            name='2022',
            marker=dict(size=12, color='#1f77b4')
        ))
        
        fig_share_change.add_trace(go.Scatter(
            x=corp_share_change['Corporation'],
            y=corp_share_change['Share 2024 %'],
            mode='markers',
            name='2024',
            marker=dict(size=12, color='#ff7f0e')
        ))
        
        for idx, row in corp_share_change.iterrows():
            fig_share_change.add_trace(go.Scatter(
                x=[row['Corporation'], row['Corporation']],
                y=[row['Share 2022 %'], row['Share 2024 %']],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ))
        
        fig_share_change.update_layout(
            title="Pazar Payı Değişimi 2022-2024",
            xaxis_title="Corporation",
            yaxis_title="Pazar Payı %",
            height=500,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_share_change, use_container_width=True)
    
    with tabs[4]:
        st.header("Specialty vs Non-Specialty Analizi")
        
        spec_year = st.selectbox("Specialty Yılı", ['2022', '2023', '2024'], index=2, key='spec_year')
        
        specialty_data = filtered_df.groupby('Specialty Product').agg({
            f'MAT Q3 {spec_year} USD MNF': 'sum',
            f'MAT Q3 {spec_year} Units': 'sum',
            f'MAT Q3 {spec_year} Standard Units': 'sum',
            f'MAT Q3 {spec_year} SU Avg Price USD MNF': 'mean',
            f'MAT Q3 {spec_year} Unit Avg Price USD MNF': 'mean'
        }).reset_index()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_spec_usd = px.pie(
                specialty_data,
                values=f'MAT Q3 {spec_year} USD MNF',
                names='Specialty Product',
                title=f"USD MNF Dağılımı ({spec_year})",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig_spec_usd.update_layout(height=400)
            st.plotly_chart(fig_spec_usd, use_container_width=True)
        
        with col2:
            fig_spec_units = px.pie(
                specialty_data,
                values=f'MAT Q3 {spec_year} Units',
                names='Specialty Product',
                title=f"Units Dağılımı ({spec_year})",
                color_discrete_sequence=px.colors.sequential.Teal
            )
            fig_spec_units.update_layout(height=400)
            st.plotly_chart(fig_spec_units, use_container_width=True)
        
        with col3:
            fig_spec_su = px.pie(
                specialty_data,
                values=f'MAT Q3 {spec_year} Standard Units',
                names='Specialty Product',
                title=f"Standard Units Dağılımı ({spec_year})",
                color_discrete_sequence=px.colors.sequential.Magma
            )
            fig_spec_su.update_layout(height=400)
            st.plotly_chart(fig_spec_su, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Fiyat Karşılaştırması")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_spec_su_price = px.bar(
                specialty_data,
                x='Specialty Product',
                y=f'MAT Q3 {spec_year} SU Avg Price USD MNF',
                title=f"SU Avg Price Karşılaştırma ({spec_year})",
                color=f'MAT Q3 {spec_year} SU Avg Price USD MNF',
                color_continuous_scale='Viridis',
                text=f'MAT Q3 {spec_year} SU Avg Price USD MNF'
            )
            fig_spec_su_price.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
            fig_spec_su_price.update_layout(height=400)
            st.plotly_chart(fig_spec_su_price, use_container_width=True)
        
        with col2:
            fig_spec_unit_price = px.bar(
                specialty_data,
                x='Specialty Product',
                y=f'MAT Q3 {spec_year} Unit Avg Price USD MNF',
                title=f"Unit Avg Price Karşılaştırma ({spec_year})",
                color=f'MAT Q3 {spec_year} Unit Avg Price USD MNF',
                color_continuous_scale='Plasma',
                text=f'MAT Q3 {spec_year} Unit Avg Price USD MNF'
            )
            fig_spec_unit_price.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
            fig_spec_unit_price.update_layout(height=400)
            st.plotly_chart(fig_spec_unit_price, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Trend Analizi")
        
        specialty_trend = filtered_df.groupby('Specialty Product').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum'
        }).reset_index()
        
        fig_spec_trend = go.Figure()
        
        for spec_type in specialty_trend['Specialty Product'].unique():
            spec_row = specialty_trend[specialty_trend['Specialty Product'] == spec_type]
            years = []
            values = []
            if '2022' in year_filter:
                years.append('2022')
                values.append(spec_row['MAT Q3 2022 USD MNF'].values[0])
            if '2023' in year_filter:
                years.append('2023')
                values.append(spec_row['MAT Q3 2023 USD MNF'].values[0])
            if '2024' in year_filter:
                years.append('2024')
                values.append(spec_row['MAT Q3 2024 USD MNF'].values[0])
            
            fig_spec_trend.add_trace(go.Scatter(
                x=years,
                y=values,
                mode='lines+markers',
                name=spec_type,
                line=dict(width=3),
                marker=dict(size=10)
            ))
        
        fig_spec_trend.update_layout(
            title="Specialty vs Non-Specialty USD MNF Trendi",
            xaxis_title="Yıl",
            yaxis_title="USD MNF",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_spec_trend, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Ülke Bazlı Specialty Analizi")
        
        country_spec = filtered_df.groupby(['Country', 'Specialty Product'])[[f'MAT Q3 {spec_year} USD MNF']].sum().reset_index()
        country_spec_pivot = country_spec.pivot(index='Country', columns='Specialty Product', values=f'MAT Q3 {spec_year} USD MNF').fillna(0)
        country_spec_pivot = country_spec_pivot.head(15)
        
        fig_country_spec = go.Figure()
        
        for col in country_spec_pivot.columns:
            fig_country_spec.add_trace(go.Bar(
                x=country_spec_pivot.index,
                y=country_spec_pivot[col],
                name=col
            ))
        
        fig_country_spec.update_layout(
            title=f"Ülke Bazlı Specialty Dağılımı ({spec_year})",
            xaxis_title="Country",
            yaxis_title="USD MNF",
            barmode='stack',
            height=500,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_country_spec, use_container_width=True)
    
    with tabs[5]:
        st.header("Fiyat & Mix Analizi")
        
        st.subheader("Fiyat vs Hacim Değişimi")
        
        price_volume_data = filtered_df.groupby('Molecule').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2022 Units': 'sum',
            'MAT Q3 2024 Units': 'sum',
            'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2024 SU Avg Price USD MNF': 'mean'
        }).reset_index()
        
        price_volume_data['Volume Change %'] = price_volume_data.apply(
            lambda x: calculate_growth(x['MAT Q3 2022 Units'], x['MAT Q3 2024 Units']), axis=1
        )
        price_volume_data['Price Change %'] = price_volume_data.apply(
            lambda x: calculate_growth(x['MAT Q3 2022 SU Avg Price USD MNF'], x['MAT Q3 2024 SU Avg Price USD MNF']), axis=1
        )
        price_volume_data['Value Change %'] = price_volume_data.apply(
            lambda x: calculate_growth(x['MAT Q3 2022 USD MNF'], x['MAT Q3 2024 USD MNF']), axis=1
        )
        
        price_volume_data = price_volume_data[price_volume_data['MAT Q3 2024 USD MNF'] > 0]
        price_volume_data = price_volume_data.sort_values('MAT Q3 2024 USD MNF', ascending=False).head(30)
        
        fig_pv_scatter = px.scatter(
            price_volume_data,
            x='Volume Change %',
            y='Price Change %',
            size='MAT Q3 2024 USD MNF',
            color='Value Change %',
            hover_data=['Molecule'],
            title="Fiyat vs Hacim Değişimi (2022-2024)",
            color_continuous_scale='RdYlGn'
        )
        
        fig_pv_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_pv_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
        
        fig_pv_scatter.update_layout(
            xaxis_title="Hacim Değişimi %",
            yaxis_title="Fiyat Değişimi %",
            height=600
        )
        st.plotly_chart(fig_pv_scatter, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Waterfall Analizi")
        
        total_2022 = filtered_df['MAT Q3 2022 USD MNF'].sum()
        total_2024 = filtered_df['MAT Q3 2024 USD MNF'].sum()
        
        volume_impact = (filtered_df['MAT Q3 2024 Units'].sum() - filtered_df['MAT Q3 2022 Units'].sum()) * filtered_df['MAT Q3 2022 SU Avg Price USD MNF'].mean()
        price_impact = total_2024 - total_2022 - volume_impact
        
        waterfall_data = pd.DataFrame({
            'Category': ['2022 USD MNF', 'Hacim Etkisi', 'Fiyat Etkisi', '2024 USD MNF'],
            'Value': [total_2022, volume_impact, price_impact, total_2024],
            'Measure': ['absolute', 'relative', 'relative', 'total']
        })
        
        fig_waterfall = go.Figure(go.Waterfall(
            x=waterfall_data['Category'],
            y=waterfall_data['Value'],
            measure=waterfall_data['Measure'],
            text=[format_number(v) for v in waterfall_data['Value']],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#d62728"}},
            increasing={"marker": {"color": "#2ca02c"}},
            totals={"marker": {"color": "#1f77b4"}}
        ))
        
        fig_waterfall.update_layout(
            title="USD MNF Değişimi Waterfall (2022-2024)",
            showlegend=False,
            height=500
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Ülke Bazlı Fiyat-Hacim Matrisi")
        
        country_pv_data = filtered_df.groupby('Country').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2022 Units': 'sum',
            'MAT Q3 2024 Units': 'sum',
            'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2024 SU Avg Price USD MNF': 'mean'
        }).reset_index()
        
        country_pv_data['Volume Change %'] = country_pv_data.apply(
            lambda x: calculate_growth(x['MAT Q3 2022 Units'], x['MAT Q3 2024 Units']), axis=1
        )
        country_pv_data['Price Change %'] = country_pv_data.apply(
            lambda x: calculate_growth(x['MAT Q3 2022 SU Avg Price USD MNF'], x['MAT Q3 2024 SU Avg Price USD MNF']), axis=1
        )
        
        country_pv_data = country_pv_data.sort_values('MAT Q3 2024 USD MNF', ascending=False).head(20)
        
        fig_country_pv = px.scatter(
            country_pv_data,
            x='Volume Change %',
            y='Price Change %',
            size='MAT Q3 2024 USD MNF',
            color='Country',
            hover_data=['Country'],
            title="Ülke Bazlı Fiyat-Hacim Matrisi (2022-2024)"
        )
        
        fig_country_pv.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_country_pv.add_vline(x=0, line_dash="dash", line_color="gray")
        
        fig_country_pv.update_layout(
            xaxis_title="Hacim Değişimi %",
            yaxis_title="Fiyat Değişimi %",
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig_country_pv, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Corporation Fiyat-Mix Analizi")
        
        corp_pm_data = filtered_df.groupby('Corporation').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2024 SU Avg Price USD MNF': 'mean'
        }).reset_index()
        
        corp_pm_data['Value Change %'] = corp_pm_data.apply(
            lambda x: calculate_growth(x['MAT Q3 2022 USD MNF'], x['MAT Q3 2024 USD MNF']), axis=1
        )
        corp_pm_data['Price Change %'] = corp_pm_data.apply(
            lambda x: calculate_growth(x['MAT Q3 2022 SU Avg Price USD MNF'], x['MAT Q3 2024 SU Avg Price USD MNF']), axis=1
        )
        
        corp_pm_data = corp_pm_data.sort_values('MAT Q3 2024 USD MNF', ascending=False).head(15)
        
        fig_corp_pm = go.Figure()
        
        fig_corp_pm.add_trace(go.Bar(
            x=corp_pm_data['Corporation'],
            y=corp_pm_data['Value Change %'],
            name='Value Change %',
            marker_color='#1f77b4'
        ))
        
        fig_corp_pm.add_trace(go.Scatter(
            x=corp_pm_data['Corporation'],
            y=corp_pm_data['Price Change %'],
            name='Price Change %',
            mode='markers',
            marker=dict(size=12, color='#ff7f0e', symbol='diamond')
        ))
        
        fig_corp_pm.update_layout(
            title="Corporation Value vs Price Change (2022-2024)",
            xaxis_title="Corporation",
            yaxis_title="Change %",
            height=500,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_corp_pm, use_container_width=True)
    
    with tabs[6]:
        st.header("Pack / Strength / Size Analizi")
        
        pack_analysis_type = st.selectbox("Analiz Tipi", ["International Pack", "International Strength", "International Size"])
        pack_year = st.selectbox("Pack Yılı", ['2022', '2023', '2024'], index=2, key='pack_year')
        
        if pack_analysis_type == "International Pack":
            group_col = 'International Pack'
        elif pack_analysis_type == "International Strength":
            group_col = 'International Strength'
        else:
            group_col = 'International Size'
        
        pack_data = filtered_df.groupby(group_col).agg({
            f'MAT Q3 {pack_year} USD MNF': 'sum',
            f'MAT Q3 {pack_year} Units': 'sum',
            f'MAT Q3 {pack_year} Standard Units': 'sum',
            f'MAT Q3 {pack_year} SU Avg Price USD MNF': 'mean',
            f'MAT Q3 {pack_year} Unit Avg Price USD MNF': 'mean'
        }).reset_index()
        
        pack_data = pack_data.sort_values(f'MAT Q3 {pack_year} USD MNF', ascending=False).head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pack_usd = px.bar(
                pack_data,
                x=group_col,
                y=f'MAT Q3 {pack_year} USD MNF',
                title=f"{pack_analysis_type} USD MNF ({pack_year})",
                color=f'MAT Q3 {pack_year} USD MNF',
                color_continuous_scale='Blues'
            )
            fig_pack_usd.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig_pack_usd, use_container_width=True)
        
        with col2:
            fig_pack_units = px.bar(
                pack_data,
                x=group_col,
                y=f'MAT Q3 {pack_year} Units',
                title=f"{pack_analysis_type} Units ({pack_year})",
                color=f'MAT Q3 {pack_year} Units',
                color_continuous_scale='Greens'
            )
            fig_pack_units.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig_pack_units, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Fiyat Analizi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pack_su_price = px.bar(
                pack_data,
                x=group_col,
                y=f'MAT Q3 {pack_year} SU Avg Price USD MNF',
                title=f"{pack_analysis_type} SU Avg Price ({pack_year})",
                color=f'MAT Q3 {pack_year} SU Avg Price USD MNF',
                color_continuous_scale='Reds'
            )
            fig_pack_su_price.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig_pack_su_price, use_container_width=True)
        
        with col2:
            fig_pack_unit_price = px.bar(
                pack_data,
                x=group_col,
                y=f'MAT Q3 {pack_year} Unit Avg Price USD MNF',
                title=f"{pack_analysis_type} Unit Avg Price ({pack_year})",
                color=f'MAT Q3 {pack_year} Unit Avg Price USD MNF',
                color_continuous_scale='Purples'
            )
            fig_pack_unit_price.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig_pack_unit_price, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Trend Analizi")
        
        pack_trend_data = filtered_df.groupby(group_col).agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum'
        }).reset_index()
        
        pack_trend_data = pack_trend_data.sort_values('MAT Q3 2024 USD MNF', ascending=False).head(10)
        
        fig_pack_trend = go.Figure()
        
        for idx, row in pack_trend_data.iterrows():
            years = []
            values = []
            if '2022' in year_filter:
                years.append('2022')
                values.append(row['MAT Q3 2022 USD MNF'])
            if '2023' in year_filter:
                years.append('2023')
                values.append(row['MAT Q3 2023 USD MNF'])
            if '2024' in year_filter:
                years.append('2024')
                values.append(row['MAT Q3 2024 USD MNF'])
            
            fig_pack_trend.add_trace(go.Scatter(
                x=years,
                y=values,
                mode='lines+markers',
                name=str(row[group_col]),
                line=dict(width=2)
            ))
        
        fig_pack_trend.update_layout(
            title=f"Top 10 {pack_analysis_type} USD MNF Trendi",
            xaxis_title="Yıl",
            yaxis_title="USD MNF",
            height=500,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        st.plotly_chart(fig_pack_trend, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Kombinasyon Analizi")
        
        combo_data = filtered_df.groupby(['International Pack', 'International Strength']).agg({
            f'MAT Q3 {pack_year} USD MNF': 'sum'
        }).reset_index()
        
        combo_data = combo_data.sort_values(f'MAT Q3 {pack_year} USD MNF', ascending=False).head(20)
        
        fig_combo = px.treemap(
            combo_data,
            path=['International Pack', 'International Strength'],
            values=f'MAT Q3 {pack_year} USD MNF',
            title=f"Pack & Strength Kombinasyonu Treemap ({pack_year})"
        )
        fig_combo.update_layout(height=600)
        st.plotly_chart(fig_combo, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Size & Volume Analizi")
        
        size_vol_data = filtered_df.groupby('International Size').agg({
            f'MAT Q3 {pack_year} USD MNF': 'sum',
            f'MAT Q3 {pack_year} Units': 'sum',
            f'MAT Q3 {pack_year} SU Avg Price USD MNF': 'mean'
        }).reset_index()
        
        size_vol_data = size_vol_data.sort_values(f'MAT Q3 {pack_year} USD MNF', ascending=False).head(15)
        
        fig_size_vol = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_size_vol.add_trace(
            go.Bar(
                x=size_vol_data['International Size'],
                y=size_vol_data[f'MAT Q3 {pack_year} USD MNF'],
                name="USD MNF",
                marker_color='#1f77b4'
            ),
            secondary_y=False
        )
        
        fig_size_vol.add_trace(
            go.Scatter(
                x=size_vol_data['International Size'],
                y=size_vol_data[f'MAT Q3 {pack_year} SU Avg Price USD MNF'],
                name="SU Avg Price",
                mode='lines+markers',
                marker=dict(size=10, color='#ff7f0e'),
                line=dict(width=3, color='#ff7f0e')
            ),
            secondary_y=True
        )
        
        fig_size_vol.update_layout(
            title=f"Size USD MNF & SU Avg Price ({pack_year})",
            xaxis_tickangle=-45,
            height=500
        )
        fig_size_vol.update_xaxis(title_text="International Size")
        fig_size_vol.update_yaxis(title_text="USD MNF", secondary_y=False)
        fig_size_vol.update_yaxis(title_text="SU Avg Price", secondary_y=True)
        
        st.plotly_chart(fig_size_vol, use_container_width=True)
    
    with tabs[7]:
        st.header("Otomatik İçgörü Motoru")
        
        insights = get_insights(filtered_df, year_filter)
        
        st.subheader("📈 Temel İçgörüler")
        
        for idx, insight in enumerate(insights, 1):
            st.success(f"**{idx}.** {insight}")
        
        st.markdown("---")
        st.subheader("🎯 Detaylı Analizler")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Büyüme Liderleri")
            
            if '2023' in year_filter and '2024' in year_filter:
                mol_growth = filtered_df.groupby('Molecule').agg({
                    'MAT Q3 2023 USD MNF': 'sum',
                    'MAT Q3 2024 USD MNF': 'sum'
                }).reset_index()
                
                mol_growth['Growth %'] = mol_growth.apply(
                    lambda x: calculate_growth(x['MAT Q3 2023 USD MNF'], x['MAT Q3 2024 USD MNF']), axis=1
                )
                
                mol_growth = mol_growth[mol_growth['MAT Q3 2024 USD MNF'] > 100000]
                top_growers = mol_growth.sort_values('Growth %', ascending=False).head(5)
                
                for idx, row in top_growers.iterrows():
                    st.info(f"**{row['Molecule']}** %{row['Growth %']:.1f} büyüme ile öne çıkıyor.")
        
        with col2:
            st.markdown("### Fiyat Değişimleri")
            
            if '2022' in year_filter and '2024' in year_filter:
                price_change_data = filtered_df.groupby('Molecule').agg({
                    'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
                    'MAT Q3 2024 SU Avg Price USD MNF': 'mean',
                    'MAT Q3 2024 USD MNF': 'sum'
                }).reset_index()
                
                price_change_data['Price Change %'] = price_change_data.apply(
                    lambda x: calculate_growth(x['MAT Q3 2022 SU Avg Price USD MNF'], x['MAT Q3 2024 SU Avg Price USD MNF']), axis=1
                )
                
                price_change_data = price_change_data[price_change_data['MAT Q3 2024 USD MNF'] > 100000]
                top_price_changes = price_change_data.sort_values('Price Change %', ascending=False).head(5)
                
                for idx, row in top_price_changes.iterrows():
                    st.warning(f"**{row['Molecule']}** fiyatı %{row['Price Change %']:.1f} değişti.")
        
        st.markdown("---")
        st.subheader("🌍 Coğrafi İçgörüler")
        
        if '2022' in year_filter and '2024' in year_filter:
            country_growth = filtered_df.groupby('Country').agg({
                'MAT Q3 2022 USD MNF': 'sum',
                'MAT Q3 2024 USD MNF': 'sum'
            }).reset_index()
            
            country_growth['Growth %'] = country_growth.apply(
                lambda x: calculate_growth(x['MAT Q3 2022 USD MNF'], x['MAT Q3 2024 USD MNF']), axis=1
            )
            
            country_growth = country_growth[country_growth['MAT Q3 2024 USD MNF'] > 500000]
            
            top_country_growers = country_growth.sort_values('Growth %', ascending=False).head(5)
            bottom_country_growers = country_growth.sort_values('Growth %', ascending=True).head(5)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### En Hızlı Büyüyen Ülkeler")
                for idx, row in top_country_growers.iterrows():
                    st.success(f"**{row['Country']}**: %{row['Growth %']:.1f}")
            
            with col2:
                st.markdown("#### Zorlu Pazarlar")
                for idx, row in bottom_country_growers.iterrows():
                    st.error(f"**{row['Country']}**: %{row['Growth %']:.1f}")
        
        st.markdown("---")
        st.subheader("🏢 Rekabet İçgörüleri")
        
        if '2023' in year_filter and '2024' in year_filter:
            corp_performance = filtered_df.groupby('Corporation').agg({
                'MAT Q3 2023 USD MNF': 'sum',
                'MAT Q3 2024 USD MNF': 'sum'
            }).reset_index()
            
            corp_performance['Growth %'] = corp_performance.apply(
                lambda x: calculate_growth(x['MAT Q3 2023 USD MNF'], x['MAT Q3 2024 USD MNF']), axis=1
            )
            
            corp_performance = corp_performance[corp_performance['MAT Q3 2024 USD MNF'] > 1000000]
            corp_performance = corp_performance.sort_values('Growth %', ascending=False)
            
            if len(corp_performance) > 0:
                st.info(f"**{corp_performance.iloc[0]['Corporation']}** şirketi %{corp_performance.iloc[0]['Growth %']:.1f} büyüme ile rakiplerini geride bırakıyor.")
                
                if len(corp_performance) > 1:
                    st.warning(f"**{corp_performance.iloc[-1]['Corporation']}** şirketi %{corp_performance.iloc[-1]['Growth %']:.1f} performans ile dikkat çekiyor.")
        
        st.markdown("---")
        st.subheader("💊 Ürün Portföyü İçgörüleri")
        
        specialty_insights = []
        
        if '2024' in year_filter:
            specialty_sales = filtered_df[filtered_df['Specialty Product'] == 'Yes']['MAT Q3 2024 USD MNF'].sum()
            total_sales = filtered_df['MAT Q3 2024 USD MNF'].sum()
            
            if total_sales > 0:
                specialty_pct = (specialty_sales / total_sales) * 100
                specialty_insights.append(f"Specialty ürünler portföyün %{specialty_pct:.1f}'ini oluşturuyor.")
            
            specialty_avg_price = filtered_df[filtered_df['Specialty Product'] == 'Yes']['MAT Q3 2024 SU Avg Price USD MNF'].mean()
            non_specialty_avg_price = filtered_df[filtered_df['Specialty Product'] != 'Yes']['MAT Q3 2024 SU Avg Price USD MNF'].mean()
            
            if specialty_avg_price > 0 and non_specialty_avg_price > 0:
                price_premium = ((specialty_avg_price - non_specialty_avg_price) / non_specialty_avg_price) * 100
                specialty_insights.append(f"Specialty ürünler ortalama %{price_premium:.1f} fiyat primi taşıyor.")
        
        for insight in specialty_insights:
            st.info(insight)
        
        st.markdown("---")
        st.subheader("📊 Özet Kart")
        
        summary_metrics = {
            "Toplam Ülke": filtered_df['Country'].nunique(),
            "Toplam Molekül": filtered_df['Molecule'].nunique(),
            "Toplam Corporation": filtered_df['Corporation'].nunique(),
            "Toplam Ürün": filtered_df['International Product'].nunique()
        }
        
        cols = st.columns(4)
        for idx, (key, value) in enumerate(summary_metrics.items()):
            cols[idx].metric(key, value)

else:
    st.info("Lütfen Excel dosyasını yükleyin.")
    st.markdown("""
    ### Beklenen Kolon Yapısı:
    
    **Tanımlayıcı Kolonlar:**
    - Source.Name, Country, Sector, Panel, Region, Sub-Region
    - Corporation, Manufacturer, Molecule List, Molecule, Chemical Salt
    - International Product, Specialty Product, NFC123
    - International Pack, International Strength, International Size
    - International Volume, International Prescription
    
    **Metrik Kolonlar (her yıl için):**
    - MAT Q3 YYYY USD MNF
    - MAT Q3 YYYY Standard Units
    - MAT Q3 YYYY Units
    - MAT Q3 YYYY SU Avg Price USD MNF
    - MAT Q3 YYYY Unit Avg Price USD MNF
    
    (YYYY: 2022, 2023, 2024)
    """)
```
```
