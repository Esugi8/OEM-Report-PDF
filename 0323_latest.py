import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Optional
import io

# =================================================================
# 1. データ構造定義 (Billion JPY / k units)
# =================================================================

class RegionalSales(BaseModel):
    japan: float
    north_america: float
    europe: float
    asia_incl_china: float
    other: float

class FinancialMetrics(BaseModel):
    revenue: float               # Target: Billion JPY (10億円)
    operating_income: float      # Target: Billion JPY (10億円)
    operating_margin_pct: float  # %
    volume: float                # Target: k units (千台)
    fx_usd: float
    regional_sales: RegionalSales

class ReportSchema(BaseModel):
    company_name: str
    prior_h1_actual: FinancialMetrics
    h1_actual: FinancialMetrics
    full_year_forecast: FinancialMetrics

# API Client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def clean_numeric_value(val):
    if val is None: return 0.0
    try:
        if isinstance(val, str):
            val = val.replace(',', '').replace('¥', '').replace(' ', '')
        return float(val)
    except ValueError:
        return 0.0

# =================================================================
# 2. 解析ロジック (単位換算をAIに明示)
# =================================================================

def process_pdf(uploaded_file):
    file_bytes = uploaded_file.read()
    gemini_file = client.files.upload(
        file=io.BytesIO(file_bytes), 
        config={'mime_type': 'application/pdf'}
    )

    prompt = """
    Extract financial results and convert to specified units.
    【CONVERSION】
    - Revenue & Income: Convert to "Billion JPY" (10億円). 
      (Example: 24.6 Trillion -> 24600 | 200,560 Million -> 200.56)
    - Volumes: Convert to "k units" (千台).
      (Example: 4,783,000 -> 4783)
    【RULES】
    - Use ONLY "Consolidated" figures for volume.
    - Treat comma (,) as thousands separator.
    """

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=[gemini_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReportSchema,
            temperature=0.0,
        ),
    )
    client.files.delete(name=gemini_file.name)
    return response.parsed

# =================================================================
# 3. UI部
# =================================================================

st.set_page_config(page_title="OEM Financial Analyser", layout="wide")
st.title("🚗 Automotive OEM Consolidated Analysis")

with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    show_charts = st.checkbox("Show Visual Charts", value=True)
    analyze_btn = st.button("🚀 Analyze Data")

if analyze_btn and uploaded_files:
    all_rows = []
    for f in uploaded_files:
        try:
            data = process_pdf(f)
            for p_key in ['prior_h1_actual', 'h1_actual', 'full_year_forecast']:
                m = getattr(data, p_key)
                reg = m.regional_sales
                all_rows.append({
                    "Company": data.company_name,
                    "Period": "Last Year (H1)" if p_key == 'prior_h1_actual' else ("Current (H1)" if p_key == 'h1_actual' else "FY Forecast"),
                    "Revenue": clean_numeric_value(m.revenue),
                    "OpIncome": clean_numeric_value(m.operating_income),
                    "Margin": clean_numeric_value(m.operating_margin_pct),
                    "Volume": clean_numeric_value(m.volume),
                    "Japan": clean_numeric_value(reg.japan), 
                    "NA": clean_numeric_value(reg.north_america), 
                    "Asia": clean_numeric_value(reg.asia_incl_china)
                })
        except Exception as e:
            st.error(f"Error in {f.name}: {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        
        # 収益順ソート
        ranking = df[df["Period"] == "Current (H1)"].sort_values("Revenue", ascending=False)["Company"].tolist()
        df['Company'] = pd.Categorical(df['Company'], categories=ranking, ordered=True)
        df = df.sort_values(["Company", "Period"])

        # テーブル表示 (会社名連結)
        st.subheader("📊 Performance Table (Billion JPY / k units)")
        display_df = df.copy()
        display_df['Company_Display'] = display_df['Company'].astype(str).where(~display_df['Company'].duplicated(), "")
        
        st.dataframe(
            display_df[["Company_Display", "Period", "Revenue", "OpIncome", "Margin", "Volume", "Japan", "NA", "Asia"]]
            .style.format({
                "Revenue": "{:,.1f}", "OpIncome": "{:,.1f}", 
                "Margin": "{:.1f}%", "Volume": "{:,.0f}"
            }, na_rep="-")
            .background_gradient(subset=["Margin"], cmap="RdYlGn", vmin=-5, vmax=10)
            .map(lambda x: 'color: red;' if isinstance(x, (int, float)) and x < 0 else '', subset=["OpIncome"]),
            use_container_width=True, hide_index=True
        )

        if show_charts:
            st.divider()
            c1, c2 = st.columns(2)
            df_24 = df[df["Period"] == "Last Year (H1)"]
            df_25 = df[df["Period"] == "Current (H1)"]

            with c1: # Revenue Chart
                fig_rev = go.Figure()
                rev_text = []
                for comp in ranking:
                    v25 = df_25[df_25["Company"] == comp]["Revenue"].values[0]
                    v24 = df_24[df_24["Company"] == comp]["Revenue"].values[0]
                    diff = ((v25 / v24) - 1) * 100 if v24 != 0 else 0
                    color = "red" if diff < 0 else "#444"
                    rev_text.append(f"<span style='color:{color}'>{diff:+.1f}%</span><br>{v25:,.0f}")

                fig_rev.add_trace(go.Bar(name='FY2024', x=df_24["Company"], y=df_24["Revenue"], marker_color='#FFB399'))
                fig_rev.add_trace(go.Bar(name='FY2025', x=df_25["Company"], y=df_25["Revenue"], marker_color='#FF4500', text=rev_text, textposition='outside'))
                fig_rev.update_layout(
                    title_text="<b>Revenue</b> (Billion JPY)", 
                    paper_bgcolor='#F2F2F2', plot_bgcolor='#F2F2F2',
                    yaxis=dict(gridcolor='white') # 修正箇所
                )
                st.plotly_chart(fig_rev, use_container_width=True)

            with c2: # Operating Income Chart
                fig_inc = go.Figure()
                inc_text = []
                for comp in ranking:
                    v25 = df_25[df_25["Company"] == comp]["OpIncome"].values[0]
                    v24 = df_24[df_24["Company"] == comp]["OpIncome"].values[0]
                    diff_txt = f"{((v25 / v24) - 1) * 100:+.1f}%" if v24 != 0 else "-"
                    color = "red" if v25 < v24 else "#444"
                    inc_text.append(f"<span style='color:{color}'>{diff_txt}</span><br>{v25:,.0f}")

                fig_inc.add_trace(go.Bar(name='FY2024', x=df_24["Company"], y=df_24["OpIncome"], marker_color='#A992E2'))
                fig_inc.add_trace(go.Bar(name='FY2025', x=df_25["Company"], y=df_25["OpIncome"], marker_color='#483D8B', text=inc_text, textposition='outside'))
                fig_inc.update_layout(
                    title_text="<b>Operating Income</b> (Billion JPY)", 
                    paper_bgcolor='#F2F2F2', plot_bgcolor='#F2F2F2',
                    yaxis=dict(gridcolor='white', zerolinecolor='grey') # 修正箇所：yaxis内に移動
                )
                st.plotly_chart(fig_inc, use_container_width=True)