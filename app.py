import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Optional
import io

# --- 1. スキーマ定義 ---
class RegionalRetail(BaseModel):
    japan: float
    north_america: float
    europe: float
    asia_incl_china: float
    other: float

class FinancialMetrics(BaseModel):
    revenue: float               # 単位: 億円
    operating_income: float      # 単位: 億円
    operating_margin_pct: float  # 単位: %
    wholesale_vol: Optional[float] # 単位: 千台
    retail_vol: float            # 単位: 千台
    fx_usd: float
    regional_retail: RegionalRetail

class ReportSchema(BaseModel):
    company_name: str
    prior_h1_actual: FinancialMetrics
    h1_actual: FinancialMetrics
    full_year_forecast: FinancialMetrics

# API Client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def standardize_currency(val):
    """全ての金額を『億円』に統一し、桁の誤認を補正"""
    if val is None or val == 0: return 0
    # ドット誤認（20.056 -> 20056）の補正
    if 1.0 < abs(val) < 100.0:
        val = val * 1000
    
    abs_val = abs(val)
    if 0 < abs_val <= 1.0: val = val * 10000
    elif 1000000 < abs_val < 100000000: val = val / 100
    elif abs_val >= 100000000: val = val / 100000000
        
    return round(val) # 整数に丸める

def standardize_volume(val):
    if val is None or val == 0: return 0
    if abs(val) > 50000: val = val / 1000
    return round(val) # 整数に丸める

# --- 2. 解析ロジック ---
def process_pdf(uploaded_file):
    file_bytes = uploaded_file.read()
    gemini_file = client.files.upload(file=io.BytesIO(file_bytes), config={'mime_type': 'application/pdf'})

    prompt = """
    Extract financial results accurately. 
    【Separator Rule】 Comma (,) is a thousands separator, NOT a decimal point. 20,056 means 20056.
    【Unit Rules】
    - Currency: Output in '100 Million JPY' (億円) as an integer.
    - Volumes: Output in '1,000 units' (千台) as an integer.
    【Definitions】
    - Toyota: Wholesale="連結販売台数", Retail="ご参考(小売)"
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[gemini_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReportSchema,
            temperature=0.1,
        ),
    )
    client.files.delete(name=gemini_file.name)
    return response.parsed

# --- 3. UI部 ---
st.set_page_config(page_title="OEM Dashboard", layout="wide")
st.title("📈 Automotive OEM Financial Dashboard")

with st.sidebar:
    st.header("1. Input & Settings")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    # --- グラフ表示のオプションを追加 ---
    show_charts = st.checkbox("Show Visual Charts (Plotly)", value=False)
    
    analyze_btn = st.button("🚀 Analyze Data")

if analyze_btn and uploaded_files:
    all_rows = []
    for uploaded_file in uploaded_files:
        try:
            data = process_pdf(uploaded_file)
            for p_key in ['prior_h1_actual', 'h1_actual', 'full_year_forecast']:
                m = getattr(data, p_key)
                reg = m.regional_retail
                all_rows.append({
                    "Company": data.company_name,
                    "Period": "Last Year (H1)" if p_key == 'prior_h1_actual' else ("Current (H1)" if p_key == 'h1_actual' else "FY Forecast"),
                    "Revenue": standardize_currency(m.revenue),
                    "OpIncome": standardize_currency(m.operating_income),
                    "Margin": round(m.operating_margin_pct) if abs(m.operating_margin_pct) < 50 else round(m.operating_margin_pct/10),
                    "Wholesale": standardize_volume(m.wholesale_vol),
                    "Retail": standardize_volume(m.retail_vol),
                    "Japan": standardize_volume(reg.japan), 
                    "NA": standardize_volume(reg.north_america), 
                    "Asia": standardize_volume(reg.asia_incl_china)
                })
        except Exception as e:
            st.error(f"Error in {uploaded_file.name}: {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        for col in ["Revenue", "OpIncome", "Margin", "Wholesale", "Retail"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- 4. 比較テーブル (小数点なしの表示設定) ---
        st.subheader("📊 Performance Comparison Table (Integer View)")
        
        styled_df = df.style.format({
            "Revenue": "{:,.0f}", 
            "OpIncome": "{:,.0f}", 
            "Margin": "{:.0f}%",     # 小数点なし
            "Wholesale": "{:,.0f}", 
            "Retail": "{:,.0f}",
            "Japan": "{:,.0f}", "NA": "{:,.0f}", "Asia": "{:,.0f}"
        }, na_rep="-")\
        .background_gradient(subset=["Margin"], cmap="RdYlGn", vmin=-5, vmax=10)\
        .bar(subset=["Revenue"], color='#d1e7dd', align='mid')\
        .map(lambda x: 'color: red;' if isinstance(x, (int, float)) and x < 0 else '', subset=["OpIncome"])

        st.dataframe(styled_df, use_container_width=True, height=400)

        # --- 5. KPI Metrics ---
        st.subheader("Executive KPI")
        k_cols = st.columns(len(df["Company"].unique()))
        for i, company in enumerate(df["Company"].unique()):
            try:
                c_df = df[df["Company"] == company]
                now = c_df[c_df["Period"] == "Current (H1)"].iloc[0]
                prev = c_df[c_df["Period"] == "Last Year (H1)"].iloc[0]
                rev_growth = ((now["Revenue"] / prev["Revenue"]) - 1) * 100
                with k_cols[i]:
                    st.metric(label=f"{company} Revenue", value=f"¥{now['Revenue']:,.0f} Oku", delta=f"{rev_growth:+.0f}% YoY")
            except: continue

        # --- 6. Charts (オプションがオンの時のみ実行) ---
        if show_charts:
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 📊 Revenue Comparison")
                fig = go.Figure()
                for p in ["Last Year (H1)", "Current (H1)"]:
                    p_df = df[df["Period"] == p]
                    fig.add_trace(go.Bar(x=p_df["Company"], y=p_df["Revenue"], name=p))
                fig.update_layout(barmode='group', legend=dict(orientation="h", y=1.2))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("### 📉 Operating Income")
                fig_inc = go.Figure()
                for p in ["Last Year (H1)", "Current (H1)"]:
                    p_df = df[df["Period"] == p]
                    fig_inc.add_trace(go.Bar(x=p_df["Company"], y=p_df["OpIncome"], name=p))
                fig_inc.update_layout(barmode='group', legend=dict(orientation="h", y=1.2))
                st.plotly_chart(fig_inc, use_container_width=True)
        else:
            st.info("💡 Charts are hidden to save processing time. Enable 'Show Charts' in the sidebar to visualize.")