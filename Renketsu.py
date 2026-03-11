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
class RegionalSales(BaseModel):
    japan: float
    north_america: float
    europe: float
    asia_incl_china: float
    other: float

class FinancialMetrics(BaseModel):
    revenue: float               
    operating_income: float      
    operating_margin_pct: float  
    volume: float                # 連結販売台数に一本化
    fx_usd: float
    regional_sales: RegionalSales

class ReportSchema(BaseModel):
    company_name: str
    prior_h1_actual: FinancialMetrics
    h1_actual: FinancialMetrics
    full_year_forecast: FinancialMetrics

# API Client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def standardize_currency(val):
    if val is None or val == 0: return 0
    abs_val = abs(val)
    # ドット誤認補正 (20.056 -> 20056)
    if 1.0 < abs_val < 100.0: val = val * 1000
    # 単位正規化（億円）
    if 0 < abs_val <= 1.0: val = val * 10000
    elif 1000000 < abs_val < 100000000: val = val / 100
    elif abs_val >= 100000000: val = val / 100000000
    return val

def standardize_volume(val):
    if val is None or val == 0: return 0
    if abs(val) > 50000: val = val / 1000
    return val

# --- 2. 解析ロジック (連結台数のみに指示を集中) ---
def process_pdf(uploaded_file):
    file_bytes = uploaded_file.read()
    gemini_file = client.files.upload(file=io.BytesIO(file_bytes), config={'mime_type': 'application/pdf'})

    prompt = """
    Extract financial results accurately from the tables.
    【Separator Rule】 Comma (,) is a thousands separator, NOT a decimal point. 20,056 means 20056.
    【Volume Rule】 Extract ONLY "Consolidated Sales Volume" (連結販売台数) as 'volume'. 
    【Margin Rule】 Extract Operating Margin % (営業利益率) EXACTLY as written. (e.g., 0.5% -> 0.5)
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
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
st.set_page_config(page_title="OEM Executive Dashboard", layout="wide")
st.title("🚗 Automotive OEM Consolidated Analysis")

with st.sidebar:
    st.header("1. Input & Settings")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    show_charts = st.checkbox("Show Visual Charts", value=False)
    analyze_btn = st.button("🚀 Analyze Data")

if analyze_btn and uploaded_files:
    all_rows = []
    for uploaded_file in uploaded_files:
        try:
            data = process_pdf(uploaded_file)
            for p_key in ['prior_h1_actual', 'h1_actual', 'full_year_forecast']:
                m = getattr(data, p_key)
                reg = m.regional_sales
                all_rows.append({
                    "Company": data.company_name,
                    "Period": "Last Year (H1)" if p_key == 'prior_h1_actual' else ("Current (H1)" if p_key == 'h1_actual' else "FY Forecast"),
                    "Revenue": standardize_currency(m.revenue),
                    "OpIncome": standardize_currency(m.operating_income),
                    "Margin": m.operating_margin_pct,
                    "Consolidated Volume": standardize_volume(m.volume),
                    "Japan": standardize_volume(reg.japan), 
                    "NA": standardize_volume(reg.north_america), 
                    "Asia": standardize_volume(reg.asia_incl_china)
                })
        except Exception as e:
            st.error(f"Error in {uploaded_file.name}: {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        
        # 数値型変換
        for col in ["Revenue", "OpIncome", "Margin", "Consolidated Volume", "Japan", "NA", "Asia"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- 4. 【今回の重要修正】会社名列の「結合表示」ロジック ---
        # 表示用のコピーを作成
        display_df = df.copy()
        
        # 同じ会社名が続く場合、2行目以降を空白にする（視覚的な結合効果）
        display_df['Company'] = display_df['Company'].where(~display_df['Company'].duplicated(), "")

        st.subheader("📊 Performance Comparison Table (Merged View)")
        
        # テーブル表示とスタイリング
        styled_df = display_df.style.format({
            "Revenue": "{:,.0f}", 
            "OpIncome": "{:,.0f}", 
            "Margin": "{:.1f}%", 
            "Consolidated Volume": "{:,.0f}", 
            "Japan": "{:,.0f}", "NA": "{:,.0f}", "Asia": "{:,.0f}"
        }, na_rep="-")\
        .background_gradient(subset=["Margin"], cmap="RdYlGn", vmin=-5, vmax=10)\
        .bar(subset=["Revenue"], color='#d1e7dd', align='mid')\
        .map(lambda x: 'color: red;' if isinstance(x, (int, float)) and x < 0 else '', subset=["OpIncome"])

        # インデックスを非表示にして、より「レポート」らしい見た目にする
        st.dataframe(styled_df, use_container_width=True, height=400, hide_index=True)

        # --- 5. KPI & Charts ---
        # (以下、KPIとグラフの描画は、計算が必要なため元の `df` を使用して継続)
        st.subheader("Executive KPI")
        companies = df["Company"].unique()
        k_cols = st.columns(len(companies))
        for i, company in enumerate(companies):
            try:
                c_df = df[df["Company"] == company]
                now = c_df[c_df["Period"] == "Current (H1)"].iloc[0]
                prev = c_df[c_df["Period"] == "Last Year (H1)"].iloc[0]
                rev_growth = ((now["Revenue"] / prev["Revenue"]) - 1) * 100
                with k_cols[i]:
                    st.metric(label=f"{company}", value=f"¥{now['Revenue']:,.0f} Oku", delta=f"{rev_growth:+.1f}% YoY")
            except: continue

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