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
    """全ての金額を『億円』に統一し、桁の誤認（ドット/カンマ間違い）を補正"""
    if val is None or val == 0: return 0
    
    # AIが 20,056 を 20.056 (ドット誤認) と出した場合の補正
    # 大手OEMの営業利益や売上が 100億円（数値として100）未満になることは通常無いため、
    # 1.0〜100.0 の間の小数が来た場合はドット誤認と判断して1000倍する
    if 1.0 < abs(val) < 100.0:
        return val * 1000
    
    abs_val = abs(val)
    # 1. 兆単位での出力 (例: 2.0 兆) -> 1万倍して億円へ
    if 0 < abs_val <= 1.0: 
        return val * 10000
    # 2. 百万円単位での出力 (例: 2,005,600) -> 100で割って億円へ
    if 1000000 < abs_val < 100000000:
        return val / 100
    # 3. 円単位での巨大出力 -> 1億で割って億円へ
    if abs_val >= 100000000:
        return val / 100000000
        
    return val

def standardize_volume(val):
    if val is None or val == 0: return 0
    # 台単位 (例: 4,783,000) -> 1000で割って千台へ
    if abs(val) > 50000:
        return val / 1000
    return val

# --- 2. 解析ロジック (プロンプトを大幅に強化) ---
def process_pdf(uploaded_file):
    file_bytes = uploaded_file.read()
    gemini_file = client.files.upload(file=io.BytesIO(file_bytes), config={'mime_type': 'application/pdf'})

    prompt = """
    Extract financial results accurately. 
    
    【CRITICAL: SEPARATOR RULE】
    - In Japanese reports, commas (,) are used as THOUSANDS SEPARATORS, not decimal points.
    - Example: "20,056" is TWENTY THOUSAND FIFTY-SIX, NOT twenty point zero five six.
    - DO NOT output decimal points for Oku-yen values unless it's for Margin (%).
    
    【Unit Rules】
    - ALL Currency: Output in '100 Million JPY' (億円) as a RAW NUMBER.
      - "20,056億円" -> 20056
      - "2兆56億円" -> 20056
      - "2,005,600百万円" -> 20056
    - ALL Volumes: Output in '1,000 units' (千台).
      - "4,783,000台" -> 4783
    
    【Definitions】
    - Toyota: Wholesale="連結販売台数", Retail="ご参考(小売)"
    - Nissan: Retail="小売販売台数"
    """

    response = client.models.generate_content(
        model="gemini-flash-latest", # 最新かつ安定した2.0-flashを推奨
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
st.set_page_config(page_title="OEM YoY Dashboard", layout="wide")
st.title("📈 Automotive OEM Financial Dashboard")

with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
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
                    "Margin": m.operating_margin_pct if abs(m.operating_margin_pct) < 50 else m.operating_margin_pct/10,
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
        # 型変換
        for col in ["Revenue", "OpIncome", "Margin", "Wholesale", "Retail"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- 4. 可視化：比較テーブル ---
        st.subheader("📊 Performance Comparison Table (Units: 100M JPY / 1k Units)")
        
        styled_df = df.style.format({
            "Revenue": "{:,.0f}", "OpIncome": "{:,.0f}", 
            "Margin": "{:.1f}%", "Wholesale": "{:,.0f}", "Retail": "{:,.0f}"
        }, na_rep="-")\
        .background_gradient(subset=["Margin"], cmap="RdYlGn", vmin=-5, vmax=10)\
        .bar(subset=["Revenue"], color='#d1e7dd', align='mid')\
        .map(lambda x: 'color: red;' if isinstance(x, (int, float)) and x < 0 else '', subset=["OpIncome"])

        st.dataframe(styled_df, use_container_width=True, height=400)

        # --- 5. KPI Metrics ---
        st.subheader("Executive KPI (H1 Growth)")
        k_cols = st.columns(len(df["Company"].unique()))
        for i, company in enumerate(df["Company"].unique()):
            try:
                c_df = df[df["Company"] == company]
                now = c_df[c_df["Period"] == "Current (H1)"].iloc[0]
                prev = c_df[c_df["Period"] == "Last Year (H1)"].iloc[0]
                rev_growth = ((now["Revenue"] / prev["Revenue"]) - 1) * 100
                with k_cols[i]:
                    st.metric(
                        label=f"{company} Revenue", 
                        value=f"¥{now['Revenue']:,.0f} Oku", 
                        delta=f"{rev_growth:+.1f}% YoY"
                    )
            except: continue

        # --- 6. Charts ---
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