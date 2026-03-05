import streamlit as st
import pandas as pd
import numpy as np
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
    revenue: float
    operating_income: float
    operating_margin_pct: float
    wholesale_vol: Optional[float]
    retail_vol: float
    fx_usd: float
    regional_retail: RegionalRetail

class ReportSchema(BaseModel):
    company_name: str
    prior_h1_actual: FinancialMetrics
    h1_actual: FinancialMetrics
    full_year_forecast: FinancialMetrics

# --- 2. 解析・計算ロジック ---
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def standardize(val):
    if val is None or val == 0: return 0
    if abs(val) > 1000000:
        return val / 100000000 if abs(val) > 100000000 else val / 1000
    return val

def process_pdf(uploaded_file):
    file_bytes = uploaded_file.read()
    gemini_file = client.files.upload(file=io.BytesIO(file_bytes), config={'mime_type': 'application/pdf'})
    prompt = "Extract H1 actual (current and prior) and Full Year forecast. Toyota: Wholesale=連結, Retail=ご参考."
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[gemini_file, prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=ReportSchema)
    )
    client.files.delete(name=gemini_file.name)
    return response.parsed

# --- 3. テーブル・スタイリング関数 ---
def style_comparison_table(df):
    summary_df = df[df["Period"].isin(["Last Year (H1)", "Current (H1)"])].copy()
    
    # 確実に数値型に変換
    for c in ["Revenue", "OpIncome", "Margin", "Retail", "Wholesale"]:
        if c in summary_df.columns:
            summary_df[c] = pd.to_numeric(summary_df[c], errors='coerce')

    return summary_df.style.format({
            "Revenue": "{:,.0f}", "OpIncome": "{:,.0f}", 
            "Margin": "{:.1f}%", "Retail": "{:,.0f}", "Wholesale": "{:,.0f}"
        }, na_rep="N/A")\
        .background_gradient(subset=["Margin"], cmap="RdYlGn", vmin=-5, vmax=10)\
        .bar(subset=["Revenue"], color='#d1e7dd', align='mid')\
        .map(lambda x: 'color: red;' if isinstance(x, (int, float)) and x < 0 else 'color: black;', subset=["OpIncome"])

# --- 4. UI部 ---
st.set_page_config(page_title="OEM Visual Analytics", layout="wide")

st.title("🚗 OEM Financial Visual Comparison")
st.write("各指標の規模と成長率を色分けで直感的に把握できます。")

with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    analyze_btn = st.button("🚀 Run Analysis")

if analyze_btn and uploaded_files:
    rows = []
    for f in uploaded_files:
        try:
            data = process_pdf(f)
            for pk in ['prior_h1_actual', 'h1_actual', 'full_year_forecast']:
                m = getattr(data, pk)
                rows.append({
                    "Company": data.company_name,
                    "Period": "Last Year (H1)" if pk == 'prior_h1_actual' else ("Current (H1)" if pk == 'h1_actual' else "FY Forecast"),
                    "Revenue": standardize(m.revenue),
                    "OpIncome": standardize(m.operating_income),
                    "Margin": m.operating_margin_pct,
                    "Retail": standardize(m.retail_vol),
                    "Wholesale": standardize(m.wholesale_vol),
                    "FX": m.fx_usd
                })
        except Exception as e:
            st.error(f"Error: {e}")
    
    if rows:
        df = pd.DataFrame(rows)
        # 数値型へ強制変換
        for col in ["Revenue", "OpIncome", "Margin", "Retail", "Wholesale", "FX"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        st.subheader("📊 Executive Summary Table")
        st.dataframe(style_comparison_table(df), use_container_width=True)

        st.subheader("📈 Year-over-Year Growth (%)")
        yoy_list = []
        for company in df["Company"].unique():
            c_df = df[df["Company"] == company]
            try:
                now = c_df[c_df["Period"] == "Current (H1)"].iloc[0]
                prev = c_df[c_df["Period"] == "Last Year (H1)"].iloc[0]
                yoy_list.append({
                    "Company": company,
                    "Revenue Growth (%)": (now["Revenue"] / prev["Revenue"] - 1) * 100 if prev["Revenue"] != 0 else np.nan,
                    "OpIncome Growth (%)": (now["OpIncome"] / prev["OpIncome"] - 1) * 100 if prev["OpIncome"] != 0 else np.nan,
                    "Retail Sales Growth (%)": (now["Retail"] / prev["Retail"] - 1) * 100 if prev["Retail"] != 0 else np.nan
                })
            except: continue
        
        if yoy_list:
            yoy_df = pd.DataFrame(yoy_list)
            yoy_cols = ["Revenue Growth (%)", "OpIncome Growth (%)", "Retail Sales Growth (%)"]
            for c in yoy_cols:
                yoy_df[c] = pd.to_numeric(yoy_df[c], errors='coerce')

            # 【重要】エラーの原因を「安全なフォーマッタ」で修正
            st.dataframe(
                yoy_df.style.format(
                    lambda x: f"{x:+.1f}%" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A"
                ).map(
                    lambda x: 'background-color: #f8d7da; color: #721c24;' if isinstance(x, (int, float)) and x < 0 else ('background-color: #d1e7dd; color: #0f5132;' if isinstance(x, (int, float)) and x > 0 else ''),
                    subset=yoy_cols
                ),
                use_container_width=True
            )

        st.divider()
        st.info("💡 営業利益がマイナスの場合は赤字（赤色）で表示されます。")