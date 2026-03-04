import streamlit as st
import pandas as pd
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
    revenue: float
    operating_income: float
    operating_margin_pct: float
    wholesale_vol: Optional[float]
    retail_vol: float
    fx_usd: float
    regional_retail: RegionalRetail

class ReportSchema(BaseModel):
    company_name: str
    prior_h1_actual: FinancialMetrics    # 前年同期 (Previous Year H1)
    h1_actual: FinancialMetrics          # 当期実績 (Current Year H1)
    full_year_forecast: FinancialMetrics # 通期予想 (Full Year Forecast)

# API Client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def standardize(val):
    if val is None or val == 0: return 0
    if abs(val) > 1000000:
        return val / 100000000 if abs(val) > 100000000 else val / 1000
    return val

# --- 2. 解析ロジック (前年同期の抽出指示を追加) ---
def process_pdf(uploaded_file):
    file_bytes = uploaded_file.read()
    gemini_file = client.files.upload(file=io.BytesIO(file_bytes), config={'mime_type': 'application/pdf'})

    prompt = """
    Extract financial results including PREVIOUS YEAR's data (前年同期) shown in comparison tables.
    
    1. PERIODS:
       - prior_h1_actual: Data for 2024.4-9 (Previous Year H1).
       - h1_actual: Data for 2025.4-9 (Current Year H1).
       - full_year_forecast: Outlook for the full fiscal year.

    2. DEFINITIONS (Toyota):
       - Wholesale = "連結販売台数"
       - Retail = "ご参考(小売)" or "グループ総販売台数"
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[gemini_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReportSchema,
        ),
    )
    client.files.delete(name=gemini_file.name)
    return response.parsed

# --- 3. UI部 ---
st.set_page_config(page_title="OEM YoY Insights", layout="wide")
st.title("📈 OEM Performance: YoY Comparison")

with st.sidebar:
    st.header("1. Input")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    analyze_btn = st.button("🚀 Analyze YoY Data")

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
                    "Revenue": standardize(m.revenue),
                    "OpIncome": standardize(m.operating_income),
                    "Margin": m.operating_margin_pct,
                    "Wholesale": standardize(m.wholesale_vol),
                    "Retail": standardize(m.retail_vol),
                    "Japan": standardize(reg.japan), "NA": standardize(reg.north_america), "Asia": standardize(reg.asia_incl_china)
                })
        except Exception as e:
            st.error(f"Error in {uploaded_file.name}: {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        
        # --- YoY計算 ---
        # メーカーごとにLast YearとCurrentの差分を計算
        st.subheader("Executive Metrics (vs Last Year)")
        kpi_cols = st.columns(len(df["Company"].unique()))
        
        for i, company in enumerate(df["Company"].unique()):
            c_df = df[df["Company"] == company]
            now = c_df[c_df["Period"] == "Current (H1)"].iloc[0]
            prev = c_df[c_df["Period"] == "Last Year (H1)"].iloc[0]
            
            # 成長率
            rev_diff = ((now["Revenue"] / prev["Revenue"]) - 1) * 100
            
            with kpi_cols[i]:
                st.metric(
                    label=f"{company} Revenue", 
                    value=f"¥{now['Revenue']:,.0f} B",
                    delta=f"{rev_diff:+.1f}% YoY"
                )

        # --- 比較グラフ ---
        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### 📊 Revenue Comparison (YoY)")
            fig_rev = go.Figure()
            for p in ["Last Year (H1)", "Current (H1)"]:
                p_df = df[df["Period"] == p]
                fig_rev.add_trace(go.Bar(x=p_df["Company"], y=p_df["Revenue"], name=p))
            fig_rev.update_layout(barmode='group', yaxis_title="100M JPY")
            st.plotly_chart(fig_rev, use_container_width=True)

        with c2:
            st.markdown("### 📉 Operating Income Gap")
            fig_inc = go.Figure()
            for p in ["Last Year (H1)", "Current (H1)"]:
                p_df = df[df["Period"] == p]
                fig_inc.add_trace(go.Bar(x=p_df["Company"], y=p_df["OpIncome"], name=p))
            fig_inc.update_layout(barmode='group', yaxis_title="100M JPY")
            st.plotly_chart(fig_inc, use_container_width=True)

        # 全データ表示
        st.subheader("Detailed Data")
        st.dataframe(df, use_container_width=True)