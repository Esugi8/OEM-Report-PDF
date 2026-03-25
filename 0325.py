import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Optional, List
import io
import time

# =================================================================
# 1. データ構造定義
# =================================================================
class RegionalSales(BaseModel):
    japan: float = 0.0
    north_america: float = 0.0
    europe: float = 0.0
    asia_incl_china: float = 0.0
    other: float = 0.0

class FinancialMetrics(BaseModel):
    revenue: float               # Target: Billion JPY (1,000,000,000 JPY)
    operating_income: float      # Target: Billion JPY
    operating_margin_pct: float  # %
    volume: float                # Target: k units (1,000 units)
    fx_usd: float = 0.0
    regional_sales: Optional[RegionalSales] = None

class ReportSchema(BaseModel):
    company_name: str
    prior_h1_actual: FinancialMetrics
    h1_actual: FinancialMetrics
    full_year_forecast: Optional[FinancialMetrics] = None

# =================================================================
# 2. ロジック：解析 (算術と単位特定にのみ従う)
# =================================================================
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def clean_numeric(val):
    """
    純粋な数値変換のみ。規模感による推測計算は行わない。
    """
    if val is None: return np.nan
    try:
        if isinstance(val, str):
            val = val.replace(',', '').replace('¥', '').replace(' ', '')
        return float(val)
    except: return np.nan

def process_pdf(uploaded_file, status_container):
    status_container.write(f"📂 Analyzing: {uploaded_file.name}")
    file_bytes = uploaded_file.read()
    gemini_file = client.files.upload(
        file=io.BytesIO(file_bytes), 
        config={'mime_type': 'application/pdf'}
    )

    # 規模感を教えず、資料内の単位表記に基づく計算のみを指示
    prompt = """
    Extract financial results by strictly following the units stated in the document headers.
    
    【UNIT CONVERSION LOGIC】
    Target Unit for Revenue/Income: "Billion JPY" (1,000,000,000 Yen).
    Convert the raw values in the tables based on the header unit:
    - If the header is "百万円" (Millions of Yen): Divide the value by 1,000.
    - If the header is "億円" (100 Millions of Yen): Divide the value by 10.
    - If the header is "兆円" (Trillions of Yen): Multiply the value by 1,000.

    Target Unit for Volume: "k units" (1,000 units).
    - If source is "台" (Units): Divide by 1,000.
    - If source is "万台" (10k units): Multiply by 10.
    - If source is "千台" (k units): Keep as is.

    【STRICT RULES】
    - NEVER guess the scale. Use only the math based on the unit written in the PDF.
    - Extract "Consolidated" (連結) figures only.
    - If a specific Forecast table is found, extract it; otherwise return null for forecast fields.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=[gemini_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ReportSchema,
                    temperature=0.0, # 決定論的な結果を保証
                ),
            )
            client.files.delete(name=gemini_file.name)
            return response.parsed
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                time.sleep(5)
                continue
            client.files.delete(name=gemini_file.name)
            raise e

# =================================================================
# 3. UI部：洗練されたプロフェッショナルUI
# =================================================================
st.set_page_config(page_title="Executive OEM Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #FFFFFF; }
    h1 { color: #1E293B; font-weight: 800; font-size: 2.2rem; }
    .stMetric { background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 0.5rem; }
    .stProgress > div > div > div > div { background-color: #483D8B; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Automotive OEM Global Performance")

if 'master_df' not in st.session_state:
    st.session_state.master_df = None

with st.sidebar:
    st.header("Step 1: Data Ingestion")
    uploaded_files = st.file_uploader("Upload PDF Reports", type="pdf", accept_multiple_files=True)
    analyze_btn = st.button("Run AI Analysis", use_container_width=True)
    st.divider()
    show_charts = st.checkbox("Show Visual Charts", value=True)

# --- A. 解析プロセス ---
if analyze_btn and uploaded_files:
    all_rows = []
    total = len(uploaded_files)
    with st.status("Analyzing...", expanded=True) as status:
        prog = st.progress(0)
        for i, f in enumerate(uploaded_files):
            try:
                data = process_pdf(f, status)
                raw_name = data.company_name
                name = "Nissan" if any(x in raw_name.lower() for x in ["nissan", "日産"]) else raw_name
                
                periods = [('prior_h1_actual', 'Prior Year (H1)'), ('h1_actual', 'Current Year (H1)'), ('full_year_forecast', 'Full Year Forecast')]
                for key, label in periods:
                    m = getattr(data, key, None)
                    if m and (m.revenue is not None and m.revenue != 0):
                        reg = m.regional_sales if m.regional_sales else RegionalSales()
                        all_rows.append({
                            "Company": name, "Period": label,
                            "Revenue": clean_numeric(m.revenue),
                            "OpIncome": clean_numeric(m.operating_income),
                            "Margin": clean_numeric(m.operating_margin_pct),
                            "Volume": clean_numeric(m.volume),
                            "Japan": clean_numeric(reg.japan), "NA": clean_numeric(reg.north_america), "Asia": clean_numeric(reg.asia_incl_china)
                        })
                    else:
                        all_rows.append({"Company": name, "Period": label, "Revenue": np.nan, "OpIncome": np.nan, "Margin": np.nan, "Volume": np.nan, "Japan": np.nan, "NA": np.nan, "Asia": np.nan})
                prog.progress((i + 1) / total)
            except Exception as e:
                st.error(f"Error {f.name}: {e}")
        
        if all_rows:
            st.session_state.master_df = pd.DataFrame(all_rows)
            status.update(label="Analysis complete", state="complete", expanded=False)

# --- B. 表示セクション ---
if st.session_state.master_df is not None:
    df = st.session_state.master_df.copy()
    
    st.sidebar.header("Step 2: Filter Companies")
    all_oems = sorted(df["Company"].unique().tolist())
    selected_oems = [oem for oem in all_oems if st.sidebar.checkbox(oem, value=True, key=f"sb_{oem}")]

    if selected_oems:
        filtered_df = df[df["Company"].isin(selected_oems)].copy()
        
        # 収益(Revenue)順にソート (Current H1基準)
        curr_h1 = filtered_df[filtered_df["Period"] == "Current Year (H1)"].sort_values("Revenue", ascending=False)
        ranking = curr_h1["Company"].tolist()
        filtered_df['Company'] = pd.Categorical(filtered_df['Company'], categories=ranking, ordered=True)
        filtered_df = filtered_df.sort_values(["Company", "Period"])

        # 1. 比較テーブル
        st.subheader("📋 Performance Benchmarking Table")
        display_df = filtered_df.copy()
        display_df['Company_Display'] = display_df['Company'].astype(str).where(~display_df['Company'].duplicated(), "")
        
        st.dataframe(
            display_df[["Company_Display", "Period", "Revenue", "OpIncome", "Margin", "Volume", "Japan", "NA", "Asia"]]
            .style.format({
                "Revenue": "{:,.1f}", "OpIncome": "{:,.1f}", "Margin": "{:.1f}%",
                "Volume": "{:,.0f}", "Japan": "{:,.0f}", "NA": "{:,.0f}", "Asia": "{:,.0f}"
            }, na_rep="-")
            .background_gradient(subset=["Margin"], cmap="Greens", vmin=0, vmax=12)
            .map(lambda x: 'color: #E74C3C; font-weight: bold;' if isinstance(x, (int, float)) and x < 0 else '', subset=["OpIncome"]),
            use_container_width=True, hide_index=True
        )

        # 2. グラフ (指定カラー厳守)
        if show_charts:
            st.divider()
            c1, c2 = st.columns(2)
            df_25 = filtered_df[filtered_df["Period"] == "Current Year (H1)"]
            df_24 = filtered_df[filtered_df["Period"] == "Prior Year (H1)"]

            with c1: # Revenue (Orange)
                fig_rev = go.Figure()
                rev_text = [f"<span style='color:{'red' if (v2/v1-1)<0 else '#444' if v1>0 else '#444'}'>{(v2/v1-1)*100:+.1f}%</span><br><b>{v2:,.0f}</b>" if v1>0 else f"<b>{v2:,.0f}</b>" for v1, v2 in zip(df_24["Revenue"], df_25["Revenue"])]
                fig_rev.add_trace(go.Bar(name='FY2024', x=df_24["Company"], y=df_24["Revenue"], marker_color='#FFB399'))
                fig_rev.add_trace(go.Bar(name='FY2025', x=df_25["Company"], y=df_25["Revenue"], marker_color='#FF4500', text=rev_text, textposition='outside'))
                fig_rev.update_layout(title_text="<b>Revenue</b> (Billion JPY)", title_x=0.5, paper_bgcolor='#F2F2F2', plot_bgcolor='#F2F2F2', yaxis=dict(gridcolor='white'))
                st.plotly_chart(fig_rev, use_container_width=True)

            with c2: # Operating Income (Purple)
                fig_inc = go.Figure()
                inc_text = [f"<b>{v2:,.0f}</b>" for v2 in df_25["OpIncome"]]
                fig_inc.add_trace(go.Bar(name='FY2024', x=df_24["Company"], y=df_24["OpIncome"], marker_color='#A992E2'))
                fig_inc.add_trace(go.Bar(name='FY2025', x=df_25["Company"], y=df_25["OpIncome"], marker_color='#483D8B', text=inc_text, textposition='outside'))
                fig_inc.update_layout(title_text="<b>Operating Income</b> (Billion JPY)", title_x=0.5, paper_bgcolor='#F2F2F2', plot_bgcolor='#F2F2F2', yaxis=dict(gridcolor='white', zerolinecolor='grey'))
                st.plotly_chart(fig_inc, use_container_width=True)