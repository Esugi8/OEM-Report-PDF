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
# 1. データ構造定義 (Schema)
# =================================================================
class RegionalSales(BaseModel):
    japan: Optional[float] = None
    north_america: Optional[float] = None
    europe: Optional[float] = None
    asia_excl_japan: Optional[float] = None
    other: Optional[float] = None

class FinancialMetrics(BaseModel):
    revenue: float               
    operating_income: float      
    operating_margin_pct: float  
    volume: float                
    fx_usd: float = 0.0
    regional_sales: Optional[RegionalSales] = None

class ReportSchema(BaseModel):
    company_name: str
    prior_h1_actual: FinancialMetrics
    h1_actual: FinancialMetrics
    full_year_forecast: Optional[FinancialMetrics] = None

# =================================================================
# 2. ロジック：解析・クレンジング
# =================================================================
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def clean_numeric(val):
    if val is None: return np.nan
    try:
        if isinstance(val, str):
            val = val.replace(',', '').replace('¥', '').replace(' ', '')
        return float(val)
    except: return np.nan

def normalize_company_name(name):
    name_map = {
        "Toyota": ["トヨタ", "Toyota"],
        "Nissan": ["日産", "Nissan"],
        "Honda": ["本田", "Honda"],
        "Suzuki": ["スズキ", "Suzuki"],
        "Mazda": ["マツダ", "Mazda"],
        "Subaru": ["スバル", "Subaru"],
        "Isuzu": ["いすゞ", "Isuzu"],
        "Mitsubishi": ["三菱", "Mitsubishi"]
    }
    for standard, aliases in name_map.items():
        if any(alias.lower() in name.lower() for alias in aliases):
            return standard
    return name

def process_pdf(uploaded_file, status_container):
    status_container.write(f"📂 Analyzing Segment Integration: {uploaded_file.name}")
    file_bytes = uploaded_file.read()
    gemini_file = client.files.upload(
        file=io.BytesIO(file_bytes), 
        config={'mime_type': 'application/pdf'}
    )

    # ページ指定を排し、製品セグメントを跨いだ『算術的統合』を命じるプロンプト
    prompt = """
    Extract financial and regional sales results by following these strict logical rules.

    【1. UNIT CONVERSION LOGIC】
    Target currency unit: "Billion JPY" (1,000,000,000 JPY).
    Identify the unit label (百万円, 億円, 兆円) and apply:
    - Millions (百万円): Value / 1,000
    - 100 Millions (億円): Value / 10
    - Trillions (兆円): Value * 1,000

    【2. ISUZU-STYLE SEGMENT INTEGRATION】
    - Some companies like Isuzu report regional sales separately for different product segments (e.g., "CV" and "LCV").
    - You MUST find the regional data for EACH segment and SUM them to calculate the total regional sales.
    - Example for "japan": (CV Japan Sales) + (LCV Japan Sales) = Total Japan.
    - Example for "europe": (CV Europe Sales) + (LCV Europe Sales) = Total Europe.
    
    【3. REGIONAL MAPPING DEFINITION】
    - "asia_excl_japan": Sum of all Asia regions (including "China", "Thailand", "India", "ASEAN") but EXCLUDING Japan.
    - "other": Sum of all remaining regions (Middle East, Africa, Oceania, Central/South America).

    【4. DATA HIERARCHY】
    - Financials (Revenue/Income): Use top-level "Consolidated" (連結) totals only.
    - Volume: Use only "Automobile" (四輪) business. IGNORE Motorcycles.
    
    【5. CONSISTENCY CHECK】
    - The sum of (japan + north_america + europe + asia_excl_japan + other) MUST match the "Total Volume" you extracted.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                #model="gemini-2.5-flash",
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
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                time.sleep(5)
                continue
            client.files.delete(name=gemini_file.name)
            raise e

# =================================================================
# 3. UI部：エグゼクティブ・デザイン
# =================================================================
st.set_page_config(page_title="Executive OEM Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #FFFFFF; }
    h1 { color: #1E293B; font-weight: 800; font-size: 2.2rem; }
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

# --- A. 解析 ---
if analyze_btn and uploaded_files:
    all_rows = []
    total = len(uploaded_files)
    with st.status("Standardizing across all OEMs...", expanded=True) as status:
        prog = st.progress(0)
        for i, f in enumerate(uploaded_files):
            try:
                status.write(f"Processing: {f.name}")
                data = process_pdf(f, status)
                name = normalize_company_name(data.company_name)
                
                periods = [
                    ('prior_h1_actual', 'Prior Year (H1)'), 
                    ('h1_actual', 'Current Year (H1)'), 
                    ('full_year_forecast', 'Full Year Forecast')
                ]
                for key, label in periods:
                    m = getattr(data, key, None)
                    if m and (m.revenue is not None and m.revenue != 0):
                        reg = m.regional_sales if m.regional_sales else RegionalSales()
                        all_rows.append({
                            "Company": name, "Period": label,
                            "Revenue": clean_numeric(m.revenue),
                            "OpIncome": clean_numeric(m.operating_income),
                            "Margin": m.operating_margin_pct,
                            "Total Vol": clean_numeric(m.volume),
                            "Japan": clean_numeric(reg.japan), 
                            "NA": clean_numeric(reg.north_america), 
                            "Europe": clean_numeric(reg.europe),
                            "Asia(ex.JP)": clean_numeric(reg.asia_excl_japan),
                            "Other": clean_numeric(reg.other)
                        })
                    else:
                        all_rows.append({"Company": name, "Period": label, "Revenue": np.nan})
                prog.progress((i + 1) / total)
            except Exception as e:
                st.error(f"Error {f.name}: {e}")
        if all_rows:
            st.session_state.master_df = pd.DataFrame(all_rows)
            status.update(label="Analysis complete", state="complete", expanded=False)

# --- B. 表示 ---
if st.session_state.master_df is not None:
    df = st.session_state.master_df.copy()
    
    st.sidebar.header("Step 2: Filter Companies")
    all_oems = sorted([x for x in df["Company"].unique() if pd.notna(x)])
    selected_oems = [oem for oem in all_oems if st.sidebar.checkbox(oem, value=True, key=f"sb_{oem}")]

    if selected_oems:
        filtered_df = df[df["Company"].isin(selected_oems)].copy()
        curr_h1 = filtered_df[filtered_df["Period"] == "Current Year (H1)"].sort_values("Revenue", ascending=False)
        ranking = curr_h1["Company"].tolist()
        filtered_df['Company'] = pd.Categorical(filtered_df['Company'], categories=ranking, ordered=True)
        filtered_df = filtered_df.sort_values(["Company", "Period"])

        # テーブル表示
        st.subheader("📋 Performance Benchmarking Table")
        display_df = filtered_df.copy()
        display_df['Company_Display'] = display_df['Company'].astype(str).where(~display_df['Company'].duplicated(), "")
        
        cols = ["Company_Display", "Period", "Revenue", "OpIncome", "Margin", "Total Vol", "Japan", "NA", "Europe", "Asia(ex.JP)", "Other"]
        st.dataframe(
            display_df[cols].style.format({
                "Revenue": "{:,.1f}", "OpIncome": "{:,.1f}", "Margin": "{:.1f}%",
                "Total Vol": "{:,.0f}", "Japan": "{:,.0f}", "NA": "{:,.0f}", "Europe": "{:,.0f}", "Asia(ex.JP)": "{:,.0f}", "Other": "{:,.0f}"
            }, na_rep="-")
            .background_gradient(subset=["Margin"], cmap="Greens", vmin=0, vmax=12)
            .map(lambda x: 'color: #E74C3C; font-weight: bold;' if isinstance(x, (int, float)) and x < 0 else '', subset=["OpIncome"]),
            use_container_width=True, hide_index=True
        )

        # グラフ (オリジナル配色厳守)
        if show_charts:
            st.divider()
            c1, c2 = st.columns(2)
            df_25 = filtered_df[filtered_df["Period"] == "Current Year (H1)"]
            df_24 = filtered_df[filtered_df["Period"] == "Prior Year (H1)"]

            with c1: # Revenue (Orange)
                fig_rev = go.Figure()
                rev_text = [f"<span style='color:{'red' if (v2/v1-1)<0 else '#444'}'>{(v2/v1-1)*100:+.1f}%</span><br><b>{v2:,.1f}</b>" if v1>0 else f"<b>{v2:,.1f}</b>" for v1, v2 in zip(df_24["Revenue"], df_25["Revenue"])]
                fig_rev.add_trace(go.Bar(name='FY2024', x=df_24["Company"], y=df_24["Revenue"], marker_color='#FFB399'))
                fig_rev.add_trace(go.Bar(name='FY2025', x=df_25["Company"], y=df_25["Revenue"], marker_color='#FF4500', text=rev_text, textposition='outside'))
                fig_rev.update_layout(title_text="<b>Revenue</b> (Billion JPY)", title_x=0.5, paper_bgcolor='#F2F2F2', plot_bgcolor='#F2F2F2', yaxis=dict(gridcolor='white'))
                st.plotly_chart(fig_rev, use_container_width=True)

            with c2: # Operating Income (Purple)
                fig_inc = go.Figure()
                inc_text = [f"<b>{v2:,.1f}</b>" for v2 in df_25["OpIncome"]]
                fig_inc.add_trace(go.Bar(name='FY2024', x=df_24["Company"], y=df_24["OpIncome"], marker_color='#A992E2'))
                fig_inc.add_trace(go.Bar(name='FY2025', x=df_25["Company"], y=df_25["OpIncome"], marker_color='#483D8B', text=inc_text, textposition='outside'))
                fig_inc.update_layout(title_text="<b>Operating Income</b> (Billion JPY)", title_x=0.5, paper_bgcolor='#F2F2F2', plot_bgcolor='#F2F2F2', yaxis=dict(gridcolor='white', zerolinecolor='grey'))
                st.plotly_chart(fig_inc, use_container_width=True)