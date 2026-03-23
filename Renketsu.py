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
    volume: float                
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
    if 1.0 < abs_val < 100.0: val = val * 1000
    if 0 < abs_val <= 1.0: val = val * 10000
    elif 1000000 < abs_val < 100000000: val = val / 100
    elif abs_val >= 100000000: val = val / 100000000
    return val

def standardize_volume(val):
    if val is None or val == 0: return 0
    if abs(val) > 50000: val = val / 1000
    return val

# --- 2. 解析ロジック ---
def process_pdf(uploaded_file):
    file_bytes = uploaded_file.read()
    gemini_file = client.files.upload(file=io.BytesIO(file_bytes), config={'mime_type': 'application/pdf'})

    prompt = """
    Extract financial results accurately from the tables.
    【Separator Rule】 Comma (,) is a thousands separator. 20,056 means 20056.
    【Volume Rule】 Extract ONLY "Consolidated Sales Volume" (連結販売台数) as 'volume'. 
    【Margin Rule】 Extract Operating Margin % (営業利益率) EXACTLY as written. (e.g., 0.5% -> 0.5)
    """

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
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
st.set_page_config(page_title="OEM Financial Ranking", layout="wide")
st.title("🚗 Automotive OEM Consolidated Analysis")
st.caption("Sorted by 2025 H1 Operating Income (Descending)")

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
                    "OpIncome": standardize_currency(m.operating_income), # ソート用に先頭付近に配置
                    "Revenue": standardize_currency(m.revenue),
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
        num_cols = ["Revenue", "OpIncome", "Margin", "Consolidated Volume", "Japan", "NA", "Asia"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- 【修正：収益（Revenue）順でランキングを確定】 ---
        # 1. 今期の実績(Current H1)データだけを抽出し、売上高(Revenue)順に並べる
        current_h1_data = df[df["Period"] == "Current (H1)"].sort_values("Revenue", ascending=False)
        # 2. 売上順の会社名リストを作成
        ranking_revenue = current_h1_data["Company"].tolist()
        
        # 3. 全体のデータフレームのCompany列をこの「売上順」に固定
        df['Company'] = pd.Categorical(df['Company'], categories=ranking_revenue, ordered=True)
        
        # 4. 期間の順序も固定（前年 -> 当期 -> 見通し）
        period_order = ["Last Year (H1)", "Current (H1)", "FY Forecast"]
        df['Period'] = pd.Categorical(df['Period'], categories=period_order, ordered=True)
        
        # 5. 会社(売上順) × 期間 の順でソートを確定
        df = df.sort_values(["Company", "Period"])

        # --- 4. 比較テーブル (Merged View) ---
        display_df = df.copy()
        # 連結表示のために重複する名前を消す（表示上のみ）
        display_df['Company_Display'] = display_df['Company'].astype(str).where(~display_df['Company'].duplicated(), "")
        cols_to_show = ["Company_Display", "Period", "Revenue", "OpIncome", "Margin", "Consolidated Volume", "Japan", "NA", "Asia"]
        display_df = display_df[cols_to_show].rename(columns={"Company_Display": "Company"})

        st.subheader("📊 Performance Comparison Table (Sorted by Revenue)")
        styled_df = display_df.style.format({
            "Revenue": "{:,.0f}", "OpIncome": "{:,.0f}", "Margin": "{:.1f}%", 
            "Consolidated Volume": "{:,.0f}", "Japan": "{:,.0f}", "NA": "{:,.0f}", "Asia": "{:,.0f}"
        }, na_rep="-")\
        .background_gradient(subset=["Margin"], cmap="RdYlGn", vmin=-5, vmax=10)\
        .bar(subset=["Revenue"], color='#d1e7dd', align='mid')\
        .map(lambda x: 'color: red;' if isinstance(x, (int, float)) and x < 0 else '', subset=["OpIncome"])

        st.dataframe(styled_df, use_container_width=True, height=450, hide_index=True)

        # --- 5. KPI Metrics ---
        st.subheader("Executive KPI (Sorted by Revenue)")
        k_cols = st.columns(len(ranking_revenue))
        for i, company in enumerate(ranking_revenue):
            try:
                c_df = df[df["Company"] == company]
                now = c_df[c_df["Period"] == "Current (H1)"].iloc[0]
                prev = c_df[c_df["Period"] == "Last Year (H1)"].iloc[0]
                # 収益の前年比を計算
                rev_growth = ((now["Revenue"] / prev["Revenue"]) - 1) * 100 if prev["Revenue"] != 0 else 0
                with k_cols[i]:
                    st.metric(
                        label=f"{company}", 
                        value=f"¥{now['Revenue']:,.0f} Oku", 
                        delta=f"{rev_growth:+.1f}% YoY Rev"
                    )
            except: continue

# --- 6. Charts (ご提示の画像デザインを再現) ---
        if show_charts:
            st.divider()
            
            # グラフ描画用のデータ準備
            # 10億円(Billion)単位に変換して画像に合わせる
            chart_df = df.copy()
            chart_df["Revenue_Bn"] = chart_df["Revenue"] / 10
            chart_df["OpIncome_Bn"] = chart_df["OpIncome"] / 10
            
            c1, c2 = st.columns(2)
            
            # --- 【左側：Revenue Chart】 ---
            with c1:
                fig_rev = go.Figure()
                
                # 年度ごとのデータ
                df_24 = chart_df[chart_df["Period"] == "Last Year (H1)"]
                df_25 = chart_df[chart_df["Period"] == "Current (H1)"]
                
                # 前年比のテキスト作成
                yoy_labels = []
                for comp in ranking_revenue:
                    try:
                        v25 = df_25[df_25["Company"] == comp]["Revenue_Bn"].values[0]
                        v24 = df_24[df_24["Company"] == comp]["Revenue_Bn"].values[0]
                        diff = ((v25 / v24) - 1) * 100
                        color = "red" if diff < 0 else "#444"
                        yoy_labels.append(f"<span style='color:{color}'>{diff:+.1f}%</span><br>{v25:,.0f}")
                    except: yoy_labels.append("")

                # FY2024 (薄いオレンジ)
                fig_rev.add_trace(go.Bar(
                    name='FY2024', x=df_24["Company"], y=df_24["Revenue_Bn"],
                    marker_color='#FFB399', offsetgroup=0
                ))
                # FY2025 (濃いオレンジ) + ラベル
                fig_rev.add_trace(go.Bar(
                    name='FY2025', x=df_25["Company"], y=df_25["Revenue_Bn"],
                    marker_color='#FF4500', offsetgroup=1,
                    text=yoy_labels, textposition='outside'
                ))

                fig_rev.update_layout(
                    title={'text': "<b>Revenue</b>", 'x':0.5, 'xanchor': 'center', 'font':{'size':24, 'color':'#444'}},
                    yaxis_title="billion JP yen",
                    paper_bgcolor='#F2F2F2', plot_bgcolor='#F2F2F2',
                    margin=dict(t=100, b=50, l=50, r=50),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis={'categoryorder':'array', 'categoryarray': ranking_revenue},
                    yaxis=dict(gridcolor='white', range=[0, 30000])
                )
                st.plotly_chart(fig_rev, use_container_width=True)

            # --- 【右側：Operating Income Chart】 ---
            with c2:
                fig_inc = go.Figure()
                
                # 前年比のテキスト作成
                yoy_labels_inc = []
                for comp in ranking_revenue:
                    try:
                        v25 = df_25[df_25["Company"] == comp]["OpIncome_Bn"].values[0]
                        v24 = df_24[df_24["Company"] == comp]["OpIncome_Bn"].values[0]
                        # 成長率（赤字転落などは "-" 表示）
                        if v24 > 0:
                            diff = ((v25 / v24) - 1) * 100
                            label_txt = f"{diff:+.1f}%"
                        else: label_txt = "-"
                        
                        color = "red" if v25 < v24 else "#444"
                        yoy_labels_inc.append(f"<span style='color:{color}'>{label_txt}</span><br>{v25:,.0f}")
                    except: yoy_labels_inc.append("")

                # FY2024 (薄いパープル)
                fig_inc.add_trace(go.Bar(
                    name='FY2024', x=df_24["Company"], y=df_24["OpIncome_Bn"],
                    marker_color='#A992E2', offsetgroup=0
                ))
                # FY2025 (濃いパープル)
                fig_inc.add_trace(go.Bar(
                    name='FY2025', x=df_25["Company"], y=df_25["OpIncome_Bn"],
                    marker_color='#483D8B', offsetgroup=1,
                    text=yoy_labels_inc, textposition='outside'
                ))

                fig_inc.update_layout(
                    title={'text': "<b>Operating Income</b>", 'x':0.5, 'xanchor': 'center', 'font':{'size':24, 'color':'#444'}},
                    yaxis_title="billion JP yen",
                    paper_bgcolor='#F2F2F2', plot_bgcolor='#F2F2F2',
                    margin=dict(t=100, b=50, l=50, r=50),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis={'categoryorder':'array', 'categoryarray': ranking_revenue},
                    yaxis=dict(gridcolor='white', zerolinecolor='grey', zerolinewidth=1)
                )
                st.plotly_chart(fig_inc, use_container_width=True)