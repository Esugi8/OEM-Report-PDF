import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Optional, List
import io

# =================================================================
# 1. DATA SCHEMA & AI CONFIGURATION
# =================================================================

class RegionalRetail(BaseModel):
    japan: float
    north_america: float
    europe: float
    asia_incl_china: float
    other: float

class FinancialMetrics(BaseModel):
    revenue: float               # Target: 100M JPY
    operating_income: float      # Target: 100M JPY
    operating_margin_pct: float  # Target: %
    wholesale_vol: Optional[float]
    retail_vol: float
    fx_usd: float
    regional_retail: RegionalRetail

class ReportSchema(BaseModel):
    company_name: str
    h1_actual: FinancialMetrics
    full_year_forecast: FinancialMetrics

# Initialize Gemini Client
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("API Key not found in .streamlit/secrets.toml")
    st.stop()

# =================================================================
# 2. CORE LOGIC: EXTRACTION & CLEANING
# =================================================================

def standardize(val):
    """Normalize large values (Yen/Units) to 100M JPY or 1k Units."""
    if val is None or val == 0: return 0
    if abs(val) > 1000000:
        return val / 100000000 if abs(val) > 100000000 else val / 1000
    return val

def process_pdf(uploaded_file):
    """Upload PDF to Gemini and get structured data."""
    # Convert Streamlit UploadedFile to bytes for Gemini API
    file_bytes = uploaded_file.read()
    
    # Upload to Gemini File API
    gemini_file = client.files.upload(
        file=io.BytesIO(file_bytes),
        config={'mime_type': 'application/pdf'}
    )

    prompt = """
    Extract financial results and forecasts from this document.
    - Scale all currency to 100 Million JPY (Oku-yen).
    - Scale all volumes to 1,000 units.
    - Margin should be a percentage (e.g., 8.5).
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[gemini_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReportSchema,
        ),
    )
    
    # Clean up file from cloud
    client.files.delete(name=gemini_file.name)
    return response.parsed

# =================================================================
# 3. UI: STREAMLIT DASHBOARD
# =================================================================

st.set_page_config(page_title="OEM Executive Insights", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fb; }
    div[data-testid="metric-container"] {
        background-color: white; border: 1px solid #e1e4e8; padding: 15px; border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 OEM Executive Dashboard")
st.write("Compare global automotive financial performance using Generative AI.")

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("1. Data Input")
    uploaded_files = st.file_uploader(
        "Upload Financial Result PDFs", 
        type="pdf", 
        accept_multiple_files=True
    )
    analyze_btn = st.button("🚀 Analyze Reports", use_container_width=True)

# --- Processing Logic ---
if analyze_btn and uploaded_files:
    all_rows = []
    
    with st.status("Analyzing documents with Gemini 2.5 Flash...", expanded=True) as status:
        for uploaded_file in uploaded_files:
            st.write(f"Processing {uploaded_file.name}...")
            try:
                data = process_pdf(uploaded_file)
                name = data.company_name
                
                for period in ['h1_actual', 'full_year_forecast']:
                    metrics = getattr(data, period)
                    reg = metrics.regional_retail
                    all_rows.append({
                        "Company": name,
                        "Type": "H1 Actual" if period == 'h1_actual' else "FY Forecast",
                        "Revenue (100M)": standardize(metrics.revenue),
                        "OpIncome (100M)": standardize(metrics.operating_income),
                        "Margin (%)": metrics.operating_margin_pct,
                        "Wholesale (1k)": standardize(metrics.wholesale_vol),
                        "Retail (1k)": standardize(metrics.retail_vol),
                        "Japan": standardize(reg.japan),
                        "N.America": standardize(reg.north_america),
                        "Asia": standardize(reg.asia_incl_china),
                        "Other": standardize(reg.other)
                    })
            except Exception as e:
                st.error(f"Error in {uploaded_file.name}: {e}")
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    if all_rows:
        df = pd.DataFrame(all_rows)
        actual_df = df[df["Type"] == "H1 Actual"]

        # --- 2. KPI Metrics ---
        st.subheader("Executive Summary (H1 Actual)")
        kpi_cols = st.columns(len(actual_df))
        for i, (idx, row) in enumerate(actual_df.iterrows()):
            with kpi_cols[i]:
                st.metric(
                    label=f"{row['Company']} Revenue", 
                    value=f"¥{row['Revenue (100M)']:,.0f} B",
                    delta=f"{row['Margin (%)']:.1f}% Margin"
                )

        # --- 3. Interactive Charts ---
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("### 📈 Revenue vs Profitability")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=actual_df["Company"], y=actual_df["Revenue (100M)"], name="Revenue (100M JPY)", marker_color="#3498db"))
            fig.add_trace(go.Scatter(x=actual_df["Company"], y=actual_df["Margin (%)"], name="Margin %", yaxis="y2", line=dict(color="#e74c3c", width=4), marker=dict(size=10)))
            fig.update_layout(
                yaxis=dict(title="Revenue (100M JPY)"),
                yaxis2=dict(title="Op Margin (%)", overlaying="y", side="right", range=[-5, 15]),
                legend=dict(orientation="h", y=1.2)
            )
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            st.markdown("### 🌍 Regional Sales Mix (Retail)")
            fig_reg = px.bar(
                actual_df, x="Company", y=["Japan", "N.America", "Asia", "Other"],
                barmode="stack", color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_reg.update_layout(legend=dict(orientation="h", y=1.2), yaxis_title="Units (1,000)")
            st.plotly_chart(fig_reg, use_container_width=True)

        # --- 4. Detailed Comparison Table ---
        st.subheader("📋 Financial Data Details")
        st.dataframe(
            df.style.background_gradient(subset=["Margin (%)"], cmap="RdYlGn")
            .format({"Revenue (100M)": "{:,.0f}", "OpIncome (100M)": "{:,.0f}", "Margin (%)": "{:.1f}%"}),
            use_container_width=True
        )
        
        # Download Link
        st.download_button("📥 Download Data as CSV", df.to_csv(index=False), "oem_comparison.csv", "text/csv")

else:
    st.info("👈 Please upload OEM financial PDF reports in the sidebar and click 'Analyze Reports'.")