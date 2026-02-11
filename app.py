import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Pharmevo Sales Analytics",
    layout="wide"
)

st.title("Pharmevo Sales Analytics Dashboard")
st.caption("Executive-level analytics built on aggregated SQL Server data")

# ==================================================

# ==================================================
def format_number(value):
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.2f}K"
    else:
        return f"{value:.2f}"

# ==================================================

# ==================================================
if st.sidebar.button("🔄 Refresh Data (Clear Cache)"):
    st.cache_data.clear()
    st.rerun()

def file_signature(path):
    return os.path.getmtime(path)

# ==================================================
# DATA LOADING
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_MAP = {
    "monthly_sales": "cleaned_monthly_sales.csv",
    "monthly_product_sales": "cleaned_monthly_product_sales.csv",
    "top_products": "cleaned_top_products.csv",
    "distributor_performance": "cleaned_distributor_performance.csv",
    "client_type_analysis": "cleaned_client_type_analysis.csv",
    "bonus_discount_monthly": "cleaned_bonus_discount_monthly.csv",
    "dimension_summary": "cleaned_dimension_summary.csv",
    "monthly_client_type_sales": "cleaned_monthly_client_type_sales.csv",
    "price_sensitivity": "cleaned_price_sensitivity.csv",
    "seasonality_monthly_avg": "cleaned_seasonality_monthly_avg.csv",
}

NUMERIC_COLUMNS = {
    "TotalUnits", "TotalBonus", "TotalDiscount", "TotalSales",
    "UnitsSold", "Revenue", "TotalClients",
    "AvgSellingPrice", "AvgMonthlySales"
}

sig = tuple(file_signature(os.path.join(BASE_DIR, f)) for f in FILE_MAP.values())

@st.cache_data
def load_data(sig):
    data = {}
    for key, fname in FILE_MAP.items():
        path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(path):
            st.error(f"Missing dataset: {fname}")
            st.stop()

        df = pd.read_csv(path)

        for col in df.columns:
            if col in NUMERIC_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if "Year" in df.columns and "Month" in df.columns:
            df["MonthStart"] = pd.to_datetime(
                df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
            )

        data[key] = df

    return data

data = load_data(sig)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select View",
    [
        "Executive Overview",
        "Product Performance",
        "Distributor Performance",
        "Client Analysis",
        "Promotion Impact",
        "Seasonality & Cycles",
        "Pricing Analysis",
        "Dimension Drilldown"
    ]
)

# ==================================================
# EXECUTIVE OVERVIEW
# ==================================================
if page == "Executive Overview":
    df = data["monthly_sales"].sort_values("MonthStart").copy()
    df["MoM_Growth"] = df["TotalSales"].pct_change() * 100
    df["Rolling_3M"] = df["TotalSales"].rolling(3).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Latest Sales",
        format_number(latest["TotalSales"]),
        f"{((latest['TotalSales']-prev['TotalSales'])/prev['TotalSales']*100):.2f}%"
        if prev["TotalSales"] else None
    )

    k2.metric("Latest Units", format_number(latest["TotalUnits"]))
    k3.metric("Avg Monthly Sales", format_number(df["TotalSales"].mean()))
    k4.metric("Best Month Sales", format_number(df["TotalSales"].max()))
    k5.metric("Worst Month Sales", format_number(df["TotalSales"].min()))

    st.subheader("Sales Trend with Rolling Average")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["MonthStart"], y=df["TotalSales"], name="Sales"))
    fig.add_trace(go.Scatter(x=df["MonthStart"], y=df["Rolling_3M"], name="3M Rolling Avg"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Month-on-Month Growth")
    fig2 = px.bar(df, x="MonthStart", y="MoM_Growth")
    st.plotly_chart(fig2, use_container_width=True)
