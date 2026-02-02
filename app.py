import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# Streamlit Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Pharma Sales Analytics Dashboard",
    layout="wide"
)

# --------------------------------------------------
# Data Location (CSV files are in repo ROOT)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR

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
    "monthly_sales_growth_base": "cleaned_monthly_sales_growth_base.csv",
    "product_monthly_units": "cleaned_product_monthly_units.csv",
    "product_avg_price": "cleaned_product_avg_price.csv",
}

NUMERIC_COLUMNS = {
    "TotalUnits", "TotalBonus", "TotalDiscount", "TotalSales",
    "UnitsSold", "Revenue", "TotalClients",
    "AvgSellingPrice", "AvgMonthlySales"
}

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_all_data():
    datasets = {}

    for key, filename in FILE_MAP.items():
        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            st.error(f"Required dataset not found: {filename}")
            st.stop()

        df = pd.read_csv(path)

        # Fix numeric columns
        for col in df.columns:
            if col in NUMERIC_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Create MonthStart if possible
        if "Year" in df.columns and "Month" in df.columns:
            df["MonthStart"] = pd.to_datetime(
                df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01",
                errors="coerce"
            )

        datasets[key] = df

    return datasets


def format_number(value):
    value = float(value)
    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.2f}K"
    return f"{value:,.0f}"


# --------------------------------------------------
# Load datasets
# --------------------------------------------------
data = load_all_data()

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "Executive Overview",
        "Product Analysis",
        "Distributor Analysis",
        "Client Type Analysis",
        "Promotions Analysis",
        "Seasonality",
        "Pricing",
        "Dimension Explorer",
    ]
)

# --------------------------------------------------
# Executive Overview
# --------------------------------------------------
if page == "Executive Overview":
    df = data["monthly_sales"].sort_values("MonthStart")

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Latest Month Sales",
        format_number(latest["TotalSales"]),
        f"{((latest['TotalSales'] - prev['TotalSales']) / prev['TotalSales'] * 100):.2f}%" if prev["TotalSales"] != 0 else None
    )
    c2.metric("Latest Month Units", format_number(latest["TotalUnits"]))
    c3.metric("Total Sales", format_number(df["TotalSales"].sum()))
    c4.metric("Total Units", format_number(df["TotalUnits"].sum()))

    fig = px.line(
        df,
        x="MonthStart",
        y="TotalSales",
        title="Monthly Sales Trend"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Product Analysis
# --------------------------------------------------
elif page == "Product Analysis":
    top = data["top_products"].sort_values("Revenue", ascending=False).head(20)

    fig = px.bar(
        top,
        x="Revenue",
        y="ProductName",
        orientation="h",
        title="Top Products by Revenue"
    )
    st.plotly_chart(fig, use_container_width=True)

    mps = data["monthly_product_sales"]
    products = sorted(mps["ProductName"].unique())

    selected_products = st.multiselect(
        "Select product(s)",
        products,
        default=products[:1]
    )

    if selected_products:
        fig2 = px.line(
            mps[mps["ProductName"].isin(selected_products)],
            x="MonthStart",
            y="Revenue",
            color="ProductName",
            title="Product Revenue Trend"
        )
        st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# Distributor Analysis
# --------------------------------------------------
elif page == "Distributor Analysis":
    dist = data["distributor_performance"].sort_values("Revenue", ascending=False).head(30)

    fig = px.bar(
        dist,
        x="Revenue",
        y="DistributorName",
        orientation="h",
        title="Top Distributors by Revenue"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Client Type Analysis
# --------------------------------------------------
elif page == "Client Type Analysis":
    ct = data["client_type_analysis"]

    fig = px.pie(
        ct,
        names="ClientType",
        values="Revenue",
        title="Revenue Share by Client Type"
    )
    st.plotly_chart(fig, use_container_width=True)

    mct = data["monthly_client_type_sales"]
    fig2 = px.line(
        mct,
        x="MonthStart",
        y="Revenue",
        color="ClientType",
        title="Client Type Revenue Trend"
    )
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# Promotions Analysis
# --------------------------------------------------
elif page == "Promotions Analysis":
    promo = data["bonus_discount_monthly"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=promo["MonthStart"], y=promo["TotalBonus"], name="Bonus"))
    fig.add_trace(go.Scatter(x=promo["MonthStart"], y=promo["TotalDiscount"], name="Discount"))

    fig.update_layout(title="Bonus and Discount Trend Over Time")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Seasonality
# --------------------------------------------------
elif page == "Seasonality":
    sea = data["seasonality_monthly_avg"]

    fig = px.bar(
        sea,
        x="Month",
        y="AvgMonthlySales",
        title="Average Monthly Sales (Seasonality)"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Pricing
# --------------------------------------------------
elif page == "Pricing":
    price = data["price_sensitivity"].sort_values("AvgSellingPrice", ascending=False).head(30)

    fig = px.bar(
        price,
        x="AvgSellingPrice",
        y="ProductName",
        orientation="h",
        title="Average Selling Price by Product"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Dimension Explorer
# --------------------------------------------------
elif page == "Dimension Explorer":
    dim = data["dimension_summary"]

    distributor = st.selectbox(
        "Distributor",
        ["All"] + sorted(dim["DistributorName"].unique())
    )

    client_type = st.selectbox(
        "Client Type",
        ["All"] + sorted(dim["ClientType"].unique())
    )

    filtered = dim.copy()
    if distributor != "All":
        filtered = filtered[filtered["DistributorName"] == distributor]
    if client_type != "All":
        filtered = filtered[filtered["ClientType"] == client_type]

    summary = (
        filtered
        .groupby("TeamName", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
    )

    fig = px.bar(
        summary,
        x="Revenue",
        y="TeamName",
        orientation="h",
        title="Revenue by Team"
    )
    st.plotly_chart(fig, use_container_width=True)
