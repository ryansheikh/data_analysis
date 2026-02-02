import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Pharma Sales Analytics",
    page_icon=None,
    layout="wide"
)

# ----------------------------
# Helpers
# ----------------------------
# Absolute path based on app.py location (Streamlit-safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_DEFAULT = os.path.join(BASE_DIR, "data", "cleaned")
st.write("Resolved data path:", DATA_DIR_DEFAULT)
st.write("Path exists:", os.path.exists(DATA_DIR_DEFAULT))

if os.path.exists(DATA_DIR_DEFAULT):
    st.write("Files found:")
    st.write(os.listdir(DATA_DIR_DEFAULT))
else:
    st.error("Data directory not found")




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

NUM_COL_CANDIDATES = {
    "TotalUnits", "TotalBonus", "TotalDiscount", "TotalSales",
    "UnitsSold", "Revenue", "TotalClients", "AvgSellingPrice",
    "AvgMonthlySales"
}

def _safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in NUM_COL_CANDIDATES:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # Year/Month as int
    for c in ["Year", "Month"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def _add_month_date(df: pd.DataFrame) -> pd.DataFrame:
    if "Year" in df.columns and "Month" in df.columns:
        df = df.copy()
        df["MonthStart"] = pd.to_datetime(
            df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01",
            errors="coerce"
        )
    return df

@st.cache_data(show_spinner=False)
def load_all(data_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for k, fname in FILE_MAP.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            df = _safe_read_csv(fpath)
            df = _coerce_numeric(df)
            df = _add_month_date(df)
            out[k] = df
    return out

def fmt_num(x):
    try:
        x = float(x)
    except Exception:
        return str(x)
    absx = abs(x)
    if absx >= 1e9:
        return f"{x/1e9:.2f}B"
    if absx >= 1e6:
        return f"{x/1e6:.2f}M"
    if absx >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:,.0f}"

def kpi_delta(curr, prev):
    if prev in [0, None, np.nan]:
        return None
    return (curr - prev) / prev * 100.0

# ----------------------------
# Sidebar
# ----------------------------
st.title("Pharma Sales Analytics Dashboard")

with st.sidebar:
    st.header("Settings")
    data_dir = st.text_input("Cleaned data folder", value=DATA_DIR_DEFAULT)
    data = load_all(data_dir)

    missing = [v for v in FILE_MAP.values() if not os.path.exists(os.path.join(data_dir, v))]
    with st.expander("Data status", expanded=False):
        st.write(f"Found datasets: {len(data)} / {len(FILE_MAP)}")
        if missing:
            st.write("Missing files:")
            for m in missing:
                st.write(f"- {m}")
        else:
            st.write("All files found.")

    st.divider()
    page = st.radio(
        "Page",
        options=[
            "Executive Overview",
            "Products",
            "Distributors",
            "Client Types",
            "Promotions (Bonus/Discount)",
            "Dimensions Explorer",
            "Seasonality",
            "Pricing"
        ],
        index=0
    )

# ----------------------------
# Guards
# ----------------------------
def require(keys: list[str]):
    missing_keys = [k for k in keys if k not in data]
    if missing_keys:
        st.error(
            "Required dataset(s) not found: "
            + ", ".join([FILE_MAP[k] for k in missing_keys])
            + f"\n\nCheck the folder path: {data_dir}"
        )
        st.stop()

# ----------------------------
# Pages
# ----------------------------
if page == "Executive Overview":
    require(["monthly_sales"])
    ms = data["monthly_sales"].dropna(subset=["MonthStart"]).sort_values("MonthStart").copy()

    # Filters
    min_date, max_date = ms["MonthStart"].min(), ms["MonthStart"].max()
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        date_range = st.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()) if pd.notnull(min_date) and pd.notnull(max_date) else None
        )
    with c2:
        metric = st.selectbox(
            "Primary metric",
            ["TotalSales", "TotalUnits", "TotalBonus", "TotalDiscount"],
            index=0
        )
    with c3:
        show_ma = st.checkbox("Show 3-month moving average", value=True)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        msf = ms[(ms["MonthStart"] >= start) & (ms["MonthStart"] <= end)].copy()
    else:
        msf = ms

    # KPIs (latest vs previous month)
    if len(msf) >= 2:
        latest = msf.iloc[-1]
        prev = msf.iloc[-2]
        sales_curr = float(latest.get("TotalSales", 0))
        sales_prev = float(prev.get("TotalSales", 0))
        units_curr = float(latest.get("TotalUnits", 0))
        units_prev = float(prev.get("TotalUnits", 0))

        d_sales = kpi_delta(sales_curr, sales_prev)
        d_units = kpi_delta(units_curr, units_prev)
    else:
        latest = msf.iloc[-1] if len(msf) else None
        sales_curr = float(latest.get("TotalSales", 0)) if latest is not None else 0
        units_curr = float(latest.get("TotalUnits", 0)) if latest is not None else 0
        d_sales = None
        d_units = None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest Month Sales", fmt_num(sales_curr), f"{d_sales:.2f}%" if d_sales is not None else None)
    k2.metric("Latest Month Units", fmt_num(units_curr), f"{d_units:.2f}%" if d_units is not None else None)
    k3.metric("Total Sales (selected range)", fmt_num(msf["TotalSales"].sum()))
    k4.metric("Total Units (selected range)", fmt_num(msf["TotalUnits"].sum()))

    st.subheader("Monthly Trend")
    base = msf[["MonthStart", metric]].copy()
    fig = px.line(base, x="MonthStart", y=metric, markers=True)

    if show_ma and len(msf) >= 3:
        ma = msf[[ "MonthStart", metric]].copy()
        ma["MA_3"] = ma[metric].rolling(3).mean()
        fig.add_trace(go.Scatter(x=ma["MonthStart"], y=ma["MA_3"], mode="lines", name="MA(3)"))

    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales vs Promotions")
    promo_cols = ["TotalBonus", "TotalDiscount", "TotalSales"]
    promo = msf[["MonthStart"] + [c for c in promo_cols if c in msf.columns]].copy()
    fig2 = go.Figure()
    if "TotalSales" in promo.columns:
        fig2.add_trace(go.Scatter(x=promo["MonthStart"], y=promo["TotalSales"], mode="lines+markers", name="TotalSales"))
    if "TotalBonus" in promo.columns:
        fig2.add_trace(go.Bar(x=promo["MonthStart"], y=promo["TotalBonus"], name="TotalBonus", opacity=0.6))
    if "TotalDiscount" in promo.columns:
        fig2.add_trace(go.Bar(x=promo["MonthStart"], y=promo["TotalDiscount"], name="TotalDiscount", opacity=0.6))

    fig2.update_layout(
        barmode="group",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Value",
        xaxis_title="Month"
    )
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Products":
    require(["monthly_product_sales", "top_products"])
    mps = data["monthly_product_sales"].dropna(subset=["MonthStart"]).copy()
    top = data["top_products"].copy()

    st.subheader("Top Products (Snapshot)")
    n = st.slider("Top N", 10, 50, 20, step=5)
    topn = top.sort_values("Revenue", ascending=False).head(n).copy()
    fig = px.bar(topn, x="Revenue", y="ProductName", orientation="h")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product Trend Explorer")
    products = mps["ProductName"].dropna().astype(str).sort_values().unique().tolist()
    default_products = products[:1] if products else []
    selected = st.multiselect("Select product(s)", options=products, default=default_products)

    metric = st.selectbox("Metric", ["Revenue", "UnitsSold"], index=0)
    df = mps[mps["ProductName"].isin(selected)].sort_values("MonthStart") if selected else mps.head(0)

    if len(df):
        fig2 = px.line(df, x="MonthStart", y=metric, color="ProductName", markers=True)
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Select at least one product to view trends.")

elif page == "Distributors":
    require(["distributor_performance"])
    dist = data["distributor_performance"].copy()

    st.subheader("Distributor Concentration")
    dist = dist.sort_values("Revenue", ascending=False)
    top_n = st.slider("Top N distributors", 10, min(200, max(10, len(dist))), 30)
    shown = dist.head(top_n).copy()

    fig = px.bar(shown, x="Revenue", y="DistributorName", orientation="h")
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pareto (80/20) View")
    dist2 = dist.copy()
    dist2["RevenueShare"] = dist2["Revenue"] / dist2["Revenue"].sum() if dist2["Revenue"].sum() else 0
    dist2["CumShare"] = dist2["RevenueShare"].cumsum()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=np.arange(1, len(dist2) + 1), y=dist2["RevenueShare"], name="Revenue Share"))
    fig2.add_trace(go.Scatter(x=np.arange(1, len(dist2) + 1), y=dist2["CumShare"], mode="lines", name="Cumulative Share"))
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Distributor Rank", yaxis_title="Share")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Client Types":
    require(["client_type_analysis", "monthly_client_type_sales"])
    cta = data["client_type_analysis"].copy()
    mcts = data["monthly_client_type_sales"].dropna(subset=["MonthStart"]).copy()

    st.subheader("Client Type Mix (Snapshot)")
    fig = px.pie(cta, names="ClientType", values="Revenue")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Client Type Trends")
    metric = st.selectbox("Trend metric", ["Revenue"], index=0)
    fig2 = px.line(mcts.sort_values("MonthStart"), x="MonthStart", y=metric, color="ClientType", markers=True)
    fig2.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Promotions (Bonus/Discount)":
    require(["monthly_sales", "bonus_discount_monthly"])
    ms = data["monthly_sales"].dropna(subset=["MonthStart"]).sort_values("MonthStart").copy()
    bd = data["bonus_discount_monthly"].dropna(subset=["MonthStart"]).sort_values("MonthStart").copy()

    st.subheader("Bonus & Discount Over Time")
    df = bd.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["MonthStart"], y=df["TotalBonus"], mode="lines+markers", name="TotalBonus"))
    fig.add_trace(go.Scatter(x=df["MonthStart"], y=df["TotalDiscount"], mode="lines+markers", name="TotalDiscount"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Promotion Intensity vs Sales")
    # Merge monthly promo with sales
    merged = pd.merge(
        ms[["MonthStart", "TotalSales"]].copy(),
        bd[["MonthStart", "TotalBonus", "TotalDiscount"]].copy(),
        on="MonthStart",
        how="inner"
    )
    merged["PromoTotal"] = merged["TotalBonus"] + merged["TotalDiscount"]
    fig2 = px.scatter(merged, x="PromoTotal", y="TotalSales", hover_data=["MonthStart"])
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Dimensions Explorer":
    require(["dimension_summary"])
    ds = data["dimension_summary"].copy()

    st.subheader("Multi-Dimension Revenue Explorer")
    c1, c2, c3, c4 = st.columns(4)

    distributors = ["All"] + sorted(ds["DistributorName"].dropna().astype(str).unique().tolist())
    client_types = ["All"] + sorted(ds["ClientType"].dropna().astype(str).unique().tolist())
    bricks = ["All"] + sorted(ds["BrickName"].dropna().astype(str).unique().tolist())
    teams = ["All"] + sorted(ds["TeamName"].dropna().astype(str).unique().tolist())

    with c1:
        d_sel = st.selectbox("Distributor", distributors, index=0)
    with c2:
        ct_sel = st.selectbox("Client Type", client_types, index=0)
    with c3:
        b_sel = st.selectbox("Brick", bricks, index=0)
    with c4:
        t_sel = st.selectbox("Team", teams, index=0)

    df = ds.copy()
    if d_sel != "All":
        df = df[df["DistributorName"].astype(str) == d_sel]
    if ct_sel != "All":
        df = df[df["ClientType"].astype(str) == ct_sel]
    if b_sel != "All":
        df = df[df["BrickName"].astype(str) == b_sel]
    if t_sel != "All":
        df = df[df["TeamName"].astype(str) == t_sel]

    st.write(f"Rows after filter: {len(df):,}")

    if len(df):
        # Top breakdowns
        col_a, col_b = st.columns(2)

        with col_a:
            top_by_brick = df.groupby("BrickName", dropna=False)["Revenue"].sum().sort_values(ascending=False).head(20).reset_index()
            fig = px.bar(top_by_brick, x="Revenue", y="BrickName", orientation="h")
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            top_by_team = df.groupby("TeamName", dropna=False)["Revenue"].sum().sort_values(ascending=False).head(20).reset_index()
            fig2 = px.bar(top_by_team, x="Revenue", y="TeamName", orientation="h")
            fig2.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Pivot Table")
        pivot = df.pivot_table(
            index="TeamName",
            columns="ClientType",
            values="Revenue",
            aggfunc="sum",
            fill_value=0
        )
        st.dataframe(pivot, use_container_width=True)
    else:
        st.info("No data found for selected filters.")

elif page == "Seasonality":
    require(["monthly_sales", "seasonality_monthly_avg"])
    ms = data["monthly_sales"].dropna(subset=["MonthStart"]).sort_values("MonthStart").copy()
    sea = data["seasonality_monthly_avg"].copy()

    st.subheader("Seasonality Profile")
    fig = px.bar(sea.sort_values("Month"), x="Month", y="AvgMonthlySales")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Year x Month Heatmap (Sales)")
    ms["Year"] = ms["Year"].astype(int)
    ms["Month"] = ms["Month"].astype(int)
    heat = ms.pivot_table(index="Year", columns="Month", values="TotalSales", aggfunc="sum", fill_value=0)
    fig2 = px.imshow(heat, aspect="auto")
    fig2.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Pricing":
    # accept either price_sensitivity or product_avg_price
    if "price_sensitivity" in data:
        pr = data["price_sensitivity"].copy()
    elif "product_avg_price" in data:
        pr = data["product_avg_price"].copy()
    else:
        st.error(
            "Required dataset not found: "
            f"{FILE_MAP['price_sensitivity']} or {FILE_MAP['product_avg_price']}\n\n"
            f"Check the folder path: {data_dir}"
        )
        st.stop()

    st.subheader("Average Selling Price by Product")
    min_units = st.slider("Minimum total units filter", 0, int(pr["TotalUnits"].max()) if len(pr) else 0, 0)
    df = pr[pr["TotalUnits"] >= min_units].copy()

    df = df.sort_values("AvgSellingPrice", ascending=False).head(50)
    fig = px.bar(df, x="AvgSellingPrice", y="ProductName", orientation="h")
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price vs Volume")
    df2 = pr[pr["TotalUnits"] >= max(min_units, 1)].copy()
    fig2 = px.scatter(df2, x="AvgSellingPrice", y="TotalUnits", hover_data=["ProductName"])
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)



