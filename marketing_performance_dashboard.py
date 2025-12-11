# Lifesight_MarketingPerformance_Dashboard_V2_updated.py
# Updated: make sidebar labels & download button visible, remove Granularity filter.
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ------------- Mock data generator -------------
@st.cache_data
def generate_mock_data(months_before=6, months_after=0, seed=42):
    np.random.seed(seed)
    today = pd.to_datetime(datetime.utcnow().date())
    start = today - pd.DateOffset(months=months_before)
    end = today + pd.DateOffset(months=months_after)
    dates = pd.date_range(start=start, end=end, freq="D")

    channels = {
        "Meta": {"ctr": 0.02, "cpc": 0.35, "cv_rate": 0.03, "aov": 45, "cogs_pct": 0.45, "return_rate": 0.03},
        "Google": {"ctr": 0.035, "cpc": 0.50, "cv_rate": 0.05, "aov": 60, "cogs_pct": 0.40, "return_rate": 0.025},
        "Amazon": {"ctr": 0.06, "cpc": 0.60, "cv_rate": 0.10, "aov": 55, "cogs_pct": 0.50, "return_rate": 0.04},
        "TikTok": {"ctr": 0.015, "cpc": 0.20, "cv_rate": 0.02, "aov": 35, "cogs_pct": 0.50, "return_rate": 0.05}
    }

    rows = []
    for d in dates:
        day_multiplier = 1 + 0.05 * np.sin((d.dayofyear % 30) / 30 * 2 * np.pi)
        for channel, params in channels.items():
            for camp_i in range(1, 4):
                campaign = f"{channel}_Camp_{camp_i}"
                for adset_i in range(1, 3):
                    ad_set = f"{campaign}_AS{adset_i}"
                    for creative_i in range(1, 3):
                        creative = f"{ad_set}_CR{creative_i}"

                        base_imp = {"Meta":120000, "Google":80000, "Amazon":40000, "TikTok":150000}[channel]
                        impressions = max(0, int(np.random.normal(base_imp*0.02, base_imp*0.004)))
                        impressions = int(impressions * day_multiplier * (1 + 0.01 * (camp_i-2)))

                        ctr = max(0.001, np.random.normal(params["ctr"], params["ctr"]*0.25))
                        clicks = np.random.binomial(impressions, ctr) if impressions>0 else 0

                        cv_rate = max(0.001, np.random.normal(params["cv_rate"], params["cv_rate"]*0.25))
                        conversions = np.random.binomial(clicks, cv_rate) if clicks>0 else 0

                        cpc = max(0.05, np.random.normal(params["cpc"], params["cpc"]*0.12))
                        spend = round(clicks * cpc, 2)

                        aov = max(5, np.random.normal(params["aov"], params["aov"]*0.12))
                        revenue = round(conversions * aov, 2)
                        orders = conversions

                        cogs_pct = max(0.2, np.random.normal(params["cogs_pct"], 0.05))
                        cogs = round(revenue * cogs_pct, 2)

                        return_rate = max(0.0, np.random.normal(params["return_rate"], params["return_rate"]*0.3))
                        returns = int(round(orders * return_rate))
                        returned_value = round(returns * aov * 0.9, 2)

                        if orders > 0:
                            new_pct = np.random.beta(2,5)
                            new_customers = int(round(orders * new_pct))
                            returning_customers = orders - new_customers
                        else:
                            new_customers = 0
                            returning_customers = 0

                        ctr_actual = round((clicks / impressions) if impressions>0 else 0, 4)
                        cvr_actual = round((conversions / clicks) if clicks>0 else 0, 4)
                        cac = round(spend / conversions, 2) if conversions>0 else np.nan
                        roas = round(revenue / spend, 2) if spend>0 else np.nan
                        profit = round(revenue - cogs - returned_value - spend, 2)

                        rows.append({
                            "date": d.date().isoformat(),
                            "channel": channel,
                            "campaign": campaign,
                            "ad_set": ad_set,
                            "creative": creative,
                            "impressions": impressions,
                            "clicks": clicks,
                            "spend": spend,
                            "conversions": conversions,
                            "orders": orders,
                            "revenue": revenue,
                            "cogs": cogs,
                            "returns": returns,
                            "returned_value": returned_value,
                            "new_customers": new_customers,
                            "returning_customers": returning_customers,
                            "ctr": ctr_actual,
                            "cvr": cvr_actual,
                            "aov": round(aov, 2),
                            "cac": cac,
                            "roas": roas,
                            "profit": profit,
                            "funnel_step": "total"
                        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ------------- Styling & helpers -------------
PRIMARY_PURPLE = "#6B21A8"
ACCENT_GREEN = "#14B8A6"
CARD_BG = "#FAFAFB"
PAGE_BG = "#ffffff"
TEXT_COLOR = "#0f172a"

def inject_css():
    st.markdown(
        f"""
    <style>
    .stApp {{ background: {PAGE_BG}; color: {TEXT_COLOR}; }}
    .reportview-container .main .block-container{{padding-top:1.5rem; padding-left:2rem; padding-right:2rem;}}
    .topband {{
        background: linear-gradient(90deg,{PRIMARY_PURPLE} 0%, #7c3aed 100%);
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        margin-bottom: 6px;
    }}
    .kpi-large {{
        background: {CARD_BG};
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        color: {TEXT_COLOR};
    }}
    .kpi-small {{
        background: {CARD_BG};
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
        color: {TEXT_COLOR};
    }}
    .insight {{
        background: #f8fafc;
        padding: 12px;
        border-radius: 8px;
        color: {TEXT_COLOR};
    }}
    .kpi-label {{ font-size:14px; color:#374151; }}
    .kpi-value {{ font-size:28px; font-weight:700; color:{TEXT_COLOR}; }}
    .kpi-delta {{ font-size:13px; color:#059669; }}

    /* Sidebar label/selectbox and download button visibility on dark sidebar */
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .markdown-text-container,
    div[data-testid="stSidebar"] .css-1emrehy {
        color: #ffffff !important;
    }

    /* Download button styling so label text is visible */
    div[data-testid="stDownloadButton"] button {
        color: #ffffff !important;
        background-color: #262730 !important;
        border: 1px solid #ffffff40 !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background-color: #3a3b3c !important;
        color: #ffffff !important;
    }

    /* small safeguard for checkbox labels in sidebar */
    div[data-testid="stSidebar"] .stCheckbox label { color: #ffffff !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )

def compute_exec_kpis(df):
    total_revenue = df["revenue"].sum()
    total_spend = df["spend"].sum()
    mer = round(total_revenue / total_spend, 2) if total_spend>0 else np.nan
    total_cogs = df["cogs"].sum()
    gross_margin = round((total_revenue - total_cogs) / total_revenue, 4) if total_revenue>0 else np.nan
    total_conversions = df["conversions"].sum()
    total_new_cust = df["new_customers"].sum()
    cac = round(df["spend"].sum() / total_conversions, 2) if total_conversions>0 else np.nan
    ltv = round(total_revenue / max(1, total_new_cust), 2) if total_new_cust>0 else np.nan
    ltv_cac = round(ltv / cac, 2) if cac and cac>0 else np.nan
    profit = df["profit"].sum()
    return {
        "total_revenue": total_revenue,
        "total_spend": total_spend,
        "mer": mer,
        "gross_margin": gross_margin,
        "cac": cac,
        "ltv": ltv,
        "ltv_cac": ltv_cac,
        "profit": profit
    }

def compute_pop(cur, prev):
    if prev is None or prev == 0 or np.isnan(prev):
        return None
    try:
        return (cur - prev) / abs(prev)
    except:
        return None

def delta_html(cur, prev):
    pop = compute_pop(cur, prev)
    if pop is None:
        return ""
    sym = "▲" if pop > 0 else "▼"
    color = "#059669" if pop > 0 else "#dc2626"
    return f"<div style='color:{color}; font-weight:600'>{sym} {abs(pop)*100:.1f}% vs prev</div>"

def delta_html_inverted(cur, prev):
    pop = compute_pop(cur, prev)
    if pop is None:
        return ""
    sym = "▲" if pop > 0 else "▼"
    color = "#059669" if pop < 0 else "#dc2626"
    return f"<div style='color:{color}; font-weight:600'>{sym} {abs(pop)*100:.1f}% vs prev</div>"

# ------------- Plot builders (same as before) -------------
def plot_spend_revenue_trend(df):
    ts = df.groupby("date").agg({"revenue":"sum","spend":"sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ts["date"], y=ts["spend"], name="Spend", marker_color="#8b5cf6", yaxis="y2", opacity=0.6))
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["revenue"], name="Revenue", mode="lines+markers", line=dict(color=ACCENT_GREEN, width=3)))
    fig.update_layout(
        title=dict(text="Revenue & Marketing Spend Trend", x=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Revenue"),
        yaxis2=dict(title="Spend", overlaying="y", side="right"),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
        height=360
    )
    return fig

def plot_roas_by_channel(df, show_targets=False, targets=None):
    agg = df.groupby("channel").agg({"revenue":"sum","spend":"sum"}).reset_index()
    agg["roas"] = agg.apply(lambda r: (r["revenue"] / r["spend"]) if r["spend"]>0 else np.nan, axis=1).round(2)
    color_map = {"Meta":"#1D4ED8","Google":"#FACC15","Amazon":"#F59E0B","TikTok":"#F43F5E"}
    fig = px.bar(agg.sort_values("roas", ascending=False), x="roas", y="channel", orientation="h",
                 text=agg["roas"], color="channel", color_discrete_map=color_map)
    fig.update_layout(title_text="ROAS by Channel", template="plotly_white", height=320, margin=dict(l=120,t=50,b=20))
    fig.update_xaxes(title="ROAS (Revenue / Spend)")
    if show_targets and targets and "roas" in targets:
        fig.add_vline(x=targets["roas"], line_dash="dash", line_color="#374151", annotation_text=f"Target ROAS: {targets['roas']}", annotation_position="top right")
    return fig

def plot_funnel_bars(df):
    impressions = df["impressions"].sum()
    clicks = df["clicks"].sum()
    product_views = int(df["orders"].sum() * 3.5)
    add_to_cart = int(product_views * 0.25)
    purchases = int(df["orders"].sum())

    steps = pd.DataFrame({
        "step": ["Impressions", "Clicks", "Product Views", "Add to Cart", "Purchases"],
        "value": [impressions, clicks, product_views, add_to_cart, purchases]
    })

    if impressions > 0:
        steps["pct_of_impr"] = steps["value"] / impressions * 100
    else:
        steps["pct_of_impr"] = 0.0

    pct_prev = []
    prev_val = None
    for v in steps["value"]:
        if prev_val is None or prev_val == 0:
            pct_prev.append(100.0 if prev_val is None else 0.0)
        else:
            pct_prev.append(v / prev_val * 100)
        prev_val = v
    steps["pct_prev"] = pct_prev

    steps["label"] = steps.apply(lambda r: f"{int(r['value']):,}\n({r['pct_of_impr']:.2f}%)", axis=1)

    fig = px.bar(
        steps.sort_values("value", ascending=False).iloc[::-1],
        x="value",
        y="step",
        orientation="h",
        text="label",
    )

    customdata = np.column_stack((steps["value"], steps["pct_of_impr"], steps["pct_prev"]))
    customdata = customdata[steps.sort_values("value", ascending=False).index.values[::-1]]

    fig.update_traces(
        textposition="outside",
        marker_color="#93c5fd",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Count: %{customdata[0]:,}<br>"
            "% of impressions: %{customdata[1]:.2f}%<br>"
            "% of previous step: %{customdata[2]:.2f}%<extra></extra>"
        ),
        customdata=customdata
    )

    fig.update_layout(
        title_text="Marketing Funnel: Impression → Purchase",
        template="plotly_white",
        height=320,
        margin=dict(l=120, t=50, b=20),
        xaxis=dict(title="Count")
    )

    return fig

def plot_cac_trend(df):
    ts = df.groupby("date").apply(lambda x: pd.Series({"cac": x["spend"].sum() / x["conversions"].sum() if x["conversions"].sum()>0 else np.nan})).reset_index()
    ts = ts.sort_values("date")
    ts["cac_7d"] = ts["cac"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["cac"], mode="lines+markers", name="Daily CAC", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["cac_7d"], mode="lines", name="7-day avg", line=dict(dash='dash', width=3)))
    fig.update_layout(title_text="Customer Acquisition Cost (CAC) Trend", template="plotly_white", height=300, margin=dict(l=40,r=20,t=50,b=20))
    fig.update_yaxes(title="CAC")
    return fig

def plot_contribution_waterfall(df):
    rev = df["revenue"].sum()
    cogs = df["cogs"].sum()
    returns = df["returned_value"].sum()
    spend = df["spend"].sum()
    profit = rev - cogs - returns - spend
    measures = ["relative","relative","relative","relative","total"]
    fig = go.Figure(go.Waterfall(
        name="Contribution",
        orientation="v",
        measure=measures,
        x=["Revenue","COGS","Returns","Marketing Spend","Net Profit"],
        text=[f"{rev:,.0f}", f"-{cogs:,.0f}", f"-{returns:,.0f}", f"-{spend:,.0f}", f"{profit:,.0f}"],
        y=[rev, -cogs, -returns, -spend, profit],
        decreasing=dict(marker=dict(color="#ef4444")),
        increasing=dict(marker=dict(color="#10b981")),
        totals=dict(marker=dict(color="#6B21A8"))
    ))
    fig.update_layout(title_text="Contribution Margin Breakdown", template="plotly_white", height=360, margin=dict(l=40,r=20,t=60,b=20))
    return fig

# ------------- App layout -------------
st.set_page_config(page_title="Lifesight - Marketing Performance", layout="wide")
inject_css()

last_updated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
st.markdown(f"<div style='background: linear-gradient(90deg,{PRIMARY_PURPLE} 0%, #7c3aed 100%); color:white; padding:10px 16px; border-radius:8px;'><strong style='font-size:18px'>Lifesight</strong> — Marketing Performance Dashboard <span style='float:right; font-size:12px; opacity:0.9'>Last updated: {last_updated}</span></div>", unsafe_allow_html=True)

# Load data
df = generate_mock_data(months_before=6, months_after=0)

# Sidebar filters (granularity removed)
with st.sidebar:
    st.header("Filters")
    if st.button("Reset filters"):
        for k in ["channel","campaign","creative","start_date","end_date","attribution","sort_by","sort_asc"]:
            if k in st.session_state:
                del st.session_state[k]
        st.experimental_rerun()

    channel = st.selectbox("Channel", ["All"] + sorted(df["channel"].unique().tolist()), index=0, key="channel")
    campaign = st.selectbox("Campaign", ["All"] + sorted(df["campaign"].unique().tolist()), index=0, key="campaign")
    creative = st.selectbox("Creative", ["All"] + sorted(df["creative"].unique().tolist()), index=0, key="creative")

    st.markdown("### Date range")
    start_date = st.date_input("Start date", value=df["date"].min().date(), key="start_date")
    end_date = st.date_input("End date", value=df["date"].max().date(), key="end_date")

    st.markdown("---")
    # Granularity control removed per request
    st.selectbox("Attribution model (view only)", options=["Last click","Linear","Data-driven"], index=0, key="attribution")
    st.markdown("---")
    st.caption("Use filters to update the dashboard. ")

# subset data
subset = df.copy()
if channel != "All":
    subset = subset[subset["channel"]==channel]
if campaign != "All":
    subset = subset[subset["campaign"]==campaign]
if creative != "All":
    subset = subset[subset["creative"]==creative]
subset = subset[(subset["date"]>=pd.to_datetime(start_date)) & (subset["date"]<=pd.to_datetime(end_date))]

if subset.empty:
    st.warning("No data for the selected filters/date range. Adjust filters.")
    st.stop()

# compute PoP etc.
curr_start, curr_end = pd.to_datetime(start_date), pd.to_datetime(end_date)
period_days = (curr_end - curr_start).days + 1
prev_end = curr_start - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=period_days-1)
cur_kpis = compute_exec_kpis(subset)
prev_subset = df[(df["date"]>=prev_start) & (df["date"]<=prev_end)]
prev_kpis = compute_exec_kpis(prev_subset)

filter_summary = f"Filters — Channel: {channel}; Campaign: {campaign}; Creative: {creative}; Date: {start_date} to {end_date}; Attribution: {st.session_state.get('attribution','Last click')}"
st.caption(filter_summary)

# KPI cards (kept same)
left, right = st.columns([2,3], gap="large")
benchmarks = {"mer":5.0, "gpm":0.50, "roas":3.0, "cac":8.0}

with left:
    st.markdown("<div class='kpi-large'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-label'>Total Net Revenue</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>₹{cur_kpis['total_revenue']:,.0f}</div>", unsafe_allow_html=True)
    rev_delta = compute_pop(cur_kpis['total_revenue'], prev_kpis['total_revenue'])
    if rev_delta is None:
        st.markdown("<div class='kpi-delta'>No previous period</div>", unsafe_allow_html=True)
    else:
        sym = "▲" if rev_delta > 0 else "▼"
        color = "#059669" if rev_delta > 0 else "#dc2626"
        st.markdown(f"<div style='color:{color}; font-weight:600'>{sym} {abs(rev_delta)*100:.1f}% vs prev period</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    cols = st.columns(4, gap="medium")
    gpm = cur_kpis["gross_margin"]
    prev_gpm = prev_kpis["gross_margin"]
    gpm_delta = compute_pop(gpm, prev_gpm)
    with cols[0]:
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Gross Profit Margin</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{gpm*100:.1f}%</div>" if not np.isnan(gpm) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        if gpm_delta is not None:
            colc = "#059669" if gpm_delta>0 else "#dc2626"
            st.markdown(f"<div style='color:{colc}'> {'▲' if gpm_delta>0 else '▼'} {abs(gpm_delta)*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px; color:#6b7280'>Target: {benchmarks['gpm']*100:.0f}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    mer = cur_kpis["mer"]
    prev_mer = prev_kpis["mer"]
    mer_delta = compute_pop(mer, prev_mer)
    with cols[1]:
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Marketing Efficiency Ratio (MER)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{mer:.2f}</div>" if not np.isnan(mer) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        if mer_delta is not None:
            colc = "#059669" if mer_delta>0 else "#dc2626"
            st.markdown(f"<div style='color:{colc}'>{'▲' if mer_delta>0 else '▼'} {abs(mer_delta)*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px; color:#6b7280'>Target: {benchmarks['mer']:.1f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    ltv_cac = cur_kpis["ltv_cac"]
    prev_ltv_cac = prev_kpis["ltv_cac"]
    ltv_delta = compute_pop(ltv_cac, prev_ltv_cac)
    with cols[2]:
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>LTV : CAC Ratio</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{ltv_cac:.2f}</div>" if not np.isnan(ltv_cac) else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        if ltv_delta is not None:
            colc = "#059669" if ltv_delta>0 else "#dc2626"
            st.markdown(f"<div style='color:{colc}'>{'▲' if ltv_delta>0 else '▼'} {abs(ltv_delta)*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px; color:#6b7280'>Good: > 3.0</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    profit = cur_kpis["profit"]
    prev_profit = prev_kpis["profit"]
    profit_delta = compute_pop(profit, prev_profit)
    with cols[3]:
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Total Profit</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>₹{profit:,.0f}</div>", unsafe_allow_html=True)
        if profit_delta is not None:
            colc = "#059669" if profit_delta>0 else "#dc2626"
            st.markdown(f"<div style='color:{colc}'>{'▲' if profit_delta>0 else '▼'} {abs(profit_delta)*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Tabs and remaining dashboard content are unchanged from original — CMO tab below
tabs = st.tabs(["Executive (Overview)", "CMO View (Marketing)", "CFO View (Finance)"])

with tabs[1]:
    st.header("CMO View — Marketing Effectiveness & Diagnostics")

    st.subheader("CMO KPIs")
    c1, c2, c3, c4 = st.columns(4, gap="large")

    prev_impressions = prev_subset["impressions"].sum()
    prev_clicks = prev_subset["clicks"].sum()
    prev_spend = prev_subset["spend"].sum()
    prev_conversions = prev_subset["conversions"].sum()

    with c1:
        total_imp = subset['impressions'].sum()
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Total Impressions</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{total_imp:,}</div>", unsafe_allow_html=True)
        st.markdown(delta_html(total_imp, prev_impressions), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        ctr_val = subset['clicks'].sum() / subset['impressions'].sum() if subset['impressions'].sum() > 0 else 0
        prev_ctr = prev_clicks / prev_impressions if prev_impressions > 0 else None
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Click-Through Rate</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{ctr_val*100:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(delta_html(ctr_val, prev_ctr), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        cpm_val = subset['spend'].sum() / (subset['impressions'].sum()/1000) if subset['impressions'].sum() > 0 else np.nan
        prev_cpm = prev_spend / (prev_impressions/1000) if prev_impressions > 0 else None
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Avg. CPM</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{'₹{:.2f}'.format(cpm_val) if not np.isnan(cpm_val) else 'N/A'}</div>", unsafe_allow_html=True)
        st.markdown(delta_html_inverted(cpm_val, prev_cpm), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        cvr_val = subset['conversions'].sum() / subset['clicks'].sum() if subset['clicks'].sum() > 0 else 0
        prev_cvr = prev_conversions / prev_clicks if prev_clicks > 0 else None
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Avg. Conversion Rate</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{cvr_val*100:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(delta_html(cvr_val, prev_cvr), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("**ROAS by Channel** — quick comparison to guide budget allocation")
    fig_roas = plot_roas_by_channel(subset, show_targets=True, targets={"roas":3.0})
    st.plotly_chart(fig_roas, use_container_width=True)

    st.markdown("**Full Marketing Funnel** — identify drop-off points")
    fig_funnel = plot_funnel_bars(subset)
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("**Campaign & Creative Diagnostics** (Top performers)")
    diag = subset.groupby(["channel","campaign","ad_set","creative"]).agg({
        "spend":"sum","impressions":"sum","clicks":"sum","conversions":"sum","revenue":"sum"
    }).reset_index()
    diag["ctr"] = (diag["clicks"] / diag["impressions"]).round(4)
    diag["cvr"] = (diag["conversions"] / diag["clicks"]).round(4)
    diag["cpa"] = (diag["spend"] / diag["conversions"]).round(2).replace([np.inf, -np.inf], pd.NA)
    diag['roas'] = diag.apply(lambda r: (r['revenue']/r['spend']) if r['spend']>0 else np.nan, axis=1).round(2)

    # sorting UI (keep a selectbox; removed Ascending checkbox)
    sort_by = st.selectbox("Sort campaign table by", options=['revenue','spend','roas','cpa','ctr','cvr'], index=0, key='sort_by')
    # default direction: descending except cpa (lower is better)
    if sort_by == 'cpa':
        asc = True
    else:
        asc = False
    diag_sorted = diag.sort_values(by=sort_by, ascending=asc)

    try:
        styled = diag_sorted.head(200).style.format({
            'spend':'{:.2f}','impressions':'{:,}','clicks':'{:,}','conversions':'{:,}','revenue':'{:.2f}','ctr':'{:.2%}','cvr':'{:.2%}','cpa':'{:.2f}','roas':'{:.2f}'
        }).background_gradient(subset=['roas'], cmap='Greens')
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(diag_sorted.head(200), use_container_width=True)

    csv = diag_sorted.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download campaign diagnostics (CSV)", data=csv, file_name='campaign_diagnostics.csv', mime='text/csv')

# CFO tab remains same (unchanged)
with tabs[2]:
    st.header("CFO View — Financial Efficiency & Profitability")
    st.subheader("CFO KPIs")
    d1, d2, d3, d4 = st.columns(4, gap="large")

    prev_revenue = prev_subset["revenue"].sum()
    prev_spend = prev_subset["spend"].sum()
    prev_cogs = prev_subset["cogs"].sum()
    prev_orders = prev_subset["orders"].sum()
    prev_returns = prev_subset["returns"].sum()
    prev_aov = prev_subset["aov"].mean() if not prev_subset.empty else None

    with d1:
        if subset["spend"].sum() > 0:
            roi_val = (subset["revenue"].sum() - subset["spend"].sum()) / subset["spend"].sum()
        else:
            roi_val = None
        prev_roi = (prev_revenue - prev_spend) / prev_spend if prev_spend > 0 else None
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Marketing ROI</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{f'{roi_val*100:.2f}%'}" if roi_val is not None else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        if roi_val is not None:
            st.markdown(delta_html(roi_val, prev_roi), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with d2:
        if subset["revenue"].sum() > 0:
            gm_val = (subset["revenue"].sum() - subset["cogs"].sum()) / subset["revenue"].sum()
        else:
            gm_val = None
        prev_gm = (prev_revenue - prev_cogs) / prev_revenue if prev_revenue > 0 else None
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Gross Margin Rate</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{f'{gm_val*100:.2f}%'}" if gm_val is not None else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        if gm_val is not None:
            st.markdown(delta_html(gm_val, prev_gm), unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px; color:#6b7280'>Target: {0.50*100:.0f}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with d3:
        aov_val = subset['aov'].mean() if not subset.empty else None
        prev_aov_val = prev_aov
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Average Order Value</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>₹{aov_val:.2f}</div>" if aov_val is not None else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        if aov_val is not None:
            st.markdown(delta_html(aov_val, prev_aov_val), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with d4:
        refund_rate = subset["returns"].sum() / subset["orders"].sum() if subset["orders"].sum() > 0 else None
        prev_refund = prev_returns / prev_orders if prev_orders > 0 else None
        st.markdown("<div class='kpi-small'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>Refund & Return Rate</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{f'{refund_rate*100:.2f}%'}" if refund_rate is not None else "<div class='kpi-value'>N/A</div>", unsafe_allow_html=True)
        if refund_rate is not None:
            st.markdown(delta_html_inverted(refund_rate, prev_refund), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    fig_contrib = plot_contribution_waterfall(subset)
    st.plotly_chart(fig_contrib, use_container_width=True)
    st.markdown("**CAC Trend & Cost Efficiency** — monitor CAC vs historical performance")
    fig_cac = plot_cac_trend(subset)
    st.plotly_chart(fig_cac, use_container_width=True)

st.caption("Dashboard structure, KPI selection and layout follow the Lifesight assignment brief and executive needs.")
