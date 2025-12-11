# Lifesight_MarketingPerformance_Dashboard_attribution_final.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ------------- Mock / original data generator (same as before) -------------
@st.cache_data
def generate_mock_data(months_before=6, seed=42):
    np.random.seed(seed)
    today = pd.to_datetime(datetime.utcnow().date())
    start = today - pd.DateOffset(months=months_before)
    dates = pd.date_range(start=start, end=today, freq="D")

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

# ------------- Attribution simulation helpers -------------
def simulate_user_journeys_from_subset(subset_df, max_touch_len=4, seed=123):
    np.random.seed(seed)
    total_conversions = int(subset_df["conversions"].sum())
    total_revenue = float(subset_df["revenue"].sum())
    total_spend = float(subset_df["spend"].sum())

    if total_conversions == 0:
        total_conversions = 200
        total_revenue = 200 * (subset_df["aov"].mean() if not subset_df.empty else 50)
        total_spend = 200 * 8.0

    channel_shares = subset_df.groupby("channel")["spend"].sum()
    if channel_shares.sum() <= 0:
        channels = ["Meta", "Google", "Amazon", "TikTok"]
        probs = [0.25, 0.25, 0.25, 0.25]
    else:
        channel_shares = channel_shares / channel_shares.sum()
        channels = channel_shares.index.tolist()
        probs = channel_shares.values

    aov_mean = subset_df["aov"].mean() if not subset_df.empty else 50
    aov_std = subset_df["aov"].std() if not subset_df.empty and subset_df["aov"].std() > 0 else aov_mean * 0.1

    journeys = []
    for i in range(total_conversions):
        n_touches = np.random.randint(1, max_touch_len + 1)
        touches = list(np.random.choice(channels, size=n_touches, p=probs))
        order_value = max(1.0, np.random.normal(aov_mean, aov_std))
        journeys.append({"order_id": f"ord_{i+1}", "touches": touches, "order_value": order_value})

    vals = np.array([j["order_value"] for j in journeys])
    total_vals = vals.sum()
    scale = 0 if total_vals <= 0 else (total_revenue / total_vals)
    for j in journeys:
        j["order_value"] = float(j["order_value"] * scale)

    spend_per_conv = total_spend / len(journeys) if len(journeys) > 0 else 0.0
    for j in journeys:
        j["spend_per_conv"] = spend_per_conv

    return pd.DataFrame(journeys)

def compute_attribution(journeys_df, model="Last click"):
    rows = []
    for idx, r in journeys_df.iterrows():
        touches = r["touches"]
        n = len(touches)
        order_value = r["order_value"]
        spend_per_conv = r["spend_per_conv"]
        spend_each = spend_per_conv / n if n > 0 else 0.0

        if model.lower().startswith("last"):
            last = touches[-1]
            rows.append({"channel": last, "rev": order_value, "spend": spend_each, "conv": 1})
            if n > 1:
                for t in touches[:-1]:
                    rows.append({"channel": t, "rev": 0.0, "spend": spend_each, "conv": 0})
        else:
            rev_each = order_value / n if n > 0 else 0.0
            for t in touches:
                rows.append({"channel": t, "rev": rev_each, "spend": spend_each, "conv": 1.0 / n})

    att = pd.DataFrame(rows)
    if att.empty:
        return pd.DataFrame(columns=["channel","attributed_revenue","attributed_spend","conversions_attributed"])
    agg = att.groupby("channel").agg(attributed_revenue=("rev","sum"),
                                     attributed_spend=("spend","sum"),
                                     conversions_attributed=("conv","sum")).reset_index()
    return agg

# ------------- Small utilities -------------
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

# ------------- Plot helpers -------------
def plot_roas_by_channel_from_attribution(agg_df, show_targets=False, target_roas=None):
    if agg_df is None or agg_df.empty:
        st.info("No attribution data available for the selected filters.")
        return None
    agg_df = agg_df.copy()
    agg_df["roas"] = agg_df.apply(lambda r: (r["attributed_revenue"] / r["attributed_spend"]) if r["attributed_spend"]>0 else np.nan, axis=1)
    agg_df = agg_df.sort_values("roas", ascending=False)
    fig = px.bar(agg_df, x="roas", y="channel", orientation="h", text=agg_df["roas"].round(2))
    fig.update_layout(title_text="ROAS by Channel (Attributed)", template="plotly_white", height=320, margin=dict(l=120,t=50,b=20))
    if show_targets and target_roas is not None:
        fig.add_vline(x=target_roas, line_dash="dash", line_color="#374151", annotation_text=f"Target ROAS: {target_roas}", annotation_position="top right")
    return fig

# ----------------- App layout -----------------
st.set_page_config(page_title="Lifesight - Attribution Demo", layout="wide")
st.title("Lifesight — Attribution demo (Last-click vs Linear)")

# ---- CSS fixes for visibility (sidebar labels + download button) ----
st.markdown(
    """
    <style>
    /* Make sidebar labels and markdown visible on dark sidebar */
    div[data-testid="stSidebar"] label, div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stDownloadButton, div[data-testid="stSidebar"] .markdown-text-container {
        color: #ffffff !important;
    }
    /* Ensure selectbox placeholder/inner text visible */
    div[data-testid="stSidebar"] .css-1emrehy { color: #ffffff !important; }

    /* Make text inside download button visible and style it */
    div[data-testid="stDownloadButton"] button {
        color: #ffffff !important;
        background-color: #262730 !important;
        border: 1px solid #ffffff50 !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background-color: #3a3b3c !important;
        color: #ffffff !important;
    }

    /* If there's an extra small checkbox label lingering, ensure it's visible and small */
    div[data-testid="stSidebar"] .stCheckbox label { color: #ffffff !important; font-size: 13px; }

    </style>
    """,
    unsafe_allow_html=True,
)

# Load base aggregated data
df = generate_mock_data(months_before=4)

# Sidebar filters (granularity removed)
with st.sidebar:
    st.header("Filters")
    channel = st.selectbox("Channel", ["All"] + sorted(df["channel"].unique().tolist()), index=0, key="s_channel")
    start_date = st.date_input("Start date", value=df["date"].min().date(), key="s_start")
    end_date = st.date_input("End date", value=df["date"].max().date(), key="s_end")
    attribution_model = st.selectbox("Attribution model", ["Last click","Linear"], index=0, key="s_attr")
    st.markdown("**Note:** This demo uses simulated path-level journeys derived from the aggregated data so the toggle shows real differences.")

# Subset aggregated data
subset = df.copy()
if channel != "All":
    subset = subset[subset["channel"] == channel]
subset = subset[(subset["date"] >= pd.to_datetime(start_date)) & (subset["date"] <= pd.to_datetime(end_date))]

if subset.empty:
    st.warning("No data for selected filters.")
    st.stop()

# compute exec KPIs (aggregated)
cur_kpis = compute_exec_kpis(subset)

st.subheader("Executive KPIs (aggregated)")
col1, col2, col3 = st.columns(3)
col1.metric("Revenue", f"₹{cur_kpis['total_revenue']:,.0f}")
col2.metric("Spend", f"₹{cur_kpis['total_spend']:,.0f}")
col3.metric("MER", f"{cur_kpis['mer']:.2f}")

st.markdown("---")
st.subheader("Attribution simulation — how channel ROAS/CAC change by model")

with st.spinner("Simulating user journeys and computing attribution..."):
    journeys = simulate_user_journeys_from_subset(subset)
    att_df = compute_attribution(journeys, model=attribution_model)

st.markdown("**Attributed channel performance**")
if att_df is None or att_df.empty:
    st.info("No attribution data generated.")
else:
    att_df["roas"] = att_df.apply(lambda r: (r["attributed_revenue"] / r["attributed_spend"]) if r["attributed_spend"]>0 else np.nan, axis=1)
    att_df["cac_attributed"] = att_df.apply(lambda r: (r["attributed_spend"] / r["conversions_attributed"]) if r["conversions_attributed"]>0 else np.nan, axis=1)
    st.dataframe(att_df.sort_values("roas", ascending=False).reset_index(drop=True).round(2))

    fig_roas = plot_roas_by_channel_from_attribution(att_df, show_targets=True, target_roas=3.0)
    if fig_roas:
        st.plotly_chart(fig_roas, use_container_width=True)

st.markdown("---")
st.subheader("Campaign diagnostics (attributed) — top rows")

# build campaign-level attributed view
channel_attr = att_df.set_index("channel")[["attributed_revenue","attributed_spend","conversions_attributed"]].to_dict(orient="index")

camp = subset.groupby(["channel","campaign"]).agg(spend=("spend","sum"), revenue=("revenue","sum"), impressions=("impressions","sum"),
                                                 clicks=("clicks","sum"), conversions=("conversions","sum")).reset_index()
camp["channel_spend_total"] = camp.groupby("channel")["spend"].transform("sum")
camp["camp_spend_share"] = camp.apply(lambda r: (r["spend"] / r["channel_spend_total"]) if r["channel_spend_total"]>0 else 0, axis=1)

def apply_attributed_to_campaigns(row):
    ch = row["channel"]
    if ch in channel_attr and channel_attr[ch]["attributed_revenue"]>0:
        attr_rev = channel_attr[ch]["attributed_revenue"] * row["camp_spend_share"]
        attr_spend = channel_attr[ch]["attributed_spend"] * row["camp_spend_share"]
        return pd.Series({"attr_revenue": attr_rev, "attr_spend": attr_spend})
    else:
        return pd.Series({"attr_revenue": 0.0, "attr_spend": 0.0})

camp[["attr_revenue","attr_spend"]] = camp.apply(apply_attributed_to_campaigns, axis=1)
camp["attr_roas"] = camp.apply(lambda r: (r["attr_revenue"]/r["attr_spend"]) if r["attr_spend"]>0 else np.nan, axis=1)
camp["attr_cpa"] = camp.apply(lambda r: (r["attr_spend"]/r["conversions"]) if r["conversions"]>0 else np.nan, axis=1)

# ------- SORT UI (smart sorting; no checkbox) -------
sort_options = {
    "Attributed revenue": "attr_revenue",
    "Attributed ROAS": "attr_roas",
    "Attributed CPA": "attr_cpa",
    "Original spend": "spend",
    "Original revenue": "revenue"
}
sort_choice = st.selectbox("Sort campaign table by", options=list(sort_options.keys()), index=0, key="camp_sort_choice")
sort_col = sort_options[sort_choice]

# smart default direction: attr_cpa -> ascending, else descending
if sort_col == "attr_cpa":
    ascending = True
else:
    ascending = False

camp_sorted = camp.sort_values(by=sort_col, ascending=ascending)

# display top 50 rows
display_df = camp_sorted.head(50).copy()
# format numbers for display
display_df = display_df.round(2)
st.dataframe(display_df, use_container_width=True)

# Download button (CSV) — styled via CSS above
csv_bytes = display_df.to_csv(index=False).encode("utf-8")
st.download_button(label="Download campaign diagnostics (CSV)", data=csv_bytes, file_name="campaign_diagnostics_attributed.csv", mime="text/csv")

st.markdown("""
**Notes:**
- Attribution is simulated from journeys built from the aggregated dataset. When you have real touch-level events (user_id, timestamp, touch type), replace the simulation with the real path dataset.
- In this demo spend-per-conversion is split equally across touches. For production, use real ad-level costs or a more nuanced spend allocation per touch.
""")
