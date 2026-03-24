"""
E-Commerce Analytics Dashboard — Streamlit App
===============================================
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  (clean light theme, card-based)
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default header */
#MainMenu, footer, header { visibility: hidden; }

/* Background */
.stApp { background: brown; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1.5px solid #e8eaf2;
}
[data-testid="stSidebar"] * { color: #3a3d52 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem !important;
    padding: 6px 0 !important;
}

/* Page title */
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #1a1d2e;
    margin-bottom: 2px;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #7c8299;
    margin-bottom: 20px;
}

/* Section headers */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4f8ef7;
    margin-bottom: 4px;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #1a1d2e;
    margin-bottom: 16px;
}

/* Metric cards */
.metric-row { display: flex; gap: 14px; margin-bottom: 18px; flex-wrap: wrap; }
.metric-card {
    background: #ffffff;
    border: 1.5px solid #e8eaf2;
    border-radius: 14px;
    padding: 18px 22px;
    flex: 1; min-width: 130px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.metric-card .m-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #1a1d2e;
    line-height: 1;
}
.metric-card .m-label {
    font-size: 0.78rem;
    color: #7c8299;
    margin-top: 6px;
}
.metric-card .m-delta {
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 4px;
}
.delta-up   { color: #16a34a; }
.delta-down { color: #dc2626; }

/* Insight pills */
.insight-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
.insight-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #eef2ff; color: #3730a3;
    border-radius: 99px; padding: 5px 14px;
    font-size: 0.8rem; font-weight: 500;
    border: 1px solid #c7d2fe;
}
.insight-pill.red   { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }
.insight-pill.green { background: #f0fdf4; color: #15803d; border-color: #bbf7d0; }
.insight-pill.amber { background: #fffbeb; color: #92400e; border-color: #fde68a; }

/* Action cards */
.action-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; }
.action-card {
    background: #ffffff; border: 1.5px solid #e8eaf2;
    border-radius: 12px; padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.action-card .ac-icon { font-size: 1.6rem; margin-bottom: 6px; }
.action-card .ac-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.95rem; color: #1a1d2e; }
.action-card .ac-desc  { font-size: 0.8rem; color: #7c8299; margin-top: 4px; line-height: 1.5; }
.action-card.highlight { border-color: #86efac; background: #f0fdf4; }

/* Flow diagram */
.flow-wrap { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
.flow-box {
    background: #eef2ff; border: 1.5px solid #c7d2fe;
    border-radius: 10px; padding: 10px 20px;
    font-weight: 600; font-size: 0.95rem; color: #3730a3;
}
.flow-box.result {
    background: #f0fdf4; border-color: #86efac; color: #15803d;
}
.flow-arrow { color: #4f8ef7; font-size: 1.4rem; }

/* Dataset table tweaks */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Divider */
.thin-divider { height: 1.5px; background: #e8eaf2; margin: 20px 0; border-radius: 2px; }

/* Chart container card */
.chart-card {
    background: #ffffff; border: 1.5px solid #e8eaf2;
    border-radius: 16px; padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# COLOUR PALETTE (matplotlib, light theme)
# ─────────────────────────────────────────────────────────────

BG     = "#ffffff"
SURF   = "#f5f7fc"
ACCENT = "#4f8ef7"
PURPLE = "#7c3aed"
GREEN  = "#22c55e"
RED    = "#ef4444"
AMBER  = "#f59e0b"
TEXT   = "#1a1d2e"
MUTED  = "#9ca3af"

def mpl_style(fig, ax_list=None):
    fig.patch.set_facecolor(BG)
    if ax_list is None:
        return
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(SURF)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor("#e8eaf2")
            sp.set_linewidth(1)

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    data = {
        "UserID":    ["U1","U2","U3","U4","U5","U6","U7","U8","U9","U10",
                      "U11","U12","U13","U14","U15","U16","U17","U18","U19","U20"],
        "Product":   ["Shoes","Watch","Phone","Earphones","Lipstick","Socks","Shoes",
                      "Watch","Phone","Shoes","Earphones","Lipstick","Watch","Socks",
                      "Phone","Shoes","Watch","Earphones","Lipstick","Socks"],
        "Category":  ["Fashion","Fashion","Electronics","Electronics","Beauty","Fashion",
                      "Fashion","Fashion","Electronics","Fashion","Electronics","Beauty",
                      "Fashion","Fashion","Electronics","Fashion","Fashion","Electronics",
                      "Beauty","Fashion"],
        "Viewed":    ["Yes"]*20,
        "Purchased": ["Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes",
                      "Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes"],
        "Rating":    [4,2,5,4,3,4,5,2,4,4,3,3,2,4,5,4,2,4,3,5],
        "Price":     [1500,4500,25000,1800,600,200,1600,4800,26000,1550,
                      1900,650,4700,210,24500,1480,4600,1750,580,220],
    }
    df = pd.DataFrame(data)
    df["Purchased_Bool"] = df["Purchased"].map({"Yes": 1, "No": 0})
    return df

df = load_data()

# ─────────────────────────────────────────────────────────────
# DERIVED TABLES
# ─────────────────────────────────────────────────────────────

@st.cache_data
def compute_analytics(df):
    purchase_by_product = (df[df["Purchased"]=="Yes"]
                           .groupby("Product").size()
                           .sort_values(ascending=False))
    revenue_by_category = (df[df["Purchased"]=="Yes"]
                           .groupby("Category")["Price"].sum()
                           .sort_values(ascending=False))
    category_dist = df.groupby("Category").size()
    avg_rating    = df.groupby("Product")["Rating"].mean().sort_values(ascending=False)

    conversion = df.groupby("Product").agg(
        Views=("Viewed","count"),
        Purchases=("Purchased_Bool","sum"),
        Avg_Rating=("Rating","mean"),
        Avg_Price=("Price","mean")
    )
    conversion["Conversion_Rate_%"] = (
        conversion["Purchases"] / conversion["Views"] * 100
    ).round(1)

    rev_total = df[df["Purchased"]=="Yes"]["Price"].sum()
    revenue_series = df[df["Purchased"]=="Yes"].groupby("Product")["Price"].sum()
    conversion["Revenue"] = revenue_series.reindex(conversion.index, fill_value=0)
    conversion["Revenue_Share_%"] = (conversion["Revenue"] / rev_total * 100).round(1)

    def prescribe(row):
        if row["Conversion_Rate_%"] < 40 and row["Avg_Rating"] < 3:
            return "Fix Rating + Reduce Price"
        elif row["Conversion_Rate_%"] < 40:
            return "Offer Discount"
        elif row["Avg_Rating"] >= 4 and row["Conversion_Rate_%"] >= 70:
            return "Bundle Deal / Upsell"
        else:
            return "Targeted Ads"

    conversion["Action"] = conversion.apply(prescribe, axis=1)

    purchase_log = {
        "U1": ["Shoes","Socks"], "U3": ["Phone","Earphones"],
        "U4": ["Earphones"],     "U6": ["Socks"],
        "U7": ["Shoes"],         "U9": ["Phone"],
        "U10":["Shoes","Socks"], "U11":["Earphones"],
        "U14":["Socks"],         "U15":["Phone","Earphones"],
        "U16":["Shoes"],         "U18":["Earphones"],
        "U20":["Socks"],
    }
    co = defaultdict(lambda: defaultdict(int))
    for _, prods in purchase_log.items():
        for i, p1 in enumerate(prods):
            for p2 in prods[i+1:]:
                co[p1][p2] += 1
                co[p2][p1] += 1
    co = {k: dict(v) for k, v in co.items()}

    return purchase_by_product, revenue_by_category, category_dist, avg_rating, conversion, co

(purchase_by_product, revenue_by_category,
 category_dist, avg_rating, conversion, co_purchase) = compute_analytics(df)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📊 Analytics Dashboard")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📊 Descriptive",
         "🔍 Diagnostic",
         "🔮 Predictive",
         "🎯 Prescriptive"],
        label_visibility="collapsed"
    )
    st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Project**")
    st.caption("Personalized Recommendations\nin E-commerce using\nData Analytics")
    st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Team**")
    members = [
        ("Shivam",    "Manager"),
        ("Samriddh",  "Data Analyst 1"),
        ("Siddharth", "Data Analyst 2"),
        ("Shifali",   "Marketing Exec"),
        ("Shubham",   "Product Manager"),
        ("Sushant",   "Customer"),
    ]
    for name, role in members:
        st.caption(f"**{name}** — {role}")

# ─────────────────────────────────────────────────────────────
# ── PAGE: OVERVIEW ──────────────────────────────────────────
# ─────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.markdown("<div class='page-title'>E-Commerce Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Personalized Recommendations using Data Analytics · College Exam Project</div>", unsafe_allow_html=True)

    # KPI row
    total_users    = len(df)
    total_purchases= df["Purchased_Bool"].sum()
    overall_conv   = round(df["Purchased_Bool"].mean() * 100, 1)
    total_revenue  = df[df["Purchased"]=="Yes"]["Price"].sum()

    st.markdown(f"""
    <div class='metric-row'>
      <div class='metric-card'>
        <div class='m-value'>{total_users}</div>
        <div class='m-label'>Total Users</div>
        <div class='m-delta delta-up'>↑ Dataset size</div>
      </div>
      <div class='metric-card'>
        <div class='m-value'>{total_purchases}</div>
        <div class='m-label'>Total Purchases</div>
        <div class='m-delta delta-up'>↑ out of {total_users} views</div>
      </div>
      <div class='metric-card'>
        <div class='m-value'>{overall_conv}%</div>
        <div class='m-label'>Overall Conversion</div>
        <div class='m-delta delta-up'>↑ avg across products</div>
      </div>
      <div class='metric-card'>
        <div class='m-value'>₹{total_revenue:,}</div>
        <div class='m-label'>Total Revenue</div>
        <div class='m-delta delta-up'>↑ from all purchases</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**📋 Full Dataset**")
        styled = df.drop(columns=["Purchased_Bool"]).style \
            .applymap(lambda v: "background-color:#fef2f2; color:#b91c1c"
                      if v == "No" else
                      "background-color:#f0fdf4; color:#15803d"
                      if v == "Yes" else "",
                      subset=["Purchased"]) \
            .applymap(lambda v: "color:#ef4444; font-weight:600"
                      if isinstance(v, int) and v <= 2 else
                      "color:#22c55e; font-weight:600"
                      if isinstance(v, int) and v >= 4 else "",
                      subset=["Rating"])
        st.dataframe(styled, use_container_width=True, height=420)

    with col2:
        st.markdown("**🔑 Key Insight**")
        st.info("Watch has **0% conversion** despite multiple views — high price and low ratings are the root cause.")
        st.success("Shoes + Socks and Phone + Earphones are natural **bundle pairs** identified via co-purchase analysis.")
        st.warning("Beauty category has **no purchases** — needs targeted discount campaigns.")

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        st.markdown("**📌 Analytics Stages**")
        for stage, icon, desc in [
            ("Descriptive",  "📊", "What happened?"),
            ("Diagnostic",   "🔍", "Why did it happen?"),
            ("Predictive",   "🔮", "What will happen?"),
            ("Prescriptive", "🎯", "What should we do?"),
        ]:
            st.markdown(f"{icon} **{stage}** — *{desc}*")


# ─────────────────────────────────────────────────────────────
# ── PAGE: DESCRIPTIVE ───────────────────────────────────────
# ─────────────────────────────────────────────────────────────

elif page == "📊 Descriptive":
    st.markdown("<div class='section-label'>Scene 3</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Descriptive Analytics — What Happened?</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-row'>
      <span class='insight-pill green'>👟 Shoes = most purchased</span>
      <span class='insight-pill'>💰 Electronics = highest revenue</span>
      <span class='insight-pill red'>⌚ Watches = high views, low purchases</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        mpl_style(fig, ax)
        colors = [ACCENT if i == 0 else PURPLE for i in range(len(purchase_by_product))]
        bars = ax.bar(purchase_by_product.index, purchase_by_product.values,
                      color=colors, edgecolor=BG, linewidth=1.2, width=0.55)
        for bar, v in zip(bars, purchase_by_product.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    str(v), ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
        ax.set_title("Most Purchased Products", fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_ylim(0, purchase_by_product.max() + 1.2)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        mpl_style(fig)
        pie_colors = [ACCENT, AMBER, GREEN, RED]
        wedges, texts, autotexts = ax.pie(
            category_dist.values, labels=category_dist.index,
            autopct="%1.0f%%", colors=pie_colors[:len(category_dist)],
            startangle=140, pctdistance=0.75,
            wedgeprops={"edgecolor": BG, "linewidth": 2}
        )
        for t in texts:     t.set_fontsize(10); t.set_color(TEXT)
        for a in autotexts: a.set_fontsize(9);  a.set_color("white"); a.set_fontweight("bold")
        ax.set_title("Category Distribution", fontsize=11, fontweight="bold", pad=10)
        ax.set_facecolor(BG)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        mpl_style(fig, ax)
        rev_colors = [GREEN if i == 0 else PURPLE for i in range(len(revenue_by_category))]
        bars2 = ax.bar(revenue_by_category.index, revenue_by_category.values,
                       color=rev_colors, edgecolor=BG, linewidth=1.2, width=0.55)
        for bar, v in zip(bars2, revenue_by_category.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                    f"₹{v:,}", ha="center", va="bottom", color=TEXT, fontsize=8, fontweight="bold")
        ax.set_title("Revenue by Category (₹)", fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Revenue (₹)", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        mpl_style(fig, ax)
        rating_colors = [RED if r < 3 else AMBER if r < 4 else GREEN for r in avg_rating.values]
        bars3 = ax.bar(avg_rating.index, avg_rating.values,
                       color=rating_colors, edgecolor=BG, linewidth=1.2, width=0.55)
        ax.axhline(3, color=AMBER, linestyle="--", linewidth=1.2, label="Min acceptable (3★)")
        ax.set_ylim(0, 6)
        for bar, v in zip(bars3, avg_rating.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    f"{v:.1f}★", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
        ax.set_title("Average Rating by Product", fontsize=11, fontweight="bold", pad=10)
        ax.legend(facecolor=BG, edgecolor="#e8eaf2", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ─────────────────────────────────────────────────────────────
# ── PAGE: DIAGNOSTIC ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────

elif page == "🔍 Diagnostic":
    st.markdown("<div class='section-label'>Scene 4</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Diagnostic Analytics — Why Did It Happen?</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-row'>
      <span class='insight-pill red'>⌚ Watch: 0% conversion rate</span>
      <span class='insight-pill red'>⭐ Avg rating 2.0 / 5</span>
      <span class='insight-pill amber'>💰 High price, poor descriptions</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        mpl_style(fig, ax)
        conv_colors = [RED if r < 50 else GREEN for r in conversion["Conversion_Rate_%"]]
        bars = ax.bar(conversion.index, conversion["Conversion_Rate_%"],
                      color=conv_colors, edgecolor=BG, linewidth=1.2, width=0.55)
        ax.axhline(50, color=AMBER, linestyle="--", linewidth=1.5, label="50% threshold")
        for bar, v in zip(bars, conversion["Conversion_Rate_%"]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{v}%", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
        ax.set_title("Conversion Rate by Product (%)", fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Conversion %", fontsize=9)
        ax.legend(facecolor=BG, edgecolor="#e8eaf2", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        mpl_style(fig, ax)
        sc_colors = [RED if r < 50 else GREEN for r in conversion["Conversion_Rate_%"]]
        ax.scatter(conversion["Views"], conversion["Purchases"],
                   c=sc_colors, s=180, edgecolors=BG, linewidths=1.5, zorder=3)
        for prod, row in conversion.iterrows():
            ax.annotate(prod, (row["Views"], row["Purchases"]),
                        textcoords="offset points", xytext=(6, 4),
                        color=TEXT, fontsize=8)
        ax.plot([0, 6], [0, 6], color=MUTED, linestyle="--", linewidth=1, label="Perfect conversion")
        w = conversion.loc["Watch"]
        ax.annotate("⚠ Watch", xy=(w["Views"], w["Purchases"]),
                    xytext=(w["Views"]-1.8, w["Purchases"]+1),
                    color=RED, fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
        ax.set_title("Views vs Purchases", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Views", fontsize=9)
        ax.set_ylabel("Purchases", fontsize=9)
        ax.legend(facecolor=BG, edgecolor="#e8eaf2", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Watch deep-dive
    st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
    st.markdown("**⌚ Watch Product — Root Cause Deep Dive**")
    c1, c2, c3, c4 = st.columns(4)
    watch_df = df[df["Product"] == "Watch"]
    c1.metric("Views",        len(watch_df))
    c2.metric("Purchases",    int(watch_df["Purchased_Bool"].sum()))
    c3.metric("Conversion",   "0%",    delta="↓ critically low", delta_color="inverse")
    c4.metric("Avg Rating",   f"{watch_df['Rating'].mean():.1f}★", delta="↓ below 3", delta_color="inverse")

    st.markdown("""
    <div class='action-grid' style='margin-top:12px'>
      <div class='action-card'>
        <div class='ac-icon'>⭐</div>
        <div class='ac-title'>Low Ratings (avg 2/5)</div>
        <div class='ac-desc'>Customer reviews are poor — product quality or expectations mismatch</div>
      </div>
      <div class='action-card'>
        <div class='ac-icon'>💰</div>
        <div class='ac-title'>High Price (avg ₹4,700)</div>
        <div class='ac-desc'>Not competitive vs similar products on the platform</div>
      </div>
      <div class='action-card'>
        <div class='ac-icon'>📝</div>
        <div class='ac-title'>Poor Descriptions</div>
        <div class='ac-desc'>Unclear product details reduce buyer confidence</div>
      </div>
      <div class='action-card'>
        <div class='ac-icon'>🖼️</div>
        <div class='ac-title'>Bad Product Images</div>
        <div class='ac-desc'>Low-quality photos fail to convert interested viewers</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ── PAGE: PREDICTIVE ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────

elif page == "🔮 Predictive":
    st.markdown("<div class='section-label'>Scene 5</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Predictive Analytics — What Will Happen?</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-row'>
      <span class='insight-pill green'>👟 Shoes buyers → likely buy 🧦 Socks</span>
      <span class='insight-pill green'>📱 Phone buyers → likely buy 🎧 Earphones</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Co-purchase heatmap
        all_prods = sorted(set(list(co_purchase.keys()) +
                               [q for d in co_purchase.values() for q in d.keys()]))
        matrix = pd.DataFrame(0, index=all_prods, columns=all_prods)
        for p1, others in co_purchase.items():
            for p2, cnt in others.items():
                matrix.loc[p1, p2] = cnt

        fig, ax = plt.subplots(figsize=(6, 4.5))
        mpl_style(fig, ax)
        im = ax.imshow(matrix.values, cmap="Blues", aspect="auto", vmin=0, vmax=matrix.values.max())
        ax.set_xticks(range(len(all_prods))); ax.set_xticklabels(all_prods, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(all_prods))); ax.set_yticklabels(all_prods, fontsize=9)
        for i in range(len(all_prods)):
            for j in range(len(all_prods)):
                v = matrix.values[i][j]
                if v > 0:
                    ax.text(j, i, str(v), ha="center", va="center",
                            color="white", fontsize=11, fontweight="bold")
        ax.set_title("Co-Purchase Matrix", fontsize=11, fontweight="bold", pad=10)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("**🔗 Recommendation Flow Diagram**")
        st.markdown("""
        <div style='margin-top:16px'>
          <div style='font-size:0.78rem;color:#7c8299;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px'>Pattern 1</div>
          <div class='flow-wrap'>
            <div class='flow-box'>👟 Customer buys<br><b>Shoes</b></div>
            <div class='flow-arrow'>➝</div>
            <div class='flow-box'>Likely also buys</div>
            <div class='flow-arrow'>➝</div>
            <div class='flow-box result'>🧦 Socks</div>
          </div>
          <div style='font-size:0.78rem;color:#7c8299;text-transform:uppercase;letter-spacing:0.1em;margin:18px 0 10px'>Pattern 2</div>
          <div class='flow-wrap'>
            <div class='flow-box'>📱 Customer buys<br><b>Phone</b></div>
            <div class='flow-arrow'>➝</div>
            <div class='flow-box'>Likely also buys</div>
            <div class='flow-arrow'>➝</div>
            <div class='flow-box result'>🎧 Earphones</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("**🤖 Recommendation Engine Output**")
        rec_data = {
            "If Customer Buys": ["Shoes", "Phone"],
            "Recommend":        ["Socks", "Earphones"],
            "Confidence":       ["High 🟢", "High 🟢"],
        }
        st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# ── PAGE: PRESCRIPTIVE ──────────────────────────────────────
# ─────────────────────────────────────────────────────────────

elif page == "🎯 Prescriptive":
    st.markdown("<div class='section-label'>Scene 6</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Prescriptive Analytics — What Should We Do?</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-row'>
      <span class='insight-pill green'>✅ Bundle Deals for Shoes + Socks, Phone + Earphones</span>
      <span class='insight-pill red'>⚠ Fix Watch: rating + price</span>
      <span class='insight-pill amber'>🎯 Targeted ads for Beauty</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    action_colors_map = {
        "Fix Rating + Reduce Price": RED,
        "Offer Discount":            AMBER,
        "Bundle Deal / Upsell":      GREEN,
        "Targeted Ads":              ACCENT,
    }

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        mpl_style(fig, ax)
        rev_plot = conversion["Revenue_Share_%"].sort_values(ascending=False)
        rev_col  = [GREEN if v == rev_plot.max() else ACCENT for v in rev_plot.values]
        bars = ax.barh(rev_plot.index, rev_plot.values, color=rev_col, edgecolor=BG, linewidth=1.2)
        for bar, v in zip(bars, rev_plot.values):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                    f"{v}%", va="center", color=TEXT, fontsize=9, fontweight="bold")
        ax.set_title("Revenue Share by Product (%)", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Revenue Share %", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        mpl_style(fig, ax)
        for prod, row in conversion.iterrows():
            color      = action_colors_map.get(row["Action"], MUTED)
            rev_share  = float(row["Revenue_Share_%"]) if pd.notna(row["Revenue_Share_%"]) else 0.0
            bubble_sz  = max(rev_share * 40 + 80, 60)
            ax.scatter(row["Conversion_Rate_%"], row["Avg_Rating"],
                       s=bubble_sz, color=color, edgecolors=BG, linewidths=1.5, alpha=0.85, zorder=3)
            ax.annotate(prod, (row["Conversion_Rate_%"], row["Avg_Rating"]),
                        textcoords="offset points", xytext=(6, 3),
                        color=TEXT, fontsize=8)
        legend_patches = [mpatches.Patch(color=v, label=k) for k, v in action_colors_map.items()]
        ax.legend(handles=legend_patches, facecolor=BG, edgecolor="#e8eaf2", fontsize=7.5)
        ax.set_title("Action Map (bubble = revenue share)", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Conversion Rate %", fontsize=9)
        ax.set_ylabel("Avg Rating", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
    st.markdown("**🎯 Recommended Actions per Product**")

    action_table = conversion[["Conversion_Rate_%", "Avg_Rating", "Revenue_Share_%", "Action"]].copy()
    action_table.columns = ["Conversion %", "Avg Rating", "Revenue Share %", "Recommended Action"]

    def color_action(val):
        palette = {
            "Fix Rating + Reduce Price": "background-color:#fef2f2;color:#b91c1c",
            "Offer Discount":            "background-color:#fffbeb;color:#92400e",
            "Bundle Deal / Upsell":      "background-color:#f0fdf4;color:#15803d",
            "Targeted Ads":              "background-color:#eff6ff;color:#1d4ed8",
        }
        return palette.get(val, "")

    st.dataframe(
        action_table.style.applymap(color_action, subset=["Recommended Action"]),
        use_container_width=True
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='action-grid'>
      <div class='action-card highlight'>
        <div class='ac-icon'>🏷️</div>
        <div class='ac-title'>Offer Targeted Discounts</div>
        <div class='ac-desc'>Reduce Watch price &amp; run promotions on Lipstick to boost conversions</div>
      </div>
      <div class='action-card highlight'>
        <div class='ac-icon'>📦</div>
        <div class='ac-title'>Create Combo Deals</div>
        <div class='ac-desc'>Bundle Shoes + Socks and Phone + Earphones for higher AOV</div>
      </div>
      <div class='action-card highlight'>
        <div class='ac-icon'>🤖</div>
        <div class='ac-title'>Personalized Recommendations</div>
        <div class='ac-desc'>Show relevant items based on each user's past views and purchases</div>
      </div>
      <div class='action-card highlight'>
        <div class='ac-icon'>🎯</div>
        <div class='ac-title'>Targeted Advertisements</div>
        <div class='ac-desc'>Re-target users who viewed but didn't purchase using smart ad campaigns</div>
      </div>
    </div>
    """, unsafe_allow_html=True)