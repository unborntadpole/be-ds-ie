"""
E-Commerce Personalized Recommendations — Data Analytics
=========================================================
Covers all 4 analytics types from the roleplay:
  1. Descriptive  — What happened?
  2. Diagnostic   — Why did it happen?
  3. Predictive   — What will happen?
  4. Prescriptive — What should we do?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

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
    "Rating":    [4, 2, 5, 4, 3, 4, 5, 2, 4, 4, 3, 3, 2, 4, 5, 4, 2, 4, 3, 5],
    "Price":     [1500, 4500, 25000, 1800, 600, 200, 1600, 4800, 26000, 1550,
                  1900, 650, 4700, 210, 24500, 1480, 4600, 1750, 580, 220],
}

df = pd.DataFrame(data)
df["Purchased_Bool"] = df["Purchased"].map({"Yes": 1, "No": 0})

print("=" * 60)
print("  E-COMMERCE DATA ANALYTICS — FULL REPORT")
print("=" * 60)
print(f"\n📦 Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────

BG      = "#0a0f1e"
SURFACE = "#111827"
CARD    = "#1a2235"
ACCENT  = "#4f8ef7"
PURPLE  = "#7c3aed"
GREEN   = "#22d3a0"
RED     = "#f45b5b"
AMBER   = "#fbbf24"
TEXT    = "#f0f4ff"
MUTED   = "#8b9ab8"

def style_fig(fig):
    fig.patch.set_facecolor(BG)

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(CARD)
    if title:
        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


# ═══════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE ANALYTICS — What happened?
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  1. DESCRIPTIVE ANALYTICS — What happened?")
print("=" * 60)

purchase_by_product = df[df["Purchased"] == "Yes"].groupby("Product").size().sort_values(ascending=False)
revenue_by_category = df[df["Purchased"] == "Yes"].groupby("Category")["Price"].sum().sort_values(ascending=False)
category_dist       = df.groupby("Category").size()
avg_rating          = df.groupby("Product")["Rating"].mean().sort_values(ascending=False)

print(f"\n📊 Purchase count by product:\n{purchase_by_product.to_string()}")
print(f"\n💰 Revenue by category (₹):\n{revenue_by_category.to_string()}")
print(f"\n📁 Product count by category:\n{category_dist.to_string()}")
print(f"\n⭐ Average rating by product:\n{avg_rating.round(2).to_string()}")

# ── Figure 1 ──
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
style_fig(fig1)
fig1.suptitle("DESCRIPTIVE ANALYTICS — What Happened?",
              color=TEXT, fontsize=14, fontweight="bold", y=1.01)

# Bar: purchases per product
colors_bar = [ACCENT if v == purchase_by_product.max() else PURPLE
              for v in purchase_by_product.values]
bars = axes[0].bar(purchase_by_product.index, purchase_by_product.values,
                   color=colors_bar, edgecolor=BG, linewidth=1.2)
style_ax(axes[0], "Most Purchased Products", "Product", "Count")
for bar, val in zip(bars, purchase_by_product.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 str(val), ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")

# Pie: category distribution
pie_colors = [ACCENT, AMBER, GREEN, RED]
wedges, texts, autotexts = axes[1].pie(
    category_dist.values, labels=category_dist.index,
    autopct="%1.0f%%", colors=pie_colors[:len(category_dist)],
    startangle=140, pctdistance=0.75,
    wedgeprops={"edgecolor": BG, "linewidth": 2}
)
for t in texts:   t.set_color(TEXT); t.set_fontsize(9)
for a in autotexts: a.set_color(BG); a.set_fontsize(8); a.set_fontweight("bold")
axes[1].set_title("Category Distribution", color=TEXT, fontsize=12, fontweight="bold", pad=12)
axes[1].set_facecolor(BG)

# Bar: revenue by category
rev_colors = [GREEN if i == 0 else PURPLE for i in range(len(revenue_by_category))]
bars2 = axes[2].bar(revenue_by_category.index, revenue_by_category.values,
                    color=rev_colors, edgecolor=BG, linewidth=1.2)
style_ax(axes[2], "Revenue by Category (₹)", "Category", "Revenue (₹)")
for bar, val in zip(bars2, revenue_by_category.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f"₹{val:,}", ha="center", va="bottom", color=TEXT, fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig("descriptive_analytics.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("\n✅ Saved: descriptive_analytics.png")


# ═══════════════════════════════════════════════════════════════
# 2. DIAGNOSTIC ANALYTICS — Why did it happen?
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  2. DIAGNOSTIC ANALYTICS — Why did it happen?")
print("=" * 60)

conversion = df.groupby("Product").agg(
    Views=("Viewed", "count"),
    Purchases=("Purchased_Bool", "sum"),
    Avg_Rating=("Rating", "mean"),
    Avg_Price=("Price", "mean")
)
conversion["Conversion_Rate_%"] = (conversion["Purchases"] / conversion["Views"] * 100).round(1)

print(f"\n📉 Conversion rates per product:\n{conversion.to_string()}")

watch_df = df[df["Product"] == "Watch"]
print(f"\n⌚ Watch deep-dive:")
print(f"   Views: {len(watch_df)} | Purchases: {watch_df['Purchased_Bool'].sum()} "
      f"| Conv. Rate: {watch_df['Purchased_Bool'].mean()*100:.0f}%")
print(f"   Avg Rating: {watch_df['Rating'].mean():.1f} | Avg Price: ₹{watch_df['Price'].mean():,.0f}")

# ── Figure 2 ──
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
style_fig(fig2)
fig2.suptitle("DIAGNOSTIC ANALYTICS — Why Did It Happen?",
              color=TEXT, fontsize=14, fontweight="bold", y=1.01)

# Conversion rate
conv_colors = [RED if r < 50 else GREEN for r in conversion["Conversion_Rate_%"]]
bars3 = axes[0].bar(conversion.index, conversion["Conversion_Rate_%"],
                    color=conv_colors, edgecolor=BG, linewidth=1.2)
axes[0].axhline(50, color=AMBER, linestyle="--", linewidth=1.2, label="50% threshold")
style_ax(axes[0], "Conversion Rate by Product (%)", "Product", "Conversion %")
axes[0].legend(facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=8)
for bar, val in zip(bars3, conversion["Conversion_Rate_%"]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val}%", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")

# Avg rating
rating_colors = [RED if r < 3 else AMBER if r < 4 else GREEN
                 for r in conversion["Avg_Rating"]]
bars4 = axes[1].bar(conversion.index, conversion["Avg_Rating"],
                    color=rating_colors, edgecolor=BG, linewidth=1.2)
axes[1].axhline(3, color=AMBER, linestyle="--", linewidth=1.2, label="Min acceptable (3★)")
style_ax(axes[1], "Average Rating by Product", "Product", "Avg Rating (★)")
axes[1].set_ylim(0, 6)
axes[1].legend(facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=8)
for bar, val in zip(bars4, conversion["Avg_Rating"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.1f}★", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")

# Views vs Purchases scatter
scatter_colors = [RED if p < 50 else GREEN
                  for p in conversion["Conversion_Rate_%"]]
sc = axes[2].scatter(conversion["Views"], conversion["Purchases"],
                     c=scatter_colors, s=200, edgecolors=BG, linewidths=1.5, zorder=3)
for prod, row in conversion.iterrows():
    axes[2].annotate(prod, (row["Views"], row["Purchases"]),
                     textcoords="offset points", xytext=(6, 4),
                     color=TEXT, fontsize=8)
axes[2].plot([0, 10], [0, 10], color=MUTED, linestyle="--", linewidth=1,
             label="Perfect conversion")
style_ax(axes[2], "Views vs Purchases (per product)", "Total Views", "Total Purchases")
axes[2].legend(facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=8)

# Annotation for Watch
w = conversion.loc["Watch"]
axes[2].annotate("⚠ Watch: low conversion",
                 xy=(w["Views"], w["Purchases"]),
                 xytext=(w["Views"] - 2.5, w["Purchases"] + 1.2),
                 color=RED, fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

plt.tight_layout()
plt.savefig("diagnostic_analytics.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("✅ Saved: diagnostic_analytics.png")


# ═══════════════════════════════════════════════════════════════
# 3. PREDICTIVE ANALYTICS — What will happen?
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  3. PREDICTIVE ANALYTICS — What will happen?")
print("=" * 60)

# Build co-purchase matrix (simple association)
purchase_log = {
    "U1":  ["Shoes", "Socks"],
    "U3":  ["Phone", "Earphones"],
    "U4":  ["Earphones"],
    "U6":  ["Socks"],
    "U7":  ["Shoes"],
    "U9":  ["Phone"],
    "U10": ["Shoes", "Socks"],
    "U11": ["Earphones"],
    "U14": ["Socks"],
    "U15": ["Phone", "Earphones"],
    "U16": ["Shoes"],
    "U18": ["Earphones"],
    "U20": ["Socks"],
}

co_purchase = defaultdict(lambda: defaultdict(int))
for user, prods in purchase_log.items():
    for i, p1 in enumerate(prods):
        for p2 in prods[i+1:]:
            co_purchase[p1][p2] += 1
            co_purchase[p2][p1] += 1

print("\n🔗 Co-purchase counts (association rules):")
for p, others in sorted(co_purchase.items()):
    for q, count in sorted(others.items(), key=lambda x: -x[1]):
        print(f"   {p} → {q}: {count} times")

# Build recommendation table
print("\n🤖 Recommendation Engine Output:")
recommendations = {}
for product in df["Product"].unique():
    if product in co_purchase:
        best = max(co_purchase[product], key=co_purchase[product].get)
        recommendations[product] = best
        print(f"   Bought {product} → Recommend {best}")

# ── Figure 3 ──
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
style_fig(fig3)
fig3.suptitle("PREDICTIVE ANALYTICS — What Will Happen?",
              color=TEXT, fontsize=14, fontweight="bold", y=1.01)

# Co-purchase heatmap
all_prods = sorted(set(list(co_purchase.keys()) +
                       [q for d in co_purchase.values() for q in d.keys()]))
matrix = pd.DataFrame(0, index=all_prods, columns=all_prods)
for p1, others in co_purchase.items():
    for p2, cnt in others.items():
        matrix.loc[p1, p2] = cnt

im = axes[0].imshow(matrix.values, cmap="Blues", aspect="auto")
axes[0].set_xticks(range(len(all_prods))); axes[0].set_xticklabels(all_prods, rotation=30, ha="right")
axes[0].set_yticks(range(len(all_prods))); axes[0].set_yticklabels(all_prods)
axes[0].tick_params(colors=MUTED, labelsize=9)
for i in range(len(all_prods)):
    for j in range(len(all_prods)):
        val = matrix.values[i][j]
        if val > 0:
            axes[0].text(j, i, str(val), ha="center", va="center",
                         color="white", fontsize=10, fontweight="bold")
style_ax(axes[0], "Co-Purchase Matrix (Association)")
plt.colorbar(im, ax=axes[0]).ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED)

# Recommendation flow (arrow diagram)
axes[1].set_xlim(0, 10); axes[1].set_ylim(0, 10)
style_ax(axes[1], "Recommendation Flow")
axes[1].axis("off")

flows = [
    ("👟 Shoes",     "→",  "🧦 Socks",     6.0, ACCENT, GREEN),
    ("📱 Phone",     "→",  "🎧 Earphones",  3.5, ACCENT, GREEN),
]
for source, arrow, target, y, sc, tc in flows:
    axes[1].text(1.5, y, source, ha="center", va="center", fontsize=13,
                 color=TEXT, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD,
                           edgecolor=sc, linewidth=2))
    axes[1].annotate("", xy=(5.5, y), xytext=(3.2, y),
                     arrowprops=dict(arrowstyle="-|>", color=AMBER,
                                     lw=2.5, mutation_scale=18))
    axes[1].text(5.0, y + 0.4, "Customers also buy", ha="center",
                 color=AMBER, fontsize=7.5)
    axes[1].text(7.5, y, target, ha="center", va="center", fontsize=13,
                 color=TEXT, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD,
                           edgecolor=tc, linewidth=2))

axes[1].text(5, 9.2, "Association-Based Predictions", ha="center",
             color=MUTED, fontsize=9)

plt.tight_layout()
plt.savefig("predictive_analytics.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("✅ Saved: predictive_analytics.png")


# ═══════════════════════════════════════════════════════════════
# 4. PRESCRIPTIVE ANALYTICS — What should we do?
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  4. PRESCRIPTIVE ANALYTICS — What should we do?")
print("=" * 60)

# Score each product: conversion + rating + revenue contribution
rev_total = df[df["Purchased"] == "Yes"]["Price"].sum()
product_score = conversion.copy()
revenue_series = df[df["Purchased"] == "Yes"].groupby("Product")["Price"].sum()
product_score["Revenue"] = revenue_series.reindex(product_score.index, fill_value=0)
product_score["Revenue_Share_%"] = (product_score["Revenue"] / rev_total * 100).round(1)

# Action labels
def prescribe(row):
    if row["Conversion_Rate_%"] < 40 and row["Avg_Rating"] < 3:
        return "Fix Rating + Reduce Price"
    elif row["Conversion_Rate_%"] < 40:
        return "Offer Discount"
    elif row["Avg_Rating"] >= 4 and row["Conversion_Rate_%"] >= 70:
        return "Bundle Deal / Upsell"
    else:
        return "Targeted Ads"

product_score["Action"] = product_score.apply(prescribe, axis=1)
print(f"\n🎯 Prescriptive actions per product:\n{product_score[['Conversion_Rate_%','Avg_Rating','Revenue_Share_%','Action']].to_string()}")

# ── Figure 4 ──
fig4, axes = plt.subplots(1, 3, figsize=(16, 5))
style_fig(fig4)
fig4.suptitle("PRESCRIPTIVE ANALYTICS — What Should We Do?",
              color=TEXT, fontsize=14, fontweight="bold", y=1.01)

# Revenue share
rev_plot = product_score["Revenue_Share_%"].sort_values(ascending=False)
rev_col = [GREEN if v == rev_plot.max() else ACCENT for v in rev_plot.values]
bars5 = axes[0].barh(rev_plot.index, rev_plot.values, color=rev_col, edgecolor=BG, linewidth=1.2)
style_ax(axes[0], "Revenue Share by Product (%)", "Revenue Share %", "Product")
for bar, val in zip(bars5, rev_plot.values):
    axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val}%", va="center", color=TEXT, fontsize=9, fontweight="bold")

# Action bubble chart
action_colors = {
    "Fix Rating + Reduce Price": RED,
    "Offer Discount": AMBER,
    "Bundle Deal / Upsell": GREEN,
    "Targeted Ads": ACCENT,
}
for prod, row in product_score.iterrows():
    color = action_colors.get(row["Action"], MUTED)
    rev_share = row["Revenue_Share_%"] if pd.notna(row["Revenue_Share_%"]) else 0.0
    bubble_size = max(float(rev_share) * 30 + 80, 60)
    axes[1].scatter(row["Conversion_Rate_%"], row["Avg_Rating"],
                    s=bubble_size,
                    color=color, edgecolors=BG, linewidths=1.5, alpha=0.85, zorder=3)
    axes[1].annotate(prod, (row["Conversion_Rate_%"], row["Avg_Rating"]),
                     textcoords="offset points", xytext=(6, 3),
                     color=TEXT, fontsize=8)
style_ax(axes[1], "Action Map (Bubble = Revenue Share)",
         "Conversion Rate %", "Avg Rating")
legend_patches = [mpatches.Patch(color=v, label=k) for k, v in action_colors.items()]
axes[1].legend(handles=legend_patches, facecolor=CARD, edgecolor=MUTED,
               labelcolor=TEXT, fontsize=7.5, loc="lower right")

# Priority action bar chart
action_counts = product_score["Action"].value_counts()
a_colors = [action_colors.get(a, MUTED) for a in action_counts.index]
bars6 = axes[2].bar(action_counts.index, action_counts.values,
                    color=a_colors, edgecolor=BG, linewidth=1.2)
style_ax(axes[2], "Recommended Actions Count", "Action", "# Products")
axes[2].tick_params(axis="x", labelsize=7)
plt.setp(axes[2].get_xticklabels(), rotation=15, ha="right")
for bar, val in zip(bars6, action_counts.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 str(val), ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("prescriptive_analytics.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("✅ Saved: prescriptive_analytics.png")


# ═══════════════════════════════════════════════════════════════
# SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  SUMMARY DASHBOARD")
print("=" * 60)

fig5, axes = plt.subplots(2, 3, figsize=(18, 10))
style_fig(fig5)
fig5.suptitle("E-COMMERCE ANALYTICS — FULL DASHBOARD",
              color=TEXT, fontsize=16, fontweight="bold", y=1.01)

# [0,0] Purchase count
b1 = axes[0][0].bar(purchase_by_product.index, purchase_by_product.values,
                    color=[ACCENT if i == 0 else PURPLE for i in range(len(purchase_by_product))],
                    edgecolor=BG)
style_ax(axes[0][0], "Purchases per Product")
for bar, v in zip(b1, purchase_by_product.values):
    axes[0][0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.05, str(v),
                    ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")

# [0,1] Conversion rate
b2 = axes[0][1].bar(conversion.index, conversion["Conversion_Rate_%"],
                    color=[RED if r < 50 else GREEN for r in conversion["Conversion_Rate_%"]],
                    edgecolor=BG)
axes[0][1].axhline(50, color=AMBER, linestyle="--", linewidth=1.2)
style_ax(axes[0][1], "Conversion Rate %")
for bar, v in zip(b2, conversion["Conversion_Rate_%"]):
    axes[0][1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.5, f"{v}%",
                    ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")

# [0,2] Avg rating
b3 = axes[0][2].bar(conversion.index, conversion["Avg_Rating"],
                    color=[RED if r < 3 else AMBER if r < 4 else GREEN
                           for r in conversion["Avg_Rating"]],
                    edgecolor=BG)
axes[0][2].axhline(3, color=AMBER, linestyle="--", linewidth=1.2)
axes[0][2].set_ylim(0, 6)
style_ax(axes[0][2], "Average Rating ★")
for bar, v in zip(b3, conversion["Avg_Rating"]):
    axes[0][2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.05, f"{v:.1f}★",
                    ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")

# [1,0] Pie: category distribution
wedges2, texts2, autotexts2 = axes[1][0].pie(
    category_dist.values, labels=category_dist.index,
    autopct="%1.0f%%", colors=pie_colors[:len(category_dist)],
    startangle=140, pctdistance=0.75,
    wedgeprops={"edgecolor": BG, "linewidth": 2}
)
for t in texts2:   t.set_color(TEXT); t.set_fontsize(9)
for a in autotexts2: a.set_color(BG); a.set_fontsize(8); a.set_fontweight("bold")
axes[1][0].set_title("Category Distribution", color=TEXT, fontsize=11, fontweight="bold", pad=10)
axes[1][0].set_facecolor(BG)

# [1,1] Revenue share
b4 = axes[1][1].barh(rev_plot.index, rev_plot.values,
                     color=[GREEN if v == rev_plot.max() else ACCENT for v in rev_plot.values],
                     edgecolor=BG)
style_ax(axes[1][1], "Revenue Share %", "Revenue %")
for bar, v in zip(b4, rev_plot.values):
    axes[1][1].text(bar.get_width()+.3, bar.get_y()+bar.get_height()/2,
                    f"{v}%", va="center", color=TEXT, fontsize=9, fontweight="bold")

# [1,2] Action recommendations
action_col2 = [action_colors.get(a, MUTED) for a in action_counts.index]
b5 = axes[1][2].bar(range(len(action_counts)), action_counts.values,
                    color=action_col2, edgecolor=BG)
axes[1][2].set_xticks(range(len(action_counts)))
axes[1][2].set_xticklabels(action_counts.index, rotation=15, ha="right", fontsize=7)
style_ax(axes[1][2], "Recommended Actions")
for bar, v in zip(b5, action_counts.values):
    axes[1][2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.03, str(v),
                    ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")

for row in axes:
    for ax in row:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED)
        for sp in ax.spines.values():
            sp.set_edgecolor(CARD)

plt.tight_layout()
plt.savefig("summary_dashboard.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("✅ Saved: summary_dashboard.png")

print("\n" + "=" * 60)
print("  ALL DONE! FILES SAVED:")
print("  📊 descriptive_analytics.png")
print("  🔍 diagnostic_analytics.png")
print("  🔮 predictive_analytics.png")
print("  🎯 prescriptive_analytics.png")
print("  🖥️  summary_dashboard.png")
print("=" * 60)