"""
eda.py — Nairobi Property Affordability: Full Exploratory Data Analysis
========================================================================
Produces 8 publication-quality chart panels saved as PNGs + a summary report.

Charts:
  1. Top/Bottom 15 neighborhoods by avg price (horizontal bar)
  2. Price distribution (histogram + KDE)
  3. Price vs Bedrooms (box plot)
  4. Avg price per bedroom by neighborhood (bar, top 20)
  5. Affordability rank vs avg price (scatter with regression)
  6. Median vs Avg price by neighborhood (divergence chart)
  7. Price tier breakdown (pie / donut)
  8. Correlation heatmap

Run:
    python3 eda.py
    python3 eda.py --csv my_data.csv --out charts/
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
PALETTE = {
    "bg":         "#0a0c10",
    "surface":    "#12161e",
    "border":     "#1e2535",
    "text":       "#e8eaf0",
    "muted":      "#6b7494",
    "accent":     "#f0c040",
    "green":      "#4a8c2a",
    "teal":       "#1d6b5a",
    "orange":     "#c45c1a",
    "red":        "#b82030",
    "blue":       "#1a3a6b",
}

GRADIENT = ["#1a3a6b", "#1d6b5a", "#4a8c2a", "#b8860b", "#c45c1a", "#b82030"]

def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":   PALETTE["bg"],
        "axes.facecolor":     PALETTE["surface"],
        "axes.edgecolor":     PALETTE["border"],
        "axes.labelcolor":    PALETTE["text"],
        "axes.titlecolor":    PALETTE["text"],
        "axes.titlesize":     13,
        "axes.labelsize":     10,
        "axes.titlepad":      14,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.color":         PALETTE["border"],
        "grid.linewidth":     0.6,
        "xtick.color":        PALETTE["muted"],
        "ytick.color":        PALETTE["muted"],
        "xtick.labelsize":    8.5,
        "ytick.labelsize":    8.5,
        "text.color":         PALETTE["text"],
        "legend.facecolor":   PALETTE["surface"],
        "legend.edgecolor":   PALETTE["border"],
        "legend.labelcolor":  PALETTE["text"],
        "figure.titlesize":   16,
        "figure.titleweight": "bold",
        "font.family":        "monospace",
    })

def fmt_kes(x, pos=None):
    if x >= 1e6:   return f"KES {x/1e6:.1f}M"
    if x >= 1e3:   return f"KES {x/1e3:.0f}K"
    return f"KES {x:.0f}"

def short_name(loc, maxlen=22):
    name = str(loc).split(",")[0].strip()
    return name[:maxlen] + "…" if len(name) > maxlen else name


# ─────────────────────────────────────────────────────────────
# LOAD & VALIDATE
# ─────────────────────────────────────────────────────────────
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    required = ["location", "avg_price", "median_price", "median_bedrooms",
                "avg_price_per_bedroom", "median_price_per_bedroom", "affordability_rank"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.drop_duplicates(subset=["location"]).reset_index(drop=True)
    df["short_name"]     = df["location"].apply(short_name)
    df["neighborhood"]   = df["location"].apply(lambda x: str(x).split(",")[1].strip()
                                                  if "," in str(x) else str(x).split(",")[0].strip())
    df["price_tier"]     = pd.cut(
        df["affordability_rank"],
        bins=[0, df["affordability_rank"].quantile(0.33),
                 df["affordability_rank"].quantile(0.66),
                 df["affordability_rank"].max() + 1],
        labels=["Affordable", "Mid-Range", "Premium"],
        include_lowest=True
    )
    df["price_gap_pct"]  = (df["avg_price"] - df["median_price"]) / df["median_price"] * 100
    return df


# ─────────────────────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────────────────────
def print_summary(df):
    city_avg    = df["avg_price"].mean()
    city_median = df["median_price"].median()
    cheapest    = df.loc[df["affordability_rank"].idxmin()]
    priciest    = df.loc[df["affordability_rank"].idxmax()]
    best_ppb    = df.loc[df["avg_price_per_bedroom"].idxmin()]

    print("\n" + "═"*56)
    print("  NAIROBI PROPERTY MARKET — EDA SUMMARY")
    print("═"*56)
    print(f"  Locations analysed   : {len(df)}")
    print(f"  City avg price       : KES {city_avg/1e6:.2f}M")
    print(f"  City median price    : KES {city_median/1e6:.2f}M")
    print(f"  Price range          : KES {df['avg_price'].min()/1e6:.1f}M – {df['avg_price'].max()/1e6:.1f}M")
    print(f"  Avg price/bedroom    : KES {df['avg_price_per_bedroom'].mean()/1e6:.2f}M")
    print()
    print(f"  🟢 Most affordable   : {cheapest['short_name']}")
    print(f"                         KES {cheapest['avg_price']/1e6:.1f}M avg")
    print(f"  🔴 Most premium      : {priciest['short_name']}")
    print(f"                         KES {priciest['avg_price']/1e6:.1f}M avg")
    print(f"  💡 Best value/bed    : {best_ppb['short_name']}")
    print(f"                         KES {best_ppb['avg_price_per_bedroom']/1e6:.2f}M/bedroom")
    print()

    tier_counts = df["price_tier"].value_counts()
    for tier in ["Affordable", "Mid-Range", "Premium"]:
        n = tier_counts.get(tier, 0)
        print(f"  {tier:12} : {n:3} locations ({n/len(df)*100:.0f}%)")
    print("═"*56 + "\n")


# ─────────────────────────────────────────────────────────────
# CHART 1: Top / Bottom 15 by avg price
# ─────────────────────────────────────────────────────────────
def chart_top_bottom(df, out_dir):
    fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor(PALETTE["bg"])

    top15 = df.nlargest(15, "avg_price").sort_values("avg_price")
    bot15 = df.nsmallest(15, "avg_price").sort_values("avg_price")

    for ax, data, title, color in [
        (ax_top, top15, "15 Most Expensive Neighborhoods", PALETTE["red"]),
        (ax_bot, bot15, "15 Most Affordable Neighborhoods", PALETTE["green"]),
    ]:
        bars = ax.barh(data["short_name"], data["avg_price"],
                       color=color, alpha=0.85, height=0.65,
                       edgecolor=PALETTE["border"], linewidth=0.5)

        for bar, (_, row) in zip(bars, data.iterrows()):
            ax.text(bar.get_width() + data["avg_price"].max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    fmt_kes(row["avg_price"]),
                    va="center", ha="left", fontsize=8,
                    color=PALETTE["muted"])

            ax.text(bar.get_width() * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"#{int(row['affordability_rank'])}",
                    va="center", ha="left", fontsize=7.5,
                    color="white", alpha=0.7)

        ax.set_title(title, color=PALETTE["text"], pad=14, fontsize=12)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_kes))
        ax.set_xlim(0, data["avg_price"].max() * 1.22)
        ax.tick_params(axis="x", rotation=30)
        ax.set_facecolor(PALETTE["surface"])

        # City avg line
        city_avg = df["avg_price"].mean()
        ax.axvline(city_avg, color=PALETTE["accent"], linestyle="--",
                   linewidth=1.2, alpha=0.7)
        ax.text(city_avg, ax.get_ylim()[1] * 0.98,
                " city avg", color=PALETTE["accent"], fontsize=7.5, va="top")

    fig.suptitle("Nairobi Property Prices — Extremes", y=1.01,
                 color=PALETTE["text"], fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "01_top_bottom_prices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────
# CHART 2: Price distribution
# ─────────────────────────────────────────────────────────────
def chart_distribution(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    # Histogram + KDE: avg_price
    ax = axes[0]
    vals = df["avg_price"] / 1e6
    ax.hist(vals, bins=20, color=PALETTE["teal"], alpha=0.7,
            edgecolor=PALETTE["border"], linewidth=0.5)
    ax2 = ax.twinx()
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals)
        xs = np.linspace(vals.min(), vals.max(), 300)
        ax2.plot(xs, kde(xs), color=PALETTE["accent"], linewidth=2)
        ax2.set_ylabel("Density", color=PALETTE["muted"], fontsize=9)
        ax2.tick_params(axis="y", colors=PALETTE["muted"])
        ax2.set_facecolor(PALETTE["surface"])
        ax2.spines["top"].set_visible(False)
    except ImportError:
        pass

    ax.axvline(vals.mean(),   color=PALETTE["accent"], linestyle="--",
               linewidth=1.5, label=f"Mean: KES {vals.mean():.1f}M")
    ax.axvline(vals.median(), color=PALETTE["green"], linestyle=":",
               linewidth=1.5, label=f"Median: KES {vals.median():.1f}M")
    ax.set_xlabel("Average Price (KES Millions)")
    ax.set_ylabel("Count")
    ax.set_title("Price Distribution")
    ax.legend(fontsize=8)
    ax.set_facecolor(PALETTE["surface"])

    # Histogram: price per bedroom
    ax = axes[1]
    vals2 = df["avg_price_per_bedroom"] / 1e6
    colors = [GRADIENT[int(i / len(df) * (len(GRADIENT)-1))]
              for i in range(len(df))]
    n, bins, patches = ax.hist(vals2, bins=20, edgecolor=PALETTE["border"],
                                linewidth=0.5, color=PALETTE["orange"], alpha=0.8)
    ax.axvline(vals2.mean(), color=PALETTE["accent"], linestyle="--",
               linewidth=1.5, label=f"Mean: KES {vals2.mean():.2f}M")
    ax.axvline(vals2.median(), color=PALETTE["green"], linestyle=":",
               linewidth=1.5, label=f"Median: KES {vals2.median():.2f}M")
    ax.set_xlabel("Avg Price per Bedroom (KES Millions)")
    ax.set_ylabel("Count")
    ax.set_title("Price-per-Bedroom Distribution")
    ax.legend(fontsize=8)
    ax.set_facecolor(PALETTE["surface"])

    fig.suptitle("Price Distributions", color=PALETTE["text"],
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "02_price_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────
# CHART 3: Price vs Bedrooms (box plot)
# ─────────────────────────────────────────────────────────────
def chart_bedrooms_box(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor(PALETTE["bg"])

    bed_order = sorted(df["median_bedrooms"].dropna().unique())

    for ax, col, title, color in [
        (axes[0], "avg_price",             "Avg Price",         PALETTE["teal"]),
        (axes[1], "avg_price_per_bedroom",  "Avg Price / Bedroom", PALETTE["orange"]),
    ]:
        groups = [df[df["median_bedrooms"] == b][col].values / 1e6 for b in bed_order]
        bp = ax.boxplot(groups, labels=[f"{int(b)} bd" for b in bed_order],
                        patch_artist=True, notch=False,
                        medianprops=dict(color=PALETTE["accent"], linewidth=2),
                        whiskerprops=dict(color=PALETTE["muted"]),
                        capprops=dict(color=PALETTE["muted"]),
                        flierprops=dict(marker="o", markerfacecolor=PALETTE["red"],
                                        markersize=4, alpha=0.6))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        # Overlay scatter
        for i, (b, g) in enumerate(zip(bed_order, groups)):
            jitter = np.random.uniform(-0.15, 0.15, size=len(g))
            ax.scatter(i + 1 + jitter, g, alpha=0.5, s=18,
                       color=PALETTE["accent"], zorder=3)

        ax.set_ylabel("KES Millions")
        ax.set_title(f"{title} by Bedroom Count")
        ax.set_facecolor(PALETTE["surface"])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))

    fig.suptitle("Price vs Bedroom Count", color=PALETTE["text"],
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "03_price_by_bedrooms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────
# CHART 4: Price per bedroom — top 25 neighborhoods
# ─────────────────────────────────────────────────────────────
def chart_ppb_ranking(df, out_dir):
    top = df.nsmallest(25, "avg_price_per_bedroom").sort_values("avg_price_per_bedroom")
    n   = len(top)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(PALETTE["bg"])

    norm_vals = (top["avg_price_per_bedroom"] - top["avg_price_per_bedroom"].min()) / \
                (top["avg_price_per_bedroom"].max() - top["avg_price_per_bedroom"].min())

    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        "afford", ["#4a8c2a", "#b8860b", "#b82030"]
    )
    bar_colors = [cmap(v) for v in norm_vals]

    bars = ax.barh(top["short_name"], top["avg_price_per_bedroom"] / 1e6,
                   color=bar_colors, height=0.7,
                   edgecolor=PALETTE["border"], linewidth=0.4)

    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                fmt_kes(row["avg_price_per_bedroom"]),
                va="center", ha="left", fontsize=8, color=PALETTE["muted"])

        ax.text(0.04,
                bar.get_y() + bar.get_height() / 2,
                f"  {int(row['median_bedrooms'])}bd",
                va="center", ha="left", fontsize=7.5,
                color="white", alpha=0.65)

    ax.set_xlabel("Avg Price per Bedroom (KES Millions)")
    ax.set_title("Best Value per Bedroom — Top 25 Locations\n"
                 "(Sorted by lowest price/bedroom)", fontsize=12)
    ax.set_facecolor(PALETTE["surface"])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.set_xlim(0, top["avg_price_per_bedroom"].max() / 1e6 * 1.25)

    fig.suptitle("Value-per-Bedroom Analysis", color=PALETTE["text"],
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "04_price_per_bedroom.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────
# CHART 5: Rank vs Price scatter
# ─────────────────────────────────────────────────────────────
def chart_rank_scatter(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor(PALETTE["bg"])

    tier_colors = {"Affordable": PALETTE["green"],
                   "Mid-Range":  PALETTE["accent"],
                   "Premium":    PALETTE["red"]}

    for ax, xcol, xtitle in [
        (axes[0], "avg_price",            "Avg Price (KES)"),
        (axes[1], "avg_price_per_bedroom", "Avg Price / Bedroom (KES)"),
    ]:
        for tier, color in tier_colors.items():
            mask = df["price_tier"] == tier
            sub  = df[mask]
            ax.scatter(sub[xcol] / 1e6, sub["affordability_rank"],
                       c=color, alpha=0.75, s=55, label=tier,
                       edgecolors=PALETTE["border"], linewidths=0.4, zorder=3)

        # Regression line
        x = df[xcol].values / 1e6
        y = df["affordability_rank"].values
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, m * xs + b, color=PALETTE["muted"],
                linewidth=1.5, linestyle="--", alpha=0.6, label="Trend")

        # Annotate outliers
        for _, row in df.nlargest(3, xcol).iterrows():
            ax.annotate(row["short_name"],
                        (row[xcol] / 1e6, row["affordability_rank"]),
                        textcoords="offset points", xytext=(6, -3),
                        fontsize=7, color=PALETTE["muted"], alpha=0.9)

        ax.set_xlabel(f"{xtitle} (Millions)")
        ax.set_ylabel("Affordability Rank  (lower = cheaper)")
        ax.set_title(f"Rank vs {xtitle}")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
        ax.legend(fontsize=8)
        ax.set_facecolor(PALETTE["surface"])

    fig.suptitle("Affordability Rank vs Price", color=PALETTE["text"],
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "05_rank_vs_price_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────
# CHART 6: Avg vs Median divergence (price gap reveals skew)
# ─────────────────────────────────────────────────────────────
def chart_avg_median_gap(df, out_dir):
    top = df.reindex(df["price_gap_pct"].abs().nlargest(20).index)
    top = top.sort_values("price_gap_pct")

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(PALETTE["bg"])

    colors = [PALETTE["red"] if v > 0 else PALETTE["green"] for v in top["price_gap_pct"]]
    bars = ax.barh(top["short_name"], top["price_gap_pct"],
                   color=colors, height=0.7, alpha=0.82,
                   edgecolor=PALETTE["border"], linewidth=0.4)

    for bar, (_, row) in zip(bars, top.iterrows()):
        x_pos = bar.get_width()
        sign  = "+" if x_pos >= 0 else ""
        ax.text(x_pos + (0.5 if x_pos >= 0 else -0.5),
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{x_pos:.1f}%",
                va="center",
                ha="left" if x_pos >= 0 else "right",
                fontsize=8.5, color=PALETTE["text"])

    ax.axvline(0, color=PALETTE["border"], linewidth=1)
    ax.set_xlabel("(Avg Price − Median Price) / Median Price  ×  100%")
    ax.set_title("Avg vs Median Price Gap\n"
                 "Positive = avg pulled up by luxury outliers  |  "
                 "Negative = avg pulled down by budget stock",
                 fontsize=11)
    ax.set_facecolor(PALETTE["surface"])

    red_p  = mpatches.Patch(color=PALETTE["red"],   label="Avg > Median (outlier luxury)")
    grn_p  = mpatches.Patch(color=PALETTE["green"],  label="Avg < Median (outlier budget)")
    ax.legend(handles=[red_p, grn_p], fontsize=9, loc="lower right")

    fig.suptitle("Price Skew by Neighborhood", color=PALETTE["text"],
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "06_avg_vs_median_gap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────
# CHART 7: Tier breakdown donut
# ─────────────────────────────────────────────────────────────
def chart_tier_donut(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(PALETTE["bg"])

    tier_colors = [PALETTE["green"], PALETTE["accent"], PALETTE["red"]]
    tier_order  = ["Affordable", "Mid-Range", "Premium"]
    counts = df["price_tier"].value_counts().reindex(tier_order).fillna(0)

    # Donut: location count
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=None,
        autopct="%1.0f%%", startangle=90,
        colors=tier_colors,
        pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor=PALETTE["bg"], linewidth=2),
    )
    for at in autotexts:
        at.set_color(PALETTE["bg"])
        at.set_fontsize(11)
        at.set_fontweight("bold")

    ax.legend(
        handles=[mpatches.Patch(color=c, label=f"{t} ({int(n)})")
                 for c, t, n in zip(tier_colors, tier_order, counts.values)],
        loc="lower center", bbox_to_anchor=(0.5, -0.12),
        ncol=3, fontsize=9, framealpha=0.3,
    )
    ax.set_title("Locations by Price Tier")
    ax.text(0, 0, f"{len(df)}\nlocs", ha="center", va="center",
            fontsize=14, fontweight="bold", color=PALETTE["text"])

    # Donut: avg price by tier
    ax2 = axes[1]
    tier_avgs = df.groupby("price_tier")["avg_price"].mean().reindex(tier_order)
    wedges2, texts2, autotexts2 = ax2.pie(
        tier_avgs.values, labels=None,
        autopct=lambda p: f"KES {tier_avgs.values[int(round(p/100*len(tier_avgs)))-1]/1e6:.1f}M"
                          if p > 5 else "",
        startangle=90,
        colors=tier_colors,
        pctdistance=0.72,
        wedgeprops=dict(width=0.55, edgecolor=PALETTE["bg"], linewidth=2),
    )
    for at in autotexts2:
        at.set_color(PALETTE["bg"])
        at.set_fontsize(9)
        at.set_fontweight("bold")

    ax2.set_title("Avg Price Share by Tier")
    ax2.text(0, 0, "Avg\nPrice", ha="center", va="center",
             fontsize=11, color=PALETTE["text"])

    fig.suptitle("Market Tier Breakdown", color=PALETTE["text"],
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "07_tier_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────
# CHART 8: Correlation heatmap
# ─────────────────────────────────────────────────────────────
def chart_correlation(df, out_dir):
    num_cols = ["avg_price", "median_price", "median_bedrooms",
                "avg_price_per_bedroom", "median_price_per_bedroom",
                "affordability_rank", "price_gap_pct"]
    labels = ["Avg Price", "Median Price", "Bedrooms (med)",
              "Avg/Bedroom", "Median/Bedroom", "Afford. Rank", "Price Gap %"]

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor(PALETTE["bg"])

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
        annot=True, fmt=".2f", linewidths=1,
        linecolor=PALETTE["bg"], square=True,
        ax=ax,
        cbar_kws={"shrink": 0.7, "pad": 0.02},
        annot_kws={"size": 9, "color": PALETTE["text"]},
        xticklabels=labels, yticklabels=labels,
    )

    ax.set_title("Feature Correlation Matrix", pad=16)
    ax.set_facecolor(PALETTE["surface"])
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    ax.collections[0].colorbar.ax.tick_params(colors=PALETTE["muted"])

    fig.suptitle("Correlation Heatmap — Nairobi Property Data",
                 color=PALETTE["text"], fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "08_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────
# CHART 9: Neighborhood comparison — full ranked strip
# ─────────────────────────────────────────────────────────────
def chart_full_ranking(df, out_dir):
    df_sorted = df.sort_values("avg_price", ascending=False).reset_index(drop=True)
    n = len(df_sorted)

    fig, ax = plt.subplots(figsize=(10, max(12, n * 0.28)))
    fig.patch.set_facecolor(PALETTE["bg"])

    norm = (df_sorted["avg_price"] - df_sorted["avg_price"].min()) / \
           (df_sorted["avg_price"].max() - df_sorted["avg_price"].min())

    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        "g2r", [PALETTE["green"], PALETTE["accent"], PALETTE["red"]]
    )
    bar_colors = [cmap(v) for v in norm]

    bars = ax.barh(range(n), df_sorted["avg_price"] / 1e6,
                   color=bar_colors, height=0.75,
                   edgecolor=PALETTE["bg"], linewidth=0.3)

    ax.set_yticks(range(n))
    ax.set_yticklabels(df_sorted["short_name"], fontsize=7.5)
    ax.invert_yaxis()

    for i, (bar, (_, row)) in enumerate(zip(bars, df_sorted.iterrows())):
        ax.text(bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                fmt_kes(row["avg_price"]),
                va="center", ha="left", fontsize=6.5, color=PALETTE["muted"])

    city_avg = df["avg_price"].mean() / 1e6
    ax.axvline(city_avg, color=PALETTE["accent"], linestyle="--",
               linewidth=1.2, alpha=0.8)
    ax.text(city_avg + 0.05, 0, " city avg",
            color=PALETTE["accent"], fontsize=8, va="top")

    ax.set_xlabel("Average Price (KES Millions)")
    ax.set_title("All Neighborhoods — Full Price Ranking", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    ax.set_facecolor(PALETTE["surface"])
    ax.set_xlim(0, df_sorted["avg_price"].max() / 1e6 * 1.25)

    fig.suptitle("Complete Nairobi Neighborhood Price Ranking",
                 color=PALETTE["text"], fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "09_full_price_ranking.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Nairobi Property EDA")
    parser.add_argument("--csv", default="location_summary.csv")
    parser.add_argument("--out", default="charts")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    apply_dark_style()

    print(f"\n📊  Loading: {args.csv}")
    df = load_data(args.csv)
    print_summary(df)

    print(f"🎨  Generating charts → {args.out}/\n")

    chart_top_bottom(df, args.out)
    chart_distribution(df, args.out)
    chart_bedrooms_box(df, args.out)
    chart_ppb_ranking(df, args.out)
    chart_rank_scatter(df, args.out)
    chart_avg_median_gap(df, args.out)
    chart_tier_donut(df, args.out)
    chart_correlation(df, args.out)
    chart_full_ranking(df, args.out)

    print(f"\n✅  All 9 charts saved to ./{args.out}/")
    print(f"    Open folder: xdg-open {args.out}/ 2>/dev/null || open {args.out}/\n")


if __name__ == "__main__":
    main()
