import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_INPUT = Path("outputs/week1/flipkart_cleaned.csv")
OUTPUT_DIR    = Path("outputs/week2_charm")
FIGURES_DIR   = OUTPUT_DIR / "figures"

# ── Visual theme ───────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
ACCENT  = "#E44D26"
BLUE    = "#4A90D9"
GREEN   = "#5BAD6F"
PURPLE  = "#9B59B6"

CHARM_ENDINGS = [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]

# ── Utilities ──────────────────────────────────────────────────────────────────

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name: str):
    path = FIGURES_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [fig] {path.name}")


def section(title: str):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")


def hr():
    print("  " + "─" * 58)


# ── Load ───────────────────────────────────────────────────────────────────────

def load(path: Path) -> pd.DataFrame:
    section("LOAD CLEANED DATA")
    df = pd.read_csv(path, low_memory=False)

    # Always re-derive — these dtypes are lost on CSV round-trip
    df["price_last3"] = df["price"] % 1000
    df["charm_ending"] = df["price_last3"].isin(CHARM_ENDINGS).astype(int)
    seg_labels = ["Budget", "Entry-Mid", "Mid", "Upper-Mid", "Premium"]
    df["price_segment"] = pd.Categorical(
        pd.cut(
            df["price"],
            bins   = [0, 8_000, 15_000, 30_000, 60_000, np.inf],
            labels = seg_labels,
        ),
        categories = seg_labels,
        ordered    = True,
    )

    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns from {path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. OVERALL CHARM RATE
# ══════════════════════════════════════════════════════════════════════════════

def overall_charm(df: pd.DataFrame):
    section("1. OVERALL CHARM PRICING RATE")

    total       = len(df)
    charm_count = df["charm_ending"].sum()
    rate        = charm_count / total * 100

    print(f"  Total listings   : {total:,}")
    print(f"  Charm-priced     : {charm_count:,}  ({rate:.1f}%)")
    print(f"  Non-charm        : {total - charm_count:,}  ({100 - rate:.1f}%)")

    # ── Fig 1: Donut chart
    fig, ax = plt.subplots(figsize=(7, 7))
    sizes  = [charm_count, total - charm_count]
    labels = [f"Charm-priced\n({rate:.1f}%)", f"Non-charm\n({100-rate:.1f}%)"]
    colors = [ACCENT, BLUE]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white"),
        textprops=dict(fontsize=12),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_color("white")
    ax.set_title("Charm vs Non-Charm Priced Listings\n(Flipkart Mobiles 2026)",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    save_fig("C01_charm_donut")

    return rate


# ══════════════════════════════════════════════════════════════════════════════
# 2. DIGIT FREQUENCY — LAST 3 DIGITS
# ══════════════════════════════════════════════════════════════════════════════

def digit_frequency(df: pd.DataFrame):
    section("2. LAST-3-DIGIT FREQUENCY")

    # Count every last-3 value and focus on top endings
    last3_counts = df["price_last3"].value_counts().sort_index()

    # Highlight charm endings vs. others
    charm_mask  = last3_counts.index.isin(CHARM_ENDINGS)
    top_charm   = last3_counts[charm_mask].sort_values(ascending=False).head(10)
    top_all     = last3_counts.sort_values(ascending=False).head(20)

    print(f"\n  Top 10 charm endings:\n{top_charm.to_string()}")
    print(f"\n  Top 20 overall endings:\n{top_all.head(20).to_string()}")

    # ── Fig 2a: Bar chart — top 20 last-3 endings
    fig, ax = plt.subplots(figsize=(15, 6))
    bar_colors = [ACCENT if i in CHARM_ENDINGS else BLUE
                  for i in top_all.index]
    ax.bar(top_all.index.astype(str), top_all.values,
           color=bar_colors, edgecolor="white", width=0.7)
    for bar, val in zip(ax.patches, top_all.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", fontsize=8.5)
    ax.set_title("Top 20 Last-3-Digit Price Endings\n"
                 f"  {ACCENT} = charm ending  |  {BLUE} = other",
                 fontweight="bold")
    ax.set_xlabel("Last 3 digits of price")
    ax.set_ylabel("Number of listings")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_fig("C02a_top20_endings")

    # ── Fig 2b: Heatmap of last-2-digit frequency (psychological sweetspots)
    df["last2"] = df["price"] % 100
    last2_counts = df["last2"].value_counts().sort_index()
    grid_size = 10
    matrix = np.zeros((grid_size, grid_size))
    for val, cnt in last2_counts.items():
        r, c = int(val) // grid_size, int(val) % grid_size
        if 0 <= r < grid_size and 0 <= c < grid_size:
            matrix[r, c] = cnt

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True, fmt=".0f", cmap="YlOrRd",
        xticklabels=range(10), yticklabels=range(0, 100, 10),
        linewidths=0.5, ax=ax, cbar_kws={"label": "Count"},
    )
    ax.set_title("Last-2-Digit Frequency Heatmap\n"
                 "(row = tens digit, col = units digit)",
                 fontweight="bold")
    ax.set_xlabel("Units digit")
    ax.set_ylabel("Tens digit (× 10)")
    plt.tight_layout()
    save_fig("C02b_last2_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CHARM PRICING BY PRICE SEGMENT
# ══════════════════════════════════════════════════════════════════════════════

def charm_by_segment(df: pd.DataFrame):
    section("3. CHARM RATE BY PRICE SEGMENT")

    seg_order = ["Budget", "Entry-Mid", "Mid", "Upper-Mid", "Premium"]
    seg_stats = (
        df.groupby("price_segment", observed=True)["charm_ending"]
          .agg(count="count", charm_sum="sum")
          .assign(charm_rate=lambda x: x["charm_sum"] / x["count"] * 100)
          .reindex(seg_order)
    )
    print(f"\n  Charm rate by segment:\n{seg_stats.to_string()}")

    # ── Fig 3a: Grouped bar — count & charm rate
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    x    = np.arange(len(seg_order))
    w    = 0.4
    bars = ax1.bar(x - w/2, seg_stats["count"], width=w,
                   color=BLUE, label="SKU count", alpha=0.85, edgecolor="white")
    ax1.bar(x + w/2, seg_stats["charm_sum"], width=w,
            color=ACCENT, label="Charm-priced count", alpha=0.85, edgecolor="white")

    ax2.plot(x, seg_stats["charm_rate"], "o--", color=GREEN,
             lw=2, ms=8, label="Charm rate %")
    ax2.set_ylabel("Charm rate (%)", color=GREEN)
    ax2.tick_params(axis="y", colors=GREEN)

    ax1.set_xticks(x)
    ax1.set_xticklabels(seg_order)
    ax1.set_ylabel("Number of listings")
    ax1.set_xlabel("")
    ax1.set_title("SKU Count vs Charm-Priced Count by Segment",
                  fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    save_fig("C03a_charm_by_segment")

    # ── Fig 3b: Stacked bar (%)
    fig, ax = plt.subplots(figsize=(11, 5))
    charm_pct   = seg_stats["charm_rate"]
    nocharm_pct = 100 - charm_pct
    ax.bar(seg_order, charm_pct,   color=ACCENT, label="Charm-priced", edgecolor="white")
    ax.bar(seg_order, nocharm_pct, bottom=charm_pct,
           color=BLUE,  label="Non-charm",    edgecolor="white")
    for i, (cp, nc) in enumerate(zip(charm_pct, nocharm_pct)):
        ax.text(i, cp/2, f"{cp:.1f}%", ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage of listings (%)")
    ax.set_title("Charm vs Non-Charm by Price Segment (100% stacked)",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_fig("C03b_charm_stacked_pct")

    return seg_stats


# ══════════════════════════════════════════════════════════════════════════════
# 4. CHARM PRICING BY BRAND
# ══════════════════════════════════════════════════════════════════════════════

def charm_by_brand(df: pd.DataFrame):
    section("4. CHARM RATE BY BRAND (≥5 SKUs)")

    brand_stats = (
        df.groupby("brand")["charm_ending"]
          .agg(count="count", charm_sum="sum")
          .query("count >= 5")
          .assign(charm_rate=lambda x: x["charm_sum"] / x["count"] * 100)
          .sort_values("charm_rate", ascending=False)
    )
    print(f"\n  Brands with ≥5 SKUs: {len(brand_stats)}")
    print(f"\n  Top 15 most charm-priced brands:\n{brand_stats.head(15).to_string()}")
    print(f"\n  Bottom 10 least charm-priced brands:\n{brand_stats.tail(10).to_string()}")

    # ── Fig 4: Diverging bar from 50% midpoint
    top_bottom = pd.concat([brand_stats.head(12), brand_stats.tail(8)])
    top_bottom = top_bottom.drop_duplicates()
    fig, ax = plt.subplots(figsize=(13, 8))
    colors = [ACCENT if r >= 50 else BLUE for r in top_bottom["charm_rate"]]
    ax.barh(top_bottom.index, top_bottom["charm_rate"] - 50,
            color=colors, edgecolor="white", left=50)
    ax.axvline(50, color="black", lw=1.2, ls="--", alpha=0.6, label="50% baseline")
    ax.set_xlabel("Charm-priced rate (%)")
    ax.set_title("Brand Charm-Pricing Rate\n"
                 f"  {ACCENT} = above 50%  |  {BLUE} = below 50%",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_fig("C04_charm_by_brand")

    return brand_stats


# ══════════════════════════════════════════════════════════════════════════════
# 5. DOES CHARM PRICING CORRELATE WITH RATINGS?
# ══════════════════════════════════════════════════════════════════════════════

def charm_vs_rating(df: pd.DataFrame):
    section("5. CHARM PRICING vs RATINGS")

    if "rating" not in df.columns:
        print("  'rating' column not found — skipping.")
        return

    sub = df.dropna(subset=["rating"]).copy()

    charm     = sub[sub["charm_ending"] == 1]["rating"]
    non_charm = sub[sub["charm_ending"] == 0]["rating"]

    t_stat, p_val = stats.ttest_ind(charm, non_charm, equal_var=False)
    print(f"\n  Charm mean rating     : {charm.mean():.4f}  (n={len(charm):,})")
    print(f"  Non-charm mean rating : {non_charm.mean():.4f}  (n={len(non_charm):,})")
    print(f"  Welch t-test          : t = {t_stat:.4f}  p = {p_val:.4g}")
    if p_val < 0.05:
        print("  → Statistically significant difference (α = 0.05)")
    else:
        print("  → No statistically significant difference (α = 0.05)")

    # ── Fig 5a: KDE comparison
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.kdeplot(charm,     fill=True, color=ACCENT, alpha=0.45,
                label=f"Charm-priced  μ={charm.mean():.3f}", ax=ax)
    sns.kdeplot(non_charm, fill=True, color=BLUE,   alpha=0.45,
                label=f"Non-charm  μ={non_charm.mean():.3f}", ax=ax)
    ax.axvline(charm.mean(),     color=ACCENT, ls="--", lw=1.8)
    ax.axvline(non_charm.mean(), color=BLUE,   ls="--", lw=1.8)
    ax.set_title("Rating Distribution: Charm vs Non-Charm Priced",
                 fontweight="bold")
    ax.set_xlabel("Rating")
    ax.legend()
    plt.tight_layout()
    save_fig("C05a_charm_vs_rating_kde")

    # ── Fig 5b: Box plot by segment × charm
    seg_order = ["Budget", "Entry-Mid", "Mid", "Upper-Mid", "Premium"]
    sub["charm_label"] = sub["charm_ending"].map({1: "Charm", 0: "Non-charm"})
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=sub, x="price_segment", y="rating",
        hue="charm_label", palette={"Charm": ACCENT, "Non-charm": BLUE},
        order=seg_order, ax=ax, width=0.6, linewidth=1.2,
    )
    ax.set_title("Rating by Segment & Charm-Pricing Status", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Rating")
    plt.tight_layout()
    save_fig("C05b_rating_box_segment_charm")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CHARM PRICING vs POPULARITY (ratings_count)
# ══════════════════════════════════════════════════════════════════════════════

def charm_vs_popularity(df: pd.DataFrame):
    section("6. CHARM PRICING vs POPULARITY")

    if "ratings_count" not in df.columns:
        print("  'ratings_count' not found — skipping.")
        return

    sub = df.dropna(subset=["ratings_count"]).copy()
    sub["ratings_count_log"] = np.log1p(sub["ratings_count"])

    charm     = sub[sub["charm_ending"] == 1]["ratings_count"]
    non_charm = sub[sub["charm_ending"] == 0]["ratings_count"]

    t_stat, p_val = stats.mannwhitneyu(charm, non_charm, alternative="two-sided")
    print(f"\n  Charm median ratings_count     : {charm.median():.0f}")
    print(f"  Non-charm median ratings_count : {non_charm.median():.0f}")
    print(f"  Mann-Whitney U test            : U = {t_stat:.1f}  p = {p_val:.4g}")

    # ── Fig 6: Violin — log(ratings_count) by charm
    sub["charm_label"] = sub["charm_ending"].map({1: "Charm", 0: "Non-charm"})
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(
        data=sub, x="charm_label", y="ratings_count_log",
        palette={"Charm": ACCENT, "Non-charm": BLUE},
        inner="quartile", ax=ax,
    )
    ax.set_title("log(Ratings Count): Charm vs Non-Charm Priced",
                 fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("log(1 + Ratings Count)")
    plt.tight_layout()
    save_fig("C06_charm_vs_popularity")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PRICE ROUNDING PATTERNS (₹X,000 / ₹X,999 / ₹X,990 etc.)
# ══════════════════════════════════════════════════════════════════════════════

def rounding_patterns(df: pd.DataFrame):
    section("7. PRICE ROUNDING PATTERNS")

    df = df.copy()

    # Tag key rounding types
    def tag_rounding(p):
        last3 = int(p) % 1000
        last2 = int(p) % 100
        if last3 == 0:
            return "Round (×000)"
        elif last3 == 999:
            return "Just-below-K (999)"
        elif last3 == 990:
            return "Just-below-K (990)"
        elif last2 == 99:
            return "Just-below-hundred (×99)"
        elif last2 == 0:
            return "Round hundred (×00)"
        elif last3 in CHARM_ENDINGS:
            return "Other charm"
        else:
            return "Non-charm"

    df["rounding_type"] = df["price"].apply(tag_rounding)

    rtype_counts = df["rounding_type"].value_counts()
    print(f"\n  Rounding type distribution:\n{rtype_counts.to_string()}")

    # ── Fig 7: Horizontal bar
    fig, ax = plt.subplots(figsize=(12, 6))
    palette_map = {
        "Round (×000)"           : GREEN,
        "Just-below-K (999)"     : ACCENT,
        "Just-below-K (990)"     : "#E88D5E",
        "Just-below-hundred (×99)": "#D94F8B",
        "Round hundred (×00)"    : BLUE,
        "Other charm"            : PURPLE,
        "Non-charm"              : "#AAAAAA",
    }
    bar_colors = [palette_map.get(t, BLUE) for t in rtype_counts.index]
    ax.barh(rtype_counts.index[::-1], rtype_counts.values[::-1],
            color=bar_colors[::-1], edgecolor="white")
    for bar, val in zip(ax.patches, rtype_counts.values[::-1]):
        ax.text(val + 20, bar.get_y() + bar.get_height()/2,
                f"{val:,}  ({val/len(df)*100:.1f}%)",
                va="center", fontsize=9.5)
    ax.set_title("Price Rounding Patterns — Flipkart Mobiles 2026",
                 fontweight="bold")
    ax.set_xlabel("Number of listings")
    plt.tight_layout()
    save_fig("C07_rounding_patterns")


# ══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def summary_table(df: pd.DataFrame, seg_stats: pd.DataFrame,
                  brand_stats: pd.DataFrame, overall_rate: float):
    section("8. CHARM PRICING SUMMARY")

    print(f"\n  ── Global ──")
    print(f"  Charm-priced listings  : {df['charm_ending'].sum():,}  /  {len(df):,}  ({overall_rate:.1f}%)")

    print(f"\n  ── By segment (charm rate %) ──")
    print(seg_stats["charm_rate"].round(1).to_string())

    print(f"\n  ── Top 5 most charm-priced brands ──")
    print(brand_stats.head(5)[["count", "charm_sum", "charm_rate"]].round(1).to_string())

    print(f"\n  ── Top 5 least charm-priced brands ──")
    print(brand_stats.tail(5)[["count", "charm_sum", "charm_rate"]].round(1).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Module 2 — Charm Pricing Analysis")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    args = parser.parse_args()

    ensure_dirs()

    df = load(Path(args.input))

    overall_rate = overall_charm(df)
    digit_frequency(df)
    seg_stats    = charm_by_segment(df)
    brand_stats  = charm_by_brand(df)
    charm_vs_rating(df)
    charm_vs_popularity(df)
    rounding_patterns(df)
    summary_table(df, seg_stats, brand_stats, overall_rate)

    section("COMPLETE ✓")
    print(f"  Figures  → {FIGURES_DIR}/   (7 PNGs)")
    print("\n  Next →  rating_analysis.py  (Module 3)")


if __name__ == "__main__":
    main()