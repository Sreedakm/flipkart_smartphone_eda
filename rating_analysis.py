import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_INPUT = Path("outputs/week1/flipkart_cleaned.csv")
OUTPUT_DIR    = Path("outputs/week3_ratings")
FIGURES_DIR   = OUTPUT_DIR / "figures"

# ── Visual theme ───────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
ACCENT  = "#E44D26"
BLUE    = "#4A90D9"
GREEN   = "#5BAD6F"
PURPLE  = "#9B59B6"
YELLOW  = "#F1C40F"

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

    # Always re-derive price_segment as a proper Categorical —
    # the dtype is lost when pandas writes/reads CSV.
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

    required = ["rating", "ratings_count"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"  WARNING: columns not found — {missing}")

    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns from {path}")
    print(f"  Rows with rating     : {df['rating'].notna().sum():,}")
    if "ratings_count" in df.columns:
        print(f"  Rows with ratings_count : {df['ratings_count'].notna().sum():,}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. RATING DISTRIBUTION DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════

def rating_distribution(df: pd.DataFrame):
    section("1. RATING DISTRIBUTION DEEP-DIVE")

    r = df["rating"].dropna()

    print(f"\n  Count   : {len(r):,}")
    print(f"  Mean    : {r.mean():.4f}")
    print(f"  Median  : {r.median():.4f}")
    print(f"  Std     : {r.std():.4f}")
    print(f"  Skew    : {r.skew():.4f}")
    print(f"  Kurtosis: {r.kurtosis():.4f}")
    print(f"\n  Percentiles:")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    p{pct:02d} = {r.quantile(pct/100):.2f}")

    # Rating band counts
    bands = {
        "< 3.0"     : (r < 3.0).sum(),
        "3.0 – 3.5" : ((r >= 3.0) & (r < 3.5)).sum(),
        "3.5 – 4.0" : ((r >= 3.5) & (r < 4.0)).sum(),
        "4.0 – 4.5" : ((r >= 4.0) & (r < 4.5)).sum(),
        "≥ 4.5"     : (r >= 4.5).sum(),
    }
    print(f"\n  Rating band distribution:")
    for band, cnt in bands.items():
        print(f"    {band} : {cnt:,}  ({cnt/len(r)*100:.1f}%)")

    # ── Fig R01: Histogram with percentile lines
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram
    sns.histplot(r, bins=40, kde=True, color=BLUE, ax=axes[0])
    for pct, col, ls in [(0.25, ACCENT, "--"), (0.50, GREEN, "-"),
                          (0.75, PURPLE, "--")]:
        val = r.quantile(pct)
        axes[0].axvline(val, color=col, lw=2, ls=ls,
                        label=f"p{int(pct*100)} = {val:.2f}")
    axes[0].axvline(r.mean(), color=YELLOW, lw=2, ls=":",
                    label=f"mean = {r.mean():.2f}")
    axes[0].set_title("Rating Distribution with Percentile Lines", fontweight="bold")
    axes[0].set_xlabel("Rating")
    axes[0].legend(fontsize=9)

    # Rating band pie
    band_vals = list(bands.values())
    band_keys = list(bands.keys())
    band_cols = [ACCENT, PURPLE, BLUE, GREEN, YELLOW]
    axes[1].pie(band_vals, labels=band_keys, colors=band_cols,
                autopct="%1.1f%%", startangle=140,
                wedgeprops=dict(edgecolor="white"),
                textprops=dict(fontsize=10))
    axes[1].set_title("Listing Share by Rating Band", fontweight="bold")

    plt.suptitle("Rating Distribution — Flipkart Mobiles 2026",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("R01_rating_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# 2. RATINGS_COUNT DISTRIBUTION & ENGAGEMENT TIERS
# ══════════════════════════════════════════════════════════════════════════════

def engagement_tiers(df: pd.DataFrame) -> pd.DataFrame:
    section("2. REVIEW ENGAGEMENT TIERS")

    if "ratings_count" not in df.columns:
        print("  'ratings_count' not found — skipping.")
        return df

    df = df.copy()
    rc = df["ratings_count"].dropna()

    print(f"\n  Ratings count stats:")
    print(f"  Mean   : {rc.mean():.1f}")
    print(f"  Median : {rc.median():.1f}")
    print(f"  Max    : {rc.max():.0f}")
    print(f"  Min    : {rc.min():.0f}")

    # Engagement tiers
    df["engagement_tier"] = pd.cut(
        df["ratings_count"],
        bins   = [0, 50, 200, 500, 1000, np.inf],
        labels = ["Niche (<50)", "Low (50-200)",
                  "Moderate (200-500)", "Popular (500-1k)", "Viral (>1k)"],
    )

    tier_counts = df["engagement_tier"].value_counts().sort_index()
    print(f"\n  Engagement tier distribution:\n{tier_counts.to_string()}")

    # ── Fig R02a: Log-scale histogram of ratings_count
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].hist(np.log1p(rc), bins=50, color=BLUE, edgecolor="white", alpha=0.85)
    axes[0].axvline(np.log1p(rc.median()), color=ACCENT, lw=2, ls="--",
                    label=f"Median = {rc.median():.0f}")
    axes[0].axvline(np.log1p(rc.mean()), color=GREEN, lw=2, ls=":",
                    label=f"Mean = {rc.mean():.0f}")
    axes[0].set_title("log(Ratings Count) Distribution", fontweight="bold")
    axes[0].set_xlabel("log(1 + Ratings Count)")
    axes[0].legend()

    # Engagement tier bar
    axes[1].bar(tier_counts.index, tier_counts.values,
                color=sns.color_palette("Blues_d", len(tier_counts))[::-1],
                edgecolor="white")
    for bar, val in zip(axes[1].patches, tier_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f"{val:,}", ha="center", fontsize=9)
    axes[1].set_title("Listing Count by Engagement Tier", fontweight="bold")
    axes[1].set_ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    save_fig("R02_engagement_tiers")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. RATING vs PRICE — DETAILED
# ══════════════════════════════════════════════════════════════════════════════

def rating_vs_price(df: pd.DataFrame):
    section("3. RATING vs PRICE — DETAILED")

    rupee_fmt = mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k")
    sub = df.dropna(subset=["rating", "price"]).copy()

    seg_order = ["Budget", "Entry-Mid", "Mid", "Upper-Mid", "Premium"]

    # Mean rating per segment
    seg_rating = (
        sub.groupby("price_segment", observed=True)["rating"]
           .agg(["mean", "median", "std", "count"])
           .reindex(seg_order)
    )
    print(f"\n  Rating stats by price segment:\n{seg_rating.round(3).to_string()}")

    # Kruskal-Wallis test across segments
    groups = [sub[sub["price_segment"] == s]["rating"].dropna()
              for s in seg_order if s in sub["price_segment"].cat.categories]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        h_stat, p_val = kruskal(*groups)
        print(f"\n  Kruskal-Wallis H = {h_stat:.4f}  p = {p_val:.4g}")
        print("  → Significant rating differences across segments"
              if p_val < 0.05 else
              "  → No significant differences across segments")

    # ── Fig R03a: Mean rating per segment (bar + error)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(seg_order))
    means = seg_rating["mean"].values
    stds  = seg_rating["std"].values
    bars  = ax.bar(x, means, color=BLUE, alpha=0.8,
                   edgecolor="white", width=0.6)
    ax.errorbar(x, means, yerr=stds, fmt="none",
                color=ACCENT, capsize=5, lw=2)
    ax.set_xticks(x)
    ax.set_xticklabels(seg_order)
    ax.set_ylim(3.5, 5.0)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.03,
                f"{m:.3f}", ha="center", fontsize=10)
    ax.set_title("Mean Rating by Price Segment  (± 1 SD)", fontweight="bold")
    ax.set_ylabel("Mean Rating")
    ax.set_xlabel("")
    plt.tight_layout()
    save_fig("R03a_rating_by_segment")

    # ── Fig R03b: Violin of rating by segment
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.violinplot(
        data=sub, x="price_segment", y="rating",
        order=seg_order, palette="Blues", inner="quartile", ax=ax,
    )
    ax.set_title("Rating Distribution per Price Segment", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Rating")
    plt.tight_layout()
    save_fig("R03b_rating_violin_by_segment")

    # ── Fig R03c: Hexbin — price vs rating (full dataset)
    fig, ax = plt.subplots(figsize=(11, 6))
    hb = ax.hexbin(sub["price"], sub["rating"],
                   gridsize=40, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Count")
    ax.set_xlabel("Price (₹)")
    ax.set_ylabel("Rating")
    ax.xaxis.set_major_formatter(rupee_fmt)
    ax.set_title("Price vs Rating — Hexbin Density", fontweight="bold")
    plt.tight_layout()
    save_fig("R03c_price_rating_hexbin")


# ══════════════════════════════════════════════════════════════════════════════
# 4. RATING BY BRAND
# ══════════════════════════════════════════════════════════════════════════════

def rating_by_brand(df: pd.DataFrame):
    section("4. RATING BY BRAND (≥5 SKUs)")

    sub = df.dropna(subset=["rating"]).copy()

    brand_rating = (
        sub.groupby("brand")["rating"]
           .agg(count="count", mean="mean", median="median", std="std")
           .query("count >= 5")
           .sort_values("mean", ascending=False)
    )
    print(f"\n  Brands with ≥5 rated SKUs: {len(brand_rating)}")
    print(f"\n  Top 15 by mean rating:\n{brand_rating.head(15).round(3).to_string()}")
    print(f"\n  Bottom 10 by mean rating:\n{brand_rating.tail(10).round(3).to_string()}")

    # ── Fig R04: Horizontal bar — top + bottom brands by mean rating
    top_b   = brand_rating.head(12)
    bot_b   = brand_rating.tail(8)
    combined = pd.concat([top_b, bot_b]).drop_duplicates()
    threshold = brand_rating["mean"].median()

    fig, ax = plt.subplots(figsize=(13, 9))
    colors = [GREEN if r >= threshold else ACCENT
              for r in combined["mean"]]
    ax.barh(combined.index[::-1], combined["mean"][::-1],
            color=colors[::-1], edgecolor="white", xerr=combined["std"][::-1],
            error_kw=dict(lw=1.2, capsize=3, color="gray"))
    ax.axvline(threshold, color="black", ls="--", lw=1.2, alpha=0.6,
               label=f"Median brand rating = {threshold:.2f}")
    ax.set_xlabel("Mean Rating")
    ax.set_title("Brand Mean Rating (Top 12 + Bottom 8)\n"
                 f"  {GREEN} = above median  |  {ACCENT} = below median",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_fig("R04_rating_by_brand")

    return brand_rating


# ══════════════════════════════════════════════════════════════════════════════
# 5. RATING vs SPECS (RAM, BATTERY, STORAGE)
# ══════════════════════════════════════════════════════════════════════════════

def rating_vs_specs(df: pd.DataFrame):
    section("5. RATING vs KEY SPECS")

    sub = df.dropna(subset=["rating"]).copy()

    spec_meta = [
        ("ram",          "RAM (GB)"),
        ("rom",          "Storage (GB)"),
        ("battery",      "Battery (mAh)"),
        ("display_inch", "Display Size (″)"),
    ]
    spec_cols = [(c, lbl) for c, lbl in spec_meta if c in sub.columns]

    if not spec_cols:
        print("  No spec columns found — skipping.")
        return

    # Correlation table
    print(f"\n  Pearson r (spec vs rating):")
    for col, lbl in spec_cols:
        valid = sub.dropna(subset=[col])
        r, p  = stats.pearsonr(valid[col], valid["rating"])
        print(f"    {lbl:20s} : r = {r:+.4f}  p = {p:.3g}")

    # ── Fig R05: Scatter grid — rating vs each spec
    n_cols = min(2, len(spec_cols))
    n_rows = (len(spec_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    palette = [BLUE, GREEN, ACCENT, PURPLE]
    for ax, (col, lbl), color in zip(axes, spec_cols, palette):
        valid = sub.dropna(subset=[col])
        sample = valid.sample(min(3_000, len(valid)), random_state=42)
        ax.scatter(sample[col], sample["rating"],
                   alpha=0.2, s=10, color=color)
        # OLS line
        slope, intercept, r, p, _ = stats.linregress(valid[col], valid["rating"])
        xs = np.linspace(valid[col].min(), valid[col].max(), 200)
        ax.plot(xs, intercept + slope * xs, color="black", lw=2,
                label=f"r = {r:.3f}")
        ax.set_xlabel(lbl)
        ax.set_ylabel("Rating")
        ax.set_title(f"Rating vs {lbl}", fontweight="bold")
        ax.legend(fontsize=9)

    # Hide unused axes
    for ax in axes[len(spec_cols):]:
        ax.set_visible(False)

    plt.suptitle("Rating vs Key Specifications", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("R05_rating_vs_specs")

    # ── Fig R06: Mean rating by RAM tier (discrete spec)
    if "ram" in sub.columns:
        top_rams = sub["ram"].value_counts().nlargest(8).index.sort_values()
        ram_sub  = sub[sub["ram"].isin(top_rams)]
        ram_rating = (
            ram_sub.groupby("ram")["rating"]
                   .agg(mean="mean", std="std", count="count")
                   .sort_index()
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(ram_rating))
        ax.bar(x, ram_rating["mean"], color=BLUE, alpha=0.85,
               edgecolor="white", width=0.6)
        ax.errorbar(x, ram_rating["mean"], yerr=ram_rating["std"],
                    fmt="none", color=ACCENT, capsize=5, lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(r)} GB" for r in ram_rating.index])
        ax.set_ylim(3.5, 5.0)
        for bar, m in zip(ax.patches, ram_rating["mean"]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.03,
                    f"{m:.3f}", ha="center", fontsize=9)
        ax.set_title("Mean Rating by RAM Tier  (± 1 SD)", fontweight="bold")
        ax.set_ylabel("Mean Rating")
        ax.set_xlabel("RAM")
        plt.tight_layout()
        save_fig("R06_rating_by_ram")


# ══════════════════════════════════════════════════════════════════════════════
# 6. REVIEW-TO-RATING RATIO (ENGAGEMENT QUALITY)
# ══════════════════════════════════════════════════════════════════════════════

def review_ratio(df: pd.DataFrame):
    section("6. REVIEW-TO-RATING RATIO")

    if "reviews_count" not in df.columns or "ratings_count" not in df.columns:
        print("  Required columns not found — skipping.")
        return

    sub = df.dropna(subset=["reviews_count", "ratings_count"]).copy()
    # Avoid division by zero
    sub = sub[sub["ratings_count"] > 0]
    sub["review_ratio"] = sub["reviews_count"] / sub["ratings_count"]

    print(f"\n  Review/Rating ratio stats:")
    print(f"  Mean   : {sub['review_ratio'].mean():.4f}")
    print(f"  Median : {sub['review_ratio'].median():.4f}")
    print(f"  Std    : {sub['review_ratio'].std():.4f}")

    # Correlation with rating
    valid = sub.dropna(subset=["rating"])
    r, p  = stats.pearsonr(np.log1p(valid["review_ratio"]),
                           valid["rating"])
    print(f"\n  Correlation log(review_ratio) vs rating: r = {r:.4f}  p = {p:.4g}")

    # ── Fig R07: Scatter — review ratio vs rating
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.histplot(np.log1p(sub["review_ratio"]), bins=50,
                 kde=True, color=PURPLE, ax=axes[0])
    axes[0].set_title("Distribution of log(Review/Rating Ratio)",
                      fontweight="bold")
    axes[0].set_xlabel("log(1 + Review/Rating Ratio)")

    sample = valid.sample(min(4_000, len(valid)), random_state=42)
    axes[1].scatter(np.log1p(sample["review_ratio"]), sample["rating"],
                    alpha=0.2, s=10, color=PURPLE)
    slope, intercept, _, _, _ = stats.linregress(
        np.log1p(valid["review_ratio"]), valid["rating"])
    xs = np.linspace(np.log1p(sub["review_ratio"]).min(),
                     np.log1p(sub["review_ratio"]).max(), 200)
    axes[1].plot(xs, intercept + slope * xs, color=ACCENT, lw=2,
                 label=f"r = {r:.3f}")
    axes[1].set_xlabel("log(1 + Review/Rating Ratio)")
    axes[1].set_ylabel("Rating")
    axes[1].set_title("Review Ratio vs Rating", fontweight="bold")
    axes[1].legend()

    plt.suptitle("Review-to-Rating Engagement Analysis",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("R07_review_ratio")


# ══════════════════════════════════════════════════════════════════════════════
# 7. TOP-RATED & BOTTOM-RATED PHONES
# ══════════════════════════════════════════════════════════════════════════════

def top_bottom_phones(df: pd.DataFrame):
    section("7. TOP & BOTTOM RATED PHONES")

    rupee_fmt = lambda x: f"₹{x:,.0f}"

    # Filter for enough social proof
    MIN_RATINGS = 30
    sub = df[df["ratings_count"] >= MIN_RATINGS].dropna(subset=["rating"])

    top10 = sub.nlargest(10, "rating")[
        ["model", "brand", "price", "rating", "ratings_count"]
    ].copy()
    bot10 = sub.nsmallest(10, "rating")[
        ["model", "brand", "price", "rating", "ratings_count"]
    ].copy()

    print(f"\n  Filtered to phones with ≥{MIN_RATINGS} ratings: {len(sub):,}")
    print(f"\n  Top 10 rated phones:\n{top10.to_string(index=False)}")
    print(f"\n  Bottom 10 rated phones:\n{bot10.to_string(index=False)}")

    # ── Fig R08: Lollipop chart — top & bottom 10
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, data, color, title in [
        (axes[0], top10,   GREEN, f"Top 10 Rated  (≥{MIN_RATINGS} ratings)"),
        (axes[1], bot10,   ACCENT, f"Bottom 10 Rated  (≥{MIN_RATINGS} ratings)"),
    ]:
        labels = [
            f"{row['brand']} — {row['model'][:30]}\n₹{row['price']:,.0f}"
            for _, row in data.iterrows()
        ]
        y = np.arange(len(data))
        ax.hlines(y, 0, data["rating"].values, color="gray", lw=1.5, alpha=0.5)
        ax.scatter(data["rating"].values, y, color=color, s=80, zorder=5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(data["rating"].min() - 0.2, data["rating"].max() + 0.2)
        ax.set_xlabel("Rating")
        ax.set_title(title, fontweight="bold")
        ax.invert_yaxis()

    plt.suptitle("Top & Bottom Rated Phones on Flipkart (2026)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("R08_top_bottom_rated")


# ══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def summary(df: pd.DataFrame, brand_rating: pd.DataFrame):
    section("8. RATING ANALYSIS SUMMARY")

    r = df["rating"].dropna()
    print(f"\n  Global mean rating    : {r.mean():.3f}")
    print(f"  Global median rating  : {r.median():.3f}")
    print(f"  % rated ≥ 4.0         : {(r >= 4.0).mean()*100:.1f}%")
    print(f"  % rated < 3.5         : {(r < 3.5).mean()*100:.1f}%")
    print(f"\n  Best rated brand (≥5 SKUs) : {brand_rating['mean'].idxmax()}  "
          f"({brand_rating['mean'].max():.3f})")
    print(f"  Worst rated brand (≥5 SKUs): {brand_rating['mean'].idxmin()}  "
          f"({brand_rating['mean'].min():.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Module 3 — Rating & Review Analysis")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    args = parser.parse_args()

    ensure_dirs()

    df           = load(Path(args.input))
    rating_distribution(df)
    df           = engagement_tiers(df)
    rating_vs_price(df)
    brand_rating = rating_by_brand(df)
    rating_vs_specs(df)
    review_ratio(df)
    top_bottom_phones(df)
    summary(df, brand_rating)

    section("COMPLETE ✓")
    print(f"  Figures  → {FIGURES_DIR}/   (8 PNGs)")
    print("\n  Possible next steps:")
    print("    • brand_deep_dive.py   — per-brand spec positioning")
    print("    • spec_pricing.py      — RAM/battery/storage vs price regression")
    print("    • ml_features.py       — feature engineering for price prediction")


if __name__ == "__main__":
    main()