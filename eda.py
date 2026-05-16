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
FLIPKART_PATH = "data/flipkart_mobiles_2026.csv"
OUTPUT_DIR    = Path("outputs/week1")
CLEANED_CSV   = OUTPUT_DIR / "flipkart_cleaned.csv"
FIGURES_DIR   = OUTPUT_DIR / "figures"

# ── Visual theme ───────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
ACCENT  = "#E44D26"   # warm orange-red for highlights
BLUE    = "#4A90D9"
GREEN   = "#5BAD6F"

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


# ── Parsers ────────────────────────────────────────────────────────────────────

def _parse_price(s: pd.Series) -> pd.Series:
    """Strip ₹, commas, whitespace → float."""
    return (
        s.astype(str)
         .str.replace(r"[₹,\s]", "", regex=True)
         .str.replace(r"[^\d.]", "", regex=True)
         .pipe(pd.to_numeric, errors="coerce")
    )


def _parse_numeric(s: pd.Series) -> pd.Series:
    """Extract leading number from messy strings like '128GB', '6.5\"'."""
    return (
        s.astype(str)
         .str.extract(r"([\d.]+)", expand=False)
         .pipe(pd.to_numeric, errors="coerce")
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load(path: str) -> pd.DataFrame:
    section("1. LOAD")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Rows     : {len(df):,}")
    print(f"  Columns  : {list(df.columns)}")
    print(f"\n  Dtypes:\n{df.dtypes.to_string()}")
    print(f"\n  Head:\n{df.head(3).to_string()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN
# ══════════════════════════════════════════════════════════════════════════════

def clean(df: pd.DataFrame) -> pd.DataFrame:
    section("2. CLEAN")
    df = df.copy()

    # ── 2a. Normalise column names
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[\s\-/]+", "_", regex=True)
    )
    print(f"  Columns (normalised): {list(df.columns)}")

    # ── 2b. Numeric coercion
    coerce_map = {
        "price"         : _parse_price,
        "rating"        : _parse_numeric,
        "ratings_count" : _parse_numeric,
        "reviews_count" : _parse_numeric,
        "ram"           : _parse_numeric,
        "rom"           : _parse_numeric,
        "battery"       : _parse_numeric,
        "display_inch"  : _parse_numeric,
        "rear_camera"   : _parse_numeric,
        "front_camera"  : _parse_numeric,
    }
    for col, fn in coerce_map.items():
        if col in df.columns:
            df[col] = fn(df[col])

    # ── 2c. Brand: clean string
    if "brand" in df.columns:
        df["brand"] = df["brand"].astype(str).str.strip().str.title()

    # ── 2d. Null audit
    hr()
    null_pct = (df.isnull().sum() / len(df) * 100).round(2)
    print(f"  Null % per column:\n{null_pct.to_string()}")

    before = len(df)
    df = df.dropna(subset=["price"])
    print(f"\n  Dropped {before - len(df):,} rows with missing price.")

    # ── 2e. Duplicates
    dupes = df.duplicated(subset=["model", "brand", "price"], keep="first").sum()
    df    = df.drop_duplicates(subset=["model", "brand", "price"], keep="first")
    print(f"  Dropped {dupes:,} duplicate (model, brand, price) rows.")

    # ── 2f. Plausibility filters
    hr()
    price_ok  = df["price"].between(1_000, 3_00_000)
    rating_ok = df["rating"].between(0, 5) if "rating" in df.columns else True
    df = df[price_ok & rating_ok]
    print(f"  After plausibility filters : {len(df):,} rows")

    # ── 2g. IQR outlier flag (keep rows, just tag them)
    Q1, Q3 = df["price"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df["price_outlier"] = ~df["price"].between(Q1 - 3*IQR, Q3 + 3*IQR)
    print(f"  Price outliers flagged     : {df['price_outlier'].sum():,}")

    # ── 2h. Derived columns (used by all downstream modules)
    df["price_log"]     = np.log1p(df["price"])
    df["price_segment"] = pd.cut(
        df["price"],
        bins  = [0, 8_000, 15_000, 30_000, 60_000, np.inf],
        labels= ["Budget", "Entry-Mid", "Mid", "Upper-Mid", "Premium"],
    )
    df["price_last3"]   = df["price"] % 1000
    df["charm_ending"]  = df["price_last3"].isin(
        [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
    ).astype(int)

    hr()
    print(f"  Final clean shape: {df.shape}")
    print(f"\n  Segment distribution:\n{df['price_segment'].value_counts().sort_index().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. UNIVARIATE EDA
# ══════════════════════════════════════════════════════════════════════════════

def univariate_eda(df: pd.DataFrame):
    section("3. UNIVARIATE EDA")
    rupee_fmt = mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k")

    # ── 3a. Price distribution (raw + log)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(df["price"], bins=80, kde=True, color=ACCENT, ax=axes[0])
    axes[0].set_title("Price Distribution (₹)", fontweight="bold")
    axes[0].xaxis.set_major_formatter(rupee_fmt)
    axes[0].set_xlabel("Price")

    sns.histplot(df["price_log"], bins=60, kde=True, color=BLUE, ax=axes[1])
    axes[1].set_title("Log-Price Distribution", fontweight="bold")
    axes[1].set_xlabel("log(1 + Price)")

    plt.suptitle("Flipkart Mobiles — Price Distribution", fontsize=14, y=1.01)
    plt.tight_layout()
    save_fig("01_price_distribution")

    # ── 3b. Price segment bar chart
    seg = df["price_segment"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = sns.color_palette("Blues_d", len(seg))[::-1]
    bars = ax.bar(seg.index, seg.values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, seg.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val:,}", ha="center", fontsize=10)
    ax.set_title("Number of Phones per Price Segment", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.tight_layout()
    save_fig("02_price_segments")

    # ── 3c. Rating distribution
    if "rating" in df.columns:
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.histplot(df["rating"].dropna(), bins=40, kde=True,
                     color=GREEN, ax=ax)
        mean_r = df["rating"].mean()
        ax.axvline(mean_r, color=ACCENT, ls="--", lw=2,
                   label=f"Mean = {mean_r:.2f}")
        ax.axvline(df["rating"].median(), color=BLUE, ls=":", lw=2,
                   label=f"Median = {df['rating'].median():.2f}")
        ax.set_title("Rating Distribution", fontweight="bold")
        ax.set_xlabel("Rating (out of 5)")
        ax.legend()
        plt.tight_layout()
        save_fig("03_rating_distribution")

    # ── 3d. Key spec distributions
    spec_meta = {
        "ram"         : ("RAM (GB)",       BLUE),
        "rom"         : ("Storage (GB)",   GREEN),
        "battery"     : ("Battery (mAh)",  ACCENT),
        "display_inch": ('Display Size (")', "#9B59B6"),
    }
    spec_cols = [c for c in spec_meta if c in df.columns]
    if spec_cols:
        fig, axes = plt.subplots(1, len(spec_cols),
                                 figsize=(4.5*len(spec_cols), 5))
        if len(spec_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, spec_cols):
            label, color = spec_meta[col]
            sns.histplot(df[col].dropna(), bins=25, kde=True,
                         color=color, ax=ax)
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("")
        plt.suptitle("Key Spec Distributions", fontsize=13, y=1.02, fontweight="bold")
        plt.tight_layout()
        save_fig("04_spec_distributions")

    # ── Print summary stats
    summary_cols = [c for c in
        ["price","rating","ratings_count","reviews_count",
         "ram","rom","battery","display_inch"] if c in df.columns]
    print(f"\n  Descriptive stats:\n{df[summary_cols].describe().round(2).to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BIVARIATE EDA
# ══════════════════════════════════════════════════════════════════════════════

def bivariate_eda(df: pd.DataFrame):
    section("4. BIVARIATE EDA")
    rupee_fmt = mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k")

    # ── 4a. Price vs Rating scatter + OLS trend
    if "rating" in df.columns:
        sample = df.sample(min(5_000, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(11, 6))
        sc = ax.scatter(
            sample["price"], sample["rating"],
            alpha=0.25, s=12,
            c=sample["price_log"], cmap="YlOrRd", edgecolors="none"
        )
        plt.colorbar(sc, ax=ax, label="log(price)")

        valid = sample.dropna(subset=["price", "rating"])
        slope, intercept, r, p, _ = stats.linregress(valid["price"], valid["rating"])
        xs = np.linspace(valid["price"].min(), valid["price"].max(), 300)
        ax.plot(xs, intercept + slope*xs, color=ACCENT, lw=2.5,
                label=f"OLS  r = {r:.3f}   p = {p:.2e}")

        ax.set_title("Price vs Rating", fontweight="bold")
        ax.set_xlabel("Price (₹)")
        ax.set_ylabel("Rating")
        ax.xaxis.set_major_formatter(rupee_fmt)
        ax.legend()
        plt.tight_layout()
        save_fig("05_price_vs_rating")
        print(f"\n  Price–Rating correlation : r = {r:.4f}  (p = {p:.3g})")

    # ── 4b. Price by RAM (box)
    if "ram" in df.columns:
        top_rams = df["ram"].value_counts().nlargest(8).index.sort_values()
        sub = df[df["ram"].isin(top_rams)]
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.boxplot(data=sub, x="ram", y="price",
                    order=sorted(top_rams), palette="Blues", ax=ax)
        ax.set_title("Price Distribution by RAM", fontweight="bold")
        ax.set_xlabel("RAM (GB)")
        ax.set_ylabel("Price (₹)")
        ax.yaxis.set_major_formatter(rupee_fmt)
        plt.tight_layout()
        save_fig("06_price_by_ram")

    # ── 4c. Price by segment (violin)
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.violinplot(data=df, x="price_segment", y="price",
                   palette="muted", inner="quartile",
                   order=["Budget","Entry-Mid","Mid","Upper-Mid","Premium"],
                   ax=ax)
    ax.set_title("Price Spread within Each Segment (Violin)", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Price (₹)")
    ax.yaxis.set_major_formatter(rupee_fmt)
    plt.tight_layout()
    save_fig("07_price_violin_by_segment")

    # ── 4d. Correlation heatmap
    num_cols = [c for c in
        ["price","rating","ratings_count","reviews_count",
         "ram","rom","battery","display_inch",
         "rear_camera","front_camera"] if c in df.columns]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Matrix — Flipkart Mobiles",
                 fontsize=14, pad=12, fontweight="bold")
    plt.tight_layout()
    save_fig("08_correlation_heatmap")

    price_corr = (corr["price"].drop("price")
                               .abs()
                               .sort_values(ascending=False))
    print(f"\n  Correlations with Price:\n{price_corr.to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. BRAND-LEVEL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    section("5. BRAND-LEVEL SUMMARY")
    rupee_fmt = mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k")

    agg_dict = dict(
        count        = ("price", "size"),
        median_price = ("price", "median"),
        mean_price   = ("price", "mean"),
        std_price    = ("price", "std"),
        charm_rate   = ("charm_ending", "mean"),
    )
    if "rating" in df.columns:
        agg_dict["mean_rating"] = ("rating", "mean")

    summary = (
        df.groupby("brand")
          .agg(**agg_dict)
          .query("count >= 5")
          .sort_values("median_price", ascending=False)
    )
    summary["charm_rate_pct"] = (summary["charm_rate"] * 100).round(1)

    print(f"\n  Brands with ≥5 SKUs: {len(summary)}")
    print(f"\n  Top 20 by median price:\n{summary.head(20).to_string()}")

    # ── 5a. Top 15 brands — horizontal bar
    top15 = summary.head(15).copy()
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = [ACCENT if r >= 50 else BLUE for r in top15["charm_rate_pct"]]
    ax.barh(top15.index[::-1], top15["median_price"][::-1] / 1000,
            color=colors[::-1], edgecolor="white")
    for bar, val in zip(ax.patches, top15["median_price"][::-1] / 1000):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"₹{val:.1f}k", va="center", fontsize=9)
    ax.set_title("Median Price by Brand  (Top 15)\n"
                 f"  {ACCENT} = >50% charm-priced  |  {BLUE} = <50%",
                 fontweight="bold")
    ax.set_xlabel("Median Price (₹ '000)")
    plt.tight_layout()
    save_fig("09_brand_median_price")

    # ── 5b. Brand map: median price vs mean rating
    if "mean_rating" in summary.columns:
        fig, ax = plt.subplots(figsize=(13, 8))
        sc_data = summary.reset_index()
        ax.scatter(
            sc_data["median_price"] / 1000,
            sc_data["mean_rating"],
            s = sc_data["count"].clip(upper=1500),
            alpha=0.55, color=ACCENT,
            edgecolors="white", linewidths=0.6
        )
        for _, row in sc_data.iterrows():
            ax.annotate(row["brand"],
                        (row["median_price"]/1000, row["mean_rating"]),
                        fontsize=7.5, alpha=0.85,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Median Price (₹ '000)")
        ax.set_ylabel("Mean Rating")
        ax.set_title("Brand Map: Median Price vs Mean Rating\n"
                     "(bubble size ∝ SKU count)", fontweight="bold")
        plt.tight_layout()
        save_fig("10_brand_map")

    # ── 5c. Top 10 brands by volume
    top_vol = summary.sort_values("count", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(top_vol.index, top_vol["count"], color=BLUE, edgecolor="white")
    for bar, val in zip(ax.patches, top_vol["count"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(int(val)), ha="center", fontsize=9)
    ax.set_title("Top 10 Brands by SKU Count", fontweight="bold")
    ax.set_ylabel("Number of Listings")
    ax.set_xlabel("")
    plt.tight_layout()
    save_fig("11_brand_volume")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_cleaned(df: pd.DataFrame):
    section("6. SAVE CLEANED DATA")
    df.to_csv(CLEANED_CSV, index=False)
    print(f"  Saved → {CLEANED_CSV}  ({len(df):,} rows × {df.shape[1]} columns)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Week 1 EDA — Flipkart Mobiles")
    parser.add_argument("--flipkart", default=FLIPKART_PATH)
    args = parser.parse_args()

    ensure_dirs()

    raw      = load(args.flipkart)
    clean_df = clean(raw)
    univariate_eda(clean_df)
    bivariate_eda(clean_df)
    brand_summary(clean_df)
    save_cleaned(clean_df)

    section("COMPLETE ✓")
    print(f"  Figures  → {FIGURES_DIR}/   (11 PNGs)")
    print(f"  Cleaned  → {CLEANED_CSV}")
    print("\n  Next →  charm_pricing.py  (Module 2)")


if __name__ == "__main__":
    main()