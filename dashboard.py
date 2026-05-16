import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flipkart Mobiles 2026",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Visual theme ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
ACCENT = "#E44D26"
BLUE   = "#4A90D9"
GREEN  = "#5BAD6F"
PURPLE = "#9B59B6"

CHARM_ENDINGS = [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
SEG_ORDER     = ["Budget", "Entry-Mid", "Mid", "Upper-Mid", "Premium"]
SEG_BINS      = [0, 8_000, 15_000, 30_000, 60_000, np.inf]

CLEANED_CSV = Path("outputs/week1/flipkart_cleaned.csv")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CLEANED_CSV, low_memory=False)

    # Restore derived columns lost on CSV round-trip
    df["price_last3"]  = df["price"] % 1000
    df["charm_ending"] = df["price_last3"].isin(CHARM_ENDINGS).astype(int)
    df["price_segment"] = pd.Categorical(
        pd.cut(df["price"], bins=SEG_BINS, labels=SEG_ORDER),
        categories=SEG_ORDER, ordered=True,
    )
    if "price_log" not in df.columns:
        df["price_log"] = np.log1p(df["price"])
    if "ratings_count" in df.columns:
        df["engagement_tier"] = pd.Categorical(
            pd.cut(
                df["ratings_count"],
                bins=[0, 50, 200, 500, 1000, np.inf],
                labels=["Niche (<50)", "Low (50-200)",
                        "Moderate (200-500)", "Popular (500-1k)", "Viral (>1k)"],
            )
        )
    return df


def apply_filters(df: pd.DataFrame,
                  brands: list, segments: list,
                  price_range: tuple) -> pd.DataFrame:
    mask = (
        df["brand"].isin(brands) &
        df["price_segment"].isin(segments) &
        df["price"].between(*price_range)
    )
    return df[mask].copy()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def sidebar(df: pd.DataFrame):
    st.sidebar.title("📱 Filters")

    all_brands = sorted(df["brand"].dropna().unique())
    brands = st.sidebar.multiselect(
        "Brand", all_brands, default=all_brands,
        help="Select one or more brands"
    )

    segments = st.sidebar.multiselect(
        "Price Segment", SEG_ORDER, default=SEG_ORDER
    )

    price_min = int(df["price"].min())
    price_max = int(df["price"].max())
    price_range = st.sidebar.slider(
        "Price Range (Rs)", price_min, price_max,
        (price_min, price_max), step=500,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Dataset: {len(df):,} listings  |  "
        f"{df['brand'].nunique()} brands"
    )

    if not brands:
        st.sidebar.warning("Select at least one brand.")
    if not segments:
        st.sidebar.warning("Select at least one segment.")

    return brands or all_brands, segments or SEG_ORDER, price_range


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def tab_overview(df: pd.DataFrame):
    st.header("Market Overview")

    rupee_fmt = mticker.FuncFormatter(lambda x, _: f"Rs{x/1000:.0f}k")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Listings",   f"{len(df):,}")
    c2.metric("Brands",           f"{df['brand'].nunique()}")
    c3.metric("Median Price",     f"Rs{df['price'].median():,.0f}")
    c4.metric("Mean Rating",
              f"{df['rating'].mean():.2f}" if "rating" in df.columns else "-")
    c5.metric("Charm-Priced",
              f"{df['charm_ending'].mean()*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df["price"], bins=60, kde=True, color=ACCENT, ax=ax)
        ax.xaxis.set_major_formatter(rupee_fmt)
        ax.set_xlabel("Price")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Listings by Price Segment")
        seg = df["price_segment"].value_counts().reindex(SEG_ORDER)
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = sns.color_palette("Blues_d", len(seg))[::-1]
        bars = ax.bar(seg.index, seg.values, color=colors, edgecolor="white")
        for bar, val in zip(bars, seg.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 3,
                    f"{val:,}", ha="center", fontsize=9)
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Top 10 Brands by SKU Count")
        top10 = df["brand"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(top10.index[::-1], top10.values[::-1],
                color=BLUE, edgecolor="white")
        for bar, val in zip(ax.patches, top10.values[::-1]):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=9)
        ax.set_xlabel("Number of Listings")
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Correlation Heatmap")
        num_cols = [c for c in
            ["price", "rating", "ratings_count",
             "ram", "rom", "battery", "display_inch"]
            if c in df.columns]
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHARM PRICING
# ══════════════════════════════════════════════════════════════════════════════

def tab_charm(df: pd.DataFrame):
    st.header("Charm Pricing Analysis")

    total        = len(df)
    charm_count  = df["charm_ending"].sum()
    rate         = charm_count / total * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Charm-Priced Listings", f"{charm_count:,}")
    c2.metric("Non-Charm Listings",    f"{total - charm_count:,}")
    c3.metric("Overall Charm Rate",    f"{rate:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Charm vs Non-Charm")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            [charm_count, total - charm_count],
            labels=[f"Charm ({rate:.1f}%)", f"Non-charm ({100-rate:.1f}%)"],
            colors=[ACCENT, BLUE],
            autopct="%1.0f%%", startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white"),
        )
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Top 15 Last-3-Digit Endings")
        last3 = df["price_last3"].value_counts().head(15).sort_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [ACCENT if i in CHARM_ENDINGS else BLUE
                  for i in last3.index]
        ax.bar(last3.index.astype(str), last3.values,
               color=colors, edgecolor="white")
        ax.set_xlabel("Last 3 digits of price")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Charm Rate by Segment")
        seg_stats = (
            df.groupby("price_segment", observed=True)["charm_ending"]
              .agg(count="count", charm_sum="sum")
              .assign(charm_rate=lambda x: x["charm_sum"] / x["count"] * 100)
              .reindex(SEG_ORDER)
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [ACCENT if r >= 50 else BLUE
                  for r in seg_stats["charm_rate"]]
        ax.bar(seg_stats.index, seg_stats["charm_rate"],
               color=colors, edgecolor="white")
        ax.axhline(50, color="black", ls="--", lw=1.2, alpha=0.5,
                   label="50% line")
        ax.set_ylabel("Charm Rate (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Charm Rate by Brand (Top 15, 5+ SKUs)")
        brand_charm = (
            df.groupby("brand")["charm_ending"]
              .agg(count="count", charm_sum="sum")
              .query("count >= 5")
              .assign(charm_rate=lambda x: x["charm_sum"] / x["count"] * 100)
              .sort_values("charm_rate", ascending=False)
              .head(15)
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = [ACCENT if r >= 50 else BLUE
                  for r in brand_charm["charm_rate"]]
        ax.barh(brand_charm.index[::-1],
                brand_charm["charm_rate"][::-1],
                color=colors[::-1], edgecolor="white")
        ax.axvline(50, color="black", ls="--", lw=1.2, alpha=0.5)
        ax.set_xlabel("Charm Rate (%)")
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RATINGS
# ══════════════════════════════════════════════════════════════════════════════

def tab_ratings(df: pd.DataFrame):
    st.header("Rating & Review Analysis")

    if "rating" not in df.columns:
        st.warning("Rating column not found.")
        return

    r = df["rating"].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Rating",   f"{r.mean():.3f}")
    c2.metric("Median Rating", f"{r.median():.3f}")
    c3.metric("Rated >= 4.0",  f"{(r >= 4.0).mean()*100:.1f}%")
    c4.metric("Rated < 3.5",   f"{(r < 3.5).mean()*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(r, bins=35, kde=True, color=BLUE, ax=ax)
        ax.axvline(r.mean(),   color=ACCENT, ls="--", lw=2,
                   label=f"Mean {r.mean():.2f}")
        ax.axvline(r.median(), color=GREEN,  ls=":",  lw=2,
                   label=f"Median {r.median():.2f}")
        ax.set_xlabel("Rating")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Rating by Price Segment")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.violinplot(
            data=df, x="price_segment", y="rating",
            order=SEG_ORDER, palette="Blues",
            inner="quartile", ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Rating")
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Top 10 Rated Brands (5+ SKUs)")
        brand_r = (
            df.dropna(subset=["rating"])
              .groupby("brand")["rating"]
              .agg(count="count", mean="mean")
              .query("count >= 5")
              .sort_values("mean", ascending=False)
              .head(10)
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(brand_r.index[::-1], brand_r["mean"][::-1],
                color=GREEN, edgecolor="white")
        ax.set_xlabel("Mean Rating")
        ax.set_xlim(3.5, 5.0)
        for bar, val in zip(ax.patches, brand_r["mean"][::-1]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Price vs Rating")
        sample = df.dropna(subset=["rating"]).sample(
            min(500, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(7, 4))
        rupee_fmt = mticker.FuncFormatter(lambda x, _: f"Rs{x/1000:.0f}k")
        ax.scatter(sample["price"], sample["rating"],
                   alpha=0.3, s=12, color=BLUE)
        valid = df.dropna(subset=["rating"])
        slope, intercept, r_val, _, _ = stats.linregress(
            valid["price"], valid["rating"])
        xs = np.linspace(df["price"].min(), df["price"].max(), 200)
        ax.plot(xs, intercept + slope * xs, color=ACCENT, lw=2,
                label=f"r = {r_val:.3f}")
        ax.xaxis.set_major_formatter(rupee_fmt)
        ax.set_xlabel("Price")
        ax.set_ylabel("Rating")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    st.subheader("Top & Bottom Rated Phones (30+ ratings)")
    if "ratings_count" in df.columns:
        sub = df[df["ratings_count"] >= 30].dropna(subset=["rating"])
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Top 10**")
            top10 = sub.nlargest(10, "rating")[
                ["brand", "model", "price", "rating", "ratings_count"]
            ].reset_index(drop=True)
            top10.index += 1
            st.dataframe(top10, use_container_width=True)
        with col_b:
            st.markdown("**Bottom 10**")
            bot10 = sub.nsmallest(10, "rating")[
                ["brand", "model", "price", "rating", "ratings_count"]
            ].reset_index(drop=True)
            bot10.index += 1
            st.dataframe(bot10, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SPEC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def tab_specs(df: pd.DataFrame):
    st.header("Spec vs Price Analysis")

    rupee_fmt = mticker.FuncFormatter(lambda x, _: f"Rs{x/1000:.0f}k")

    spec_meta = {
        "ram"         : "RAM (GB)",
        "rom"         : "Storage (GB)",
        "battery"     : "Battery (mAh)",
        "display_inch": "Display (inch)",
        "rear_camera" : "Rear Camera (MP)",
        "front_camera": "Front Camera (MP)",
    }
    available = {k: v for k, v in spec_meta.items() if k in df.columns}

    st.subheader("Feature Correlation with Price")
    corrs = {}
    for col in available:
        valid = df[[col, "price"]].dropna()
        r, _ = stats.pearsonr(valid[col], valid["price"])
        corrs[col] = r
    corr_s = pd.Series(corrs).sort_values()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = [GREEN if v > 0 else ACCENT for v in corr_s]
    ax.barh(corr_s.index, corr_s.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Pearson r with Price")
    st.pyplot(fig)
    plt.close()

    st.divider()

    st.subheader("Explore: Spec vs Price")
    sel_spec = st.selectbox("Choose a spec", list(available.keys()),
                            format_func=lambda x: available[x])

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        valid = df[[sel_spec, "price"]].dropna()
        sample = valid.sample(min(500, len(valid)), random_state=42)
        ax.scatter(sample[sel_spec], sample["price"] / 1000,
                   alpha=0.3, s=12, color=BLUE)
        slope, intercept, r_val, _, _ = stats.linregress(
            valid[sel_spec], valid["price"])
        xs = np.linspace(valid[sel_spec].min(), valid[sel_spec].max(), 200)
        ax.plot(xs, (intercept + slope * xs) / 1000,
                color=ACCENT, lw=2, label=f"r = {r_val:.3f}")
        ax.set_xlabel(available[sel_spec])
        ax.set_ylabel("Price (Rs k)")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        top_vals = df[sel_spec].value_counts().nlargest(6).index.sort_values()
        sub = df[df[sel_spec].isin(top_vals)]
        sns.boxplot(data=sub, x=sel_spec, y="price",
                    order=sorted(top_vals), palette="Blues", ax=ax)
        ax.set_xlabel(available[sel_spec])
        ax.set_ylabel("Price (Rs)")
        ax.yaxis.set_major_formatter(rupee_fmt)
        st.pyplot(fig)
        plt.close()

    if "ram" in df.columns and "rom" in df.columns:
        st.subheader("Median Price Heatmap: RAM x Storage")
        pivot = (
            df.groupby(["ram", "rom"])["price"]
              .median()
              .unstack()
              .fillna(0)
        )
        top_ram = df["ram"].value_counts().nlargest(6).index.sort_values()
        top_rom = df["rom"].value_counts().nlargest(6).index.sort_values()
        pivot = pivot.loc[
            pivot.index.isin(top_ram),
            pivot.columns.isin(top_rom)
        ]
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.heatmap(pivot / 1000, annot=True, fmt=".0f",
                    cmap="YlOrRd", linewidths=0.5, ax=ax,
                    cbar_kws={"label": "Median Price (Rs k)"})
        ax.set_xlabel("Storage (GB)")
        ax.set_ylabel("RAM (GB)")
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BRAND EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def tab_brands(df: pd.DataFrame):
    st.header("Brand Explorer")

    brand_stats = (
        df.groupby("brand")
          .agg(
              count        = ("price", "size"),
              median_price = ("price", "median"),
              mean_price   = ("price", "mean"),
              charm_rate   = ("charm_ending", "mean"),
          )
          .query("count >= 5")
          .sort_values("median_price", ascending=False)
    )
    if "rating" in df.columns:
        rating_agg = (
            df.dropna(subset=["rating"])
              .groupby("brand")["rating"]
              .mean()
              .rename("mean_rating")
        )
        brand_stats = brand_stats.join(rating_agg)

    brand_stats["charm_rate_pct"] = (brand_stats["charm_rate"] * 100).round(1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Median Price by Brand (Top 15)")
        top15 = brand_stats.head(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = [ACCENT if r >= 50 else BLUE
                  for r in top15["charm_rate_pct"]]
        ax.barh(top15.index[::-1], top15["median_price"][::-1] / 1000,
                color=colors[::-1], edgecolor="white")
        ax.set_xlabel("Median Price (Rs k)")
        st.pyplot(fig)
        plt.close()

    with col2:
        if "mean_rating" in brand_stats.columns:
            st.subheader("Brand Map: Price vs Rating")
            fig, ax = plt.subplots(figsize=(7, 5))
            sc_data = brand_stats.reset_index()
            ax.scatter(
                sc_data["median_price"] / 1000,
                sc_data["mean_rating"],
                s=sc_data["count"].clip(upper=800),
                alpha=0.6, color=ACCENT,
                edgecolors="white", linewidths=0.6,
            )
            for _, row in sc_data.iterrows():
                ax.annotate(row["brand"],
                            (row["median_price"]/1000, row["mean_rating"]),
                            fontsize=7, alpha=0.85,
                            xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("Median Price (Rs k)")
            ax.set_ylabel("Mean Rating")
            ax.set_title("Bubble size proportional to SKU count")
            st.pyplot(fig)
            plt.close()

    st.subheader("Full Brand Summary Table (5+ SKUs)")
    display_cols = ["count", "median_price", "mean_price", "charm_rate_pct"]
    if "mean_rating" in brand_stats.columns:
        display_cols.append("mean_rating")
    st.dataframe(
        brand_stats[display_cols]
          .rename(columns={
              "count"         : "SKUs",
              "median_price"  : "Median Price",
              "mean_price"    : "Mean Price",
              "charm_rate_pct": "Charm %",
              "mean_rating"   : "Avg Rating",
          }),
        use_container_width=True,
        height=420,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

def tab_data(df: pd.DataFrame):
    st.header("Filtered Dataset")

    st.caption(f"Showing {len(df):,} listings after filters.")

    display_cols = [c for c in
        ["brand", "model", "price", "price_segment",
         "rating", "ratings_count", "ram", "rom",
         "battery", "display_inch", "charm_ending"]
        if c in df.columns]

    st.dataframe(
        df[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=500,
    )

    csv = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv,
        file_name="flipkart_filtered.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not CLEANED_CSV.exists():
        st.error(
            f"Cleaned data not found at `{CLEANED_CSV}`. "
            "Run `python eda.py` first to generate it."
        )
        st.stop()

    df_full = load_data()

    brands, segments, price_range = sidebar(df_full)
    df = apply_filters(df_full, brands, segments, price_range)

    if df.empty:
        st.warning("No listings match the current filters. Adjust the sidebar.")
        st.stop()

    st.title("Flipkart Mobiles 2026 — Analysis Dashboard")
    st.caption(
        f"Filtered: **{len(df):,}** listings  |  "
        f"**{df['brand'].nunique()}** brands  |  "
        f"Price Rs{price_range[0]:,} to Rs{price_range[1]:,}"
    )

    tabs = st.tabs([
        "Overview",
        "Charm Pricing",
        "Ratings",
        "Specs",
        "Brands",
        "Data",
    ])

    with tabs[0]: tab_overview(df)
    with tabs[1]: tab_charm(df)
    with tabs[2]: tab_ratings(df)
    with tabs[3]: tab_specs(df)
    with tabs[4]: tab_brands(df)
    with tabs[5]: tab_data(df)


if __name__ == "__main__":
    main()