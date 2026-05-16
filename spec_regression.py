import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_INPUT = Path("outputs/week1/flipkart_cleaned.csv")
OUTPUT_DIR    = Path("outputs/week4_regression")
FIGURES_DIR   = OUTPUT_DIR / "figures"
RESULTS_CSV   = OUTPUT_DIR / "regression_results.csv"

# ── Visual theme ───────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
ACCENT = "#E44D26"
BLUE   = "#4A90D9"
GREEN  = "#5BAD6F"
PURPLE = "#9B59B6"

SPEC_COLS = ["ram", "rom", "battery", "display_inch",
             "rear_camera", "front_camera"]
TARGET    = "price"

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


def metrics(y_true, y_pred, label: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    print(f"\n  [{label}]")
    print(f"    MAE  : ₹{mae:,.0f}")
    print(f"    RMSE : ₹{rmse:,.0f}")
    print(f"    R²   : {r2:.4f}")
    print(f"    MAPE : {mape:.2f}%")
    return dict(model=label, mae=mae, rmse=rmse, r2=r2, mape=mape)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_engineer(path: Path) -> tuple[pd.DataFrame, list[str]]:
    section("1. LOAD & FEATURE ENGINEERING")

    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    available = [c for c in SPEC_COLS if c in df.columns]
    print(f"  Spec columns found : {available}")

    # ── Keep rows where target and all available specs are present
    df = df.dropna(subset=[TARGET] + available)
    df = df[df[TARGET].between(1_000, 3_00_000)]
    print(f"  After dropna       : {len(df):,} rows")

    # ── Derived / interaction features
    if "ram" in df.columns and "rom" in df.columns:
        df["ram_x_rom"] = df["ram"] * df["rom"]

    if "rear_camera" in df.columns and "front_camera" in df.columns:
        df["total_mp"] = df["rear_camera"].fillna(0) + df["front_camera"].fillna(0)

    if "battery" in df.columns and "display_inch" in df.columns:
        df["battery_per_inch"] = (
            df["battery"] / df["display_inch"].replace(0, np.nan)
        )

    # ── Log-transform target for OLS (price is right-skewed)
    df["price_log"] = np.log1p(df[TARGET])

    # Build final feature list
    engineered = ["ram_x_rom", "total_mp", "battery_per_inch"]
    feat_cols  = available + [c for c in engineered if c in df.columns]

    # Fill remaining NaN in features with column median
    for c in feat_cols:
        df[c] = df[c].fillna(df[c].median())

    print(f"  Final feature list : {feat_cols}")
    print(f"  Final dataset      : {df.shape}")
    return df, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2. EDA — FEATURE CORRELATIONS WITH PRICE
# ══════════════════════════════════════════════════════════════════════════════

def feature_eda(df: pd.DataFrame, feat_cols: list[str]):
    section("2. FEATURE–PRICE CORRELATIONS")

    rupee_fmt = mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k")

    corrs = {}
    for col in feat_cols:
        r, p = stats.pearsonr(df[col], df[TARGET])
        corrs[col] = dict(pearson_r=round(r, 4), p_value=round(p, 6))
        print(f"  {col:22s}  r = {r:+.4f}   p = {p:.3g}")

    # ── Fig SR01: Correlation bar chart
    r_vals = {c: v["pearson_r"] for c, v in corrs.items()}
    r_series = pd.Series(r_vals).sort_values()
    colors = [GREEN if v > 0 else ACCENT for v in r_series]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(r_series.index, r_series.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    for bar, val in zip(ax.patches, r_series.values):
        ax.text(val + (0.01 if val >= 0 else -0.01),
                bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", fontsize=9)
    ax.set_xlabel("Pearson r with Price")
    ax.set_title("Feature Correlation with Price", fontweight="bold")
    plt.tight_layout()
    save_fig("SR01_feature_correlations")

    # ── Fig SR02: Scatter grid — each feature vs price
    n_cols = 3
    n_rows = (len(feat_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten()
    palette = [BLUE, GREEN, ACCENT, PURPLE, "#F39C12", "#1ABC9C",
               "#E74C3C", "#3498DB", "#2ECC71"]

    for i, (ax, col) in enumerate(zip(axes, feat_cols)):
        sample = df.sample(min(3_000, len(df)), random_state=42)
        ax.scatter(sample[col], sample[TARGET],
                   alpha=0.2, s=8, color=palette[i % len(palette)])
        slope, intercept, r, _, _ = stats.linregress(df[col], df[TARGET])
        xs = np.linspace(df[col].min(), df[col].max(), 200)
        ax.plot(xs, intercept + slope * xs, color="black", lw=1.8,
                label=f"r = {r:.3f}")
        ax.set_xlabel(col)
        ax.set_ylabel("Price (₹)")
        ax.yaxis.set_major_formatter(rupee_fmt)
        ax.set_title(f"{col} vs Price", fontweight="bold")
        ax.legend(fontsize=8)

    for ax in axes[len(feat_cols):]:
        ax.set_visible(False)

    plt.suptitle("Spec Features vs Price (scatter + OLS)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("SR02_feature_scatter_grid")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split(df: pd.DataFrame, feat_cols: list[str]):
    section("3. TRAIN / TEST SPLIT  (80 / 20)")
    X = df[feat_cols].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"  Train : {X_train.shape[0]:,}  |  Test : {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 4. BASELINE OLS (log-price target for normality)
# ══════════════════════════════════════════════════════════════════════════════

def ols_model(df: pd.DataFrame, feat_cols: list[str],
              X_train, X_test, y_train, y_test) -> dict:
    section("4. OLS REGRESSION  (log-price target)")

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    y_tr_log = np.log1p(y_train)
    y_te_log = np.log1p(y_test)

    ols = LinearRegression()
    ols.fit(X_tr_sc, y_tr_log)

    # Coefficients
    coef_df = pd.DataFrame({
        "feature"    : feat_cols,
        "coefficient": ols.coef_,
    }).sort_values("coefficient", ascending=False)
    print(f"\n  OLS coefficients (on log-price, standardised features):")
    print(coef_df.to_string(index=False))

    # Back-transform predictions to ₹
    y_pred_log = ols.predict(X_te_sc)
    y_pred     = np.expm1(y_pred_log)
    result     = metrics(y_test, y_pred, "OLS (log target)")

    # 5-fold CV R² on log-price
    cv_scores = cross_val_score(ols, X_tr_sc, y_tr_log,
                                cv=KFold(5, shuffle=True, random_state=42),
                                scoring="r2")
    print(f"\n  5-fold CV R² (log-price) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Fig SR03: Actual vs Predicted (OLS)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    rupee_fmt = mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k")

    axes[0].scatter(y_test / 1000, y_pred / 1000, alpha=0.3, s=10, color=BLUE)
    lim = max(y_test.max(), y_pred.max()) / 1000
    axes[0].plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Price (₹k)")
    axes[0].set_ylabel("Predicted Price (₹k)")
    axes[0].set_title("OLS: Actual vs Predicted", fontweight="bold")
    axes[0].legend()

    residuals = y_test - y_pred
    axes[1].scatter(y_pred / 1000, residuals / 1000, alpha=0.3, s=10, color=ACCENT)
    axes[1].axhline(0, color="black", lw=1.5, ls="--")
    axes[1].set_xlabel("Predicted Price (₹k)")
    axes[1].set_ylabel("Residual (₹k)")
    axes[1].set_title("OLS: Residuals vs Predicted", fontweight="bold")

    plt.suptitle("OLS Regression Diagnostics", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("SR03_ols_diagnostics")

    # ── Fig SR04: Coefficient bar
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [GREEN if v > 0 else ACCENT for v in coef_df["coefficient"]]
    ax.barh(coef_df["feature"][::-1], coef_df["coefficient"][::-1],
            color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Coefficient (standardised, log-price)")
    ax.set_title("OLS Regression Coefficients", fontweight="bold")
    plt.tight_layout()
    save_fig("SR04_ols_coefficients")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5. RIDGE & LASSO
# ══════════════════════════════════════════════════════════════════════════════

def regularised_models(X_train, X_test, y_train, y_test) -> list[dict]:
    section("5. RIDGE & LASSO REGRESSION")

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    y_tr_log = np.log1p(y_train)

    results = []
    for name, model, color in [
        ("Ridge (α=10)",   Ridge(alpha=10),    BLUE),
        ("Lasso (α=100)",  Lasso(alpha=100),   PURPLE),
    ]:
        model.fit(X_tr_sc, y_tr_log)
        y_pred = np.expm1(model.predict(X_te_sc))
        result = metrics(y_test, y_pred, name)
        results.append(result)

        cv = cross_val_score(
            model, X_tr_sc, y_tr_log,
            cv=KFold(5, shuffle=True, random_state=42), scoring="r2"
        )
        print(f"    5-fold CV R² : {cv.mean():.4f} ± {cv.std():.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════

def random_forest(df: pd.DataFrame, feat_cols: list[str],
                  X_train, X_test, y_train, y_test) -> tuple[dict, RandomForestRegressor]:
    section("6. RANDOM FOREST REGRESSION")

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    result = metrics(y_test, y_pred, "Random Forest")

    cv = cross_val_score(
        rf, X_train, y_train,
        cv=KFold(5, shuffle=True, random_state=42), scoring="r2"
    )
    print(f"  5-fold CV R² : {cv.mean():.4f} ± {cv.std():.4f}")

    # ── Feature importances (Gini)
    fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"\n  Feature importances (Gini impurity):\n{fi.round(4).to_string()}")

    # Permutation importance on test set
    perm = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                  random_state=42, n_jobs=-1)
    perm_fi = pd.Series(perm.importances_mean, index=feat_cols).sort_values(ascending=False)
    print(f"\n  Permutation importances:\n{perm_fi.round(4).to_string()}")

    # ── Fig SR05: RF Actual vs Predicted
    rupee_fmt = mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].scatter(y_test / 1000, y_pred / 1000, alpha=0.3, s=10, color=GREEN)
    lim = max(y_test.max(), y_pred.max()) / 1000
    axes[0].plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Price (₹k)")
    axes[0].set_ylabel("Predicted Price (₹k)")
    axes[0].set_title("Random Forest: Actual vs Predicted", fontweight="bold")
    axes[0].legend()

    residuals = y_test - y_pred
    axes[1].scatter(y_pred / 1000, residuals / 1000, alpha=0.3, s=10, color=ACCENT)
    axes[1].axhline(0, color="black", lw=1.5, ls="--")
    axes[1].set_xlabel("Predicted Price (₹k)")
    axes[1].set_ylabel("Residual (₹k)")
    axes[1].set_title("Random Forest: Residuals vs Predicted", fontweight="bold")

    plt.suptitle("Random Forest Diagnostics", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("SR05_rf_diagnostics")

    # ── Fig SR06: Feature importances — Gini vs Permutation
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    fi_plot = fi.sort_values()
    axes[0].barh(fi_plot.index, fi_plot.values, color=GREEN, edgecolor="white")
    axes[0].set_xlabel("Gini Importance")
    axes[0].set_title("RF Feature Importance (Gini)", fontweight="bold")

    perm_plot = perm_fi.sort_values()
    axes[1].barh(perm_plot.index, perm_plot.values, color=BLUE, edgecolor="white")
    axes[1].set_xlabel("Mean Decrease in R² (permuted)")
    axes[1].set_title("RF Permutation Importance", fontweight="bold")

    plt.suptitle("Random Forest — Feature Importance", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("SR06_rf_feature_importance")

    return result, rf


# ══════════════════════════════════════════════════════════════════════════════
# 7. MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def model_comparison(all_results: list[dict]):
    section("7. MODEL COMPARISON")

    results_df = pd.DataFrame(all_results).set_index("model")
    print(f"\n{results_df.round(2).to_string()}")
    results_df.to_csv(RESULTS_CSV)
    print(f"\n  Saved → {RESULTS_CSV}")

    # ── Fig SR07: Comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = [BLUE, BLUE, PURPLE, GREEN]

    for ax, metric, label in [
        (axes[0], "r2",   "R²  (higher is better)"),
        (axes[1], "rmse", "RMSE ₹  (lower is better)"),
        (axes[2], "mape", "MAPE %  (lower is better)"),
    ]:
        vals   = results_df[metric]
        colors = [GREEN if (metric == "r2" and v == vals.max()) or
                           (metric != "r2" and v == vals.min())
                  else ACCENT
                  for v in vals]
        ax.bar(vals.index, vals.values, color=colors, edgecolor="white")
        for bar, val in zip(ax.patches, vals.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.01,
                    f"{val:.2f}", ha="center", fontsize=9)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)

    plt.suptitle("Model Comparison — Spec-Based Price Regression",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("SR07_model_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# 8. RESIDUAL DEEP-DIVE (best model = RF)
# ══════════════════════════════════════════════════════════════════════════════

def residual_analysis(df: pd.DataFrame, feat_cols: list[str],
                      rf: RandomForestRegressor,
                      X_test, y_test):
    section("8. RESIDUAL DEEP-DIVE  (Random Forest)")

    y_pred    = rf.predict(X_test)
    residuals = y_test - y_pred

    # Residual distribution stats
    print(f"  Mean residual  : ₹{residuals.mean():,.0f}")
    print(f"  Std residual   : ₹{residuals.std():,.0f}")
    print(f"  Max overshoot  : ₹{residuals.max():,.0f}")
    print(f"  Max undershoot : ₹{residuals.min():,.0f}")

    # Shapiro-Wilk on sample
    _, p_sw = stats.shapiro(residuals[:min(500, len(residuals))])
    print(f"  Shapiro-Wilk p : {p_sw:.4g}"
          f"  ({'normal' if p_sw > 0.05 else 'non-normal'})")

    # ── Fig SR08: Residual histogram + Q-Q
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residuals / 1000, bins=60, color=BLUE,
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color=ACCENT, lw=2, ls="--")
    axes[0].set_xlabel("Residual (₹k)")
    axes[0].set_title("Residual Distribution", fontweight="bold")

    stats.probplot(residuals, plot=axes[1])
    axes[1].get_lines()[1].set_color(ACCENT)
    axes[1].set_title("Q-Q Plot of Residuals", fontweight="bold")

    plt.suptitle("Random Forest — Residual Analysis",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("SR08_residual_analysis")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Module 4 — Spec-Based Price Regression")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    args = parser.parse_args()

    ensure_dirs()

    df, feat_cols = load_and_engineer(Path(args.input))
    feature_eda(df, feat_cols)

    X_train, X_test, y_train, y_test = split(df, feat_cols)

    all_results = []
    all_results.append(ols_model(df, feat_cols, X_train, X_test, y_train, y_test))
    all_results += regularised_models(X_train, X_test, y_train, y_test)
    rf_result, rf = random_forest(df, feat_cols, X_train, X_test, y_train, y_test)
    all_results.append(rf_result)

    model_comparison(all_results)
    residual_analysis(df, feat_cols, rf, X_test, y_test)

    section("COMPLETE ✓")
    print(f"  Figures  → {FIGURES_DIR}/   (8 PNGs)")
    print(f"  Results  → {RESULTS_CSV}")
    print("\n  Next →  brand_deep_dive.py  |  ml_features.py  |  dashboard.py")


if __name__ == "__main__":
    main()