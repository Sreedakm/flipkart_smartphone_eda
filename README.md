# Flipkart Mobiles 2026 — EDA & Analysis Project

End-to-end exploratory data analysis and regression modelling on the
Flipkart Mobiles 2026 dataset scraped from Kaggle. The project is broken
into self-contained modules that build on each other in a clear pipeline.

---

## Project Structure

```
├── data/
│   └── flipkart_mobiles_2026.csv        # Raw Kaggle dataset (add manually)
│
├── outputs/
│   ├── week1/
│   │   ├── flipkart_cleaned.csv         # Produced by eda.py — input for all later modules
│   │   └── figures/                     # 11 PNGs from EDA
│   ├── week2_charm/
│   │   └── figures/                     # 7 PNGs from charm_pricing.py
│   ├── week3_ratings/
│   │   └── figures/                     # 8 PNGs from rating_analysis.py
│   └── week4_regression/
│       ├── regression_results.csv       # Model comparison table
│       └── figures/                     # 8 PNGs from spec_regression.py
│
├── eda.py                               # Module 1 — Load, clean, univariate & bivariate EDA
├── charm_pricing.py                     # Module 2 — Psychological pricing patterns
├── rating_analysis.py                   # Module 3 — Rating & review quality analysis
├── spec_regression.py                   # Module 4 — Spec-based price regression
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** Kaggle — Flipkart Mobiles 2026  
**File:** `data/flipkart_mobiles_2026.csv`

| Column | Description |
|---|---|
| `model` | Full product listing name |
| `brand` | Brand name |
| `price` | Listed price in ₹ |
| `rating` | Average customer rating (0–5) |
| `ratings_count` | Total number of ratings |
| `reviews_count` | Total number of written reviews |
| `ram` | RAM in GB |
| `rom` | Internal storage in GB |
| `battery` | Battery capacity in mAh |
| `display_inch` | Screen size in inches |
| `rear_camera` | Rear camera spec string |
| `front_camera` | Front camera spec string |
| `processor` | Processor name string |

The raw file is **not cleaned** — prices contain ₹ symbols and commas, camera columns are strings like `"50MP + 12MP"`, and there are duplicates. `eda.py` handles all of this.

---

## Setup

```bash
# 1. Clone / download the project folder
cd flipkart-eda

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the Kaggle dataset
mkdir -p data
# Move flipkart_mobiles_2026.csv into data/
```

---

## Running the Modules

Run them **in order** — each module reads the cleaned CSV produced by `eda.py`.

### Module 1 — EDA & Cleaning

```bash
python eda.py
# Output: outputs/week1/flipkart_cleaned.csv + 11 figures
```

**What it does:**
- Normalises column names, coerces numeric types, strips ₹/commas from price
- Drops rows with missing price, removes (model, brand, price) duplicates
- Flags IQR outliers, creates `price_segment` and `charm_ending` derived columns
- Univariate EDA: price histogram (raw + log), segment bar, rating distribution, spec distributions
- Bivariate EDA: price vs rating scatter with OLS, price by RAM boxplot, violin by segment, correlation heatmap
- Brand-level summary: median price bar, brand map (price vs rating), SKU volume

---

### Module 2 — Charm Pricing

```bash
python charm_pricing.py
# Output: outputs/week2_charm/figures/ (7 PNGs)
```

**What it does:**
- Calculates the overall rate of charm-priced listings (prices ending in ×99, ×999, etc.)
- Last-3-digit frequency bar chart and last-2-digit heatmap
- Charm rate broken down by price segment (grouped bar + 100% stacked)
- Charm rate by brand — diverging bar from the 50% midpoint
- Tests whether charm pricing correlates with higher ratings (Welch t-test + KDE)
- Tests whether charm-priced phones get more ratings (Mann-Whitney U)
- Classifies rounding patterns: round ×000 / just-below-K (999) / (990) / ×99 / other

---

### Module 3 — Rating Analysis

```bash
python rating_analysis.py
# Output: outputs/week3_ratings/figures/ (8 PNGs)
```

**What it does:**
- Deep-dive on rating distribution: skew, kurtosis, percentile lines, band breakdown
- Classifies listings into engagement tiers by `ratings_count` (Niche → Viral)
- Rating vs price: mean per segment with Kruskal-Wallis significance test + violin + hexbin
- Rating by brand: mean ± SD, best and worst rated brands
- Rating vs specs: Pearson r for RAM, storage, battery, display + scatter grid
- Review-to-rating ratio as an engagement quality signal
- Top-10 and bottom-10 rated phones (filtered to ≥30 ratings) as a lollipop chart

---

### Module 4 — Spec Regression

```bash
python spec_regression.py
# Output: outputs/week4_regression/ (8 PNGs + regression_results.csv)
```

**What it does:**
- Feature engineering: `ram_x_rom`, `total_mp`, `battery_per_inch` interaction terms
- Feature–price correlation bar chart and scatter grid
- 80/20 train–test split with 5-fold cross-validation throughout
- **OLS regression** on log-transformed price — coefficients, R², RMSE, MAPE
- **Ridge (α=10)** and **Lasso (α=100)** regularised variants
- **Random Forest** (300 trees) — typically the best performer
- Model comparison table saved to `regression_results.csv`
- Gini and permutation feature importances for Random Forest
- Residual analysis: distribution, Q-Q plot, Shapiro-Wilk normality test

**Typical results (indicative — will vary by dataset size):**

| Model | R² | RMSE |
|---|---|---|
| OLS | ~0.55 | ~₹12,000 |
| Ridge | ~0.55 | ~₹12,000 |
| Lasso | ~0.54 | ~₹12,200 |
| Random Forest | ~0.75 | ~₹8,000 |

The gap between OLS and RF shows that spec–price relationships are non-linear (e.g. premium brands extract price far above what RAM/storage alone would predict).

---

## Key Findings (summary)

- **Price distribution** is heavily right-skewed; Budget and Entry-Mid dominate by volume.
- **Charm pricing** is widespread across all segments. Budget phones charm-price at a higher rate than Premium.
- **Ratings** cluster tightly between 4.0 and 4.5 — typical of Flipkart's rating inflation. Almost no listing rates below 3.5.
- **RAM** is the strongest spec predictor of price (r ≈ 0.55–0.65 depending on dataset), followed by storage.
- **Battery and display size** have weak positive correlation with price.
- **Random Forest** explains ~75% of price variance from specs alone; the remaining ~25% is brand premium, launch recency, and features not captured in the spec columns.

The results are presented on a dashboard: https://flipkartsmartphoneeda-5nteqcvn8ucpswoarrhehs.streamlit.app/
---

## Possible Extensions

| Module | Idea |
|---|---|
| `brand_deep_dive.py` | Per-brand spec positioning maps, value-for-money index |
| `processor_analysis.py` | Cluster processors by price tier (Dimensity / Snapdragon / Exynos) |
| `ml_features.py` | Full feature engineering pipeline for a price prediction API |
| `dashboard.py` | Streamlit dashboard wrapping all four analyses |
| `nlp_reviews.py` | Sentiment analysis on review text if review text data is available |

---

## Notes

- All modules are standalone scripts (no shared state between runs beyond the cleaned CSV).
- Figures are saved at 150 DPI to `figures/` subdirectories; existing files are overwritten on re-run.
- The `rear_camera` and `front_camera` columns contain messy strings like `"50MP + 12MP"`. `eda.py` extracts only the leading numeric value (first MP figure). `spec_regression.py` derives `total_mp` from these parsed values.
- Duplicate listings (same model, brand, price) are dropped in `eda.py`. Listings where only colour or RAM variant differs but price is identical are treated as duplicates.
