#!/usr/bin/env python3
"""
============================================================================
UIDAI DATA HACKATHON 2026 - HYBRID RISK SCORING MODEL
============================================================================
File: 06_hybrid_model.py
Purpose: Combine unsupervised (Isolation Forest, Autoencoder) and
         supervised (XGBoost, Random Forest) models into a single
         unified fraud/anomaly risk score.
Author: Generated with AI Assistance
Date: January 2026
============================================================================
This script:
1. Loads:
   - 04_anomaly_scores.csv  (Isolation Forest + Autoencoder scores)
   - 05_fraud_predictions.csv (fraud probabilities)
2. Normalizes and aligns all model outputs.
3. Builds a hybrid risk score (0–100) using weighted ensemble.
4. Assigns final risk bands (Very Low … Critical).
5. Produces final, flattened output for dashboards & analysts.
6. Saves:
   - 06_hybrid_results.csv          (full record-level output)
   - hybrid_predictions.csv         (compact modeling table)
   - top_100_high_risk_records.csv  (for manual review)
   - 04_hybrid_comparison.png       (visual comparison of all models)
   - 06_hybrid_model_report.txt     (text report)
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "outputs" / "data"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
VIZ_DIR = PROJECT_ROOT / "outputs" / "visualizations"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

VIZ_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ANOMALY_FILE = DATA_DIR / "04_anomaly_scores.csv"
FRAUD_FILE = DATA_DIR / "05_fraud_predictions.csv"
HYBRID_FULL_FILE = DATA_DIR / "06_hybrid_results.csv"
HYBRID_COMPACT_FILE = REPORTS_DIR / "hybrid_predictions.csv"
TOP_100_FILE = REPORTS_DIR / "top_100_high_risk_records.csv"
REPORT_FILE = REPORTS_DIR / "06_hybrid_model_report.txt"

# Ensemble weights – can be tuned
WEIGHT_IFOREST = 0.20   # Isolation Forest (unsupervised)[web:29]
WEIGHT_AE      = 0.20   # Autoencoder (unsupervised)[web:29]
WEIGHT_XGB     = 0.35   # XGBoost fraud probability (supervised)[web:29]
WEIGHT_RF      = 0.25   # Random Forest fraud probability (supervised)[web:29]

# Ensure weights sum to 1.0
TOTAL_W = WEIGHT_IFOREST + WEIGHT_AE + WEIGHT_XGB + WEIGHT_RF
WEIGHT_IFOREST /= TOTAL_W
WEIGHT_AE      /= TOTAL_W
WEIGHT_XGB     /= TOTAL_W
WEIGHT_RF      /= TOTAL_W

# ============================================================================
# PRINT HELPERS
# ============================================================================

def print_header(text: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def print_section(text: str) -> None:
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print("─" * 80)

def print_success(text: str) -> None:
    print(f"✅ {text}")

def print_info(text: str) -> None:
    print(f"ℹ️  {text}")

def print_warning(text: str) -> None:
    print(f"⚠️  {text}")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_inputs():
    """Load anomaly and fraud prediction datasets and merge on common keys."""
    print_section("STEP 1: LOADING INPUT DATA")

    try:
        df_anom = pd.read_csv(ANOMALY_FILE)
        print_success(f"Loaded anomaly file: {ANOMALY_FILE} ({len(df_anom):,} rows)")
    except FileNotFoundError:
        print_warning(f"Missing file: {ANOMALY_FILE}")
        print_info("Run 04_anomaly_detection.py first.")
        sys.exit(1)

    try:
        df_fraud = pd.read_csv(FRAUD_FILE)
        print_success(f"Loaded fraud file: {FRAUD_FILE} ({len(df_fraud):,} rows)")
    except FileNotFoundError:
        print_warning(f"Missing file: {FRAUD_FILE}")
        print_info("Run 05_fraud_classification.py first.")
        sys.exit(1)

    # Identify join key (prefer enrolment_id if present)
    join_cols = []
    if "enrolment_id" in df_anom.columns and "enrolment_id" in df_fraud.columns:
        join_cols = ["enrolment_id"]
    else:
        # Fallback: index-based merge if structure matches
        print_warning("enrolment_id not found in both files; merging on row index.")
        df_anom = df_anom.reset_index().rename(columns={"index": "row_id"})
        df_fraud = df_fraud.reset_index().rename(columns={"index": "row_id"})
        join_cols = ["row_id"]

    df_merged = df_anom.merge(df_fraud, on=join_cols, suffixes=("_anom", "_fraud"))
    print_success(f"Merged dataset shape: {df_merged.shape[0]:,} × {df_merged.shape[1]}")

    return df_merged, join_cols

# ============================================================================
# SCORE NORMALIZATION & HYBRID SCORE
# ============================================================================

def normalize_scores(df):
    """Normalize all component scores to 0–100 scale."""
    print_section("STEP 2: NORMALIZING MODEL OUTPUTS")

    # Expectation:
    # - iso_forest_score, autoencoder_score, combined_anomaly_score: already 0–100
    # - fraud_probability: 0–1 -> convert to 0–100[web:61]
    # Columns might have suffixes after merge.
    col_iso = [c for c in df.columns if "iso_forest_score" in c][0]
    col_ae  = [c for c in df.columns if "autoencoder_score" in c][0]
    col_comb_anom = [c for c in df.columns if "combined_anomaly_score" in c][0]
    col_fraud_prob = [c for c in df.columns if "fraud_probability" in c][0]

    # Copy to canonical names
    df["iso_score_0_100"] = df[col_iso].astype(float)
    df["ae_score_0_100"] = df[col_ae].astype(float)

    # Combined anomaly score already 0–100 from previous script
    df["combined_anomaly_0_100"] = df[col_comb_anom].astype(float)

    # Fraud probability: 0–1 -> 0–100
    df["fraud_prob_0_100"] = (df[col_fraud_prob].clip(0, 1) * 100.0).astype(float)

    print_success("Standardized component scores:")
    print_info("  iso_score_0_100, ae_score_0_100, combined_anomaly_0_100, fraud_prob_0_100")

    return df

def compute_hybrid_score(df):
    """Compute final hybrid risk score (0–100) from all four models."""
    print_section("STEP 3: COMPUTING HYBRID RISK SCORE")

    # Use:
    # - combined_anomaly_0_100 as overall unsupervised anomaly signal
    # - iso_score_0_100 and ae_score_0_100 separately
    # - fraud_prob_0_100 for supervised probability[web:29]

    # Convert all to 0–1 before weighting
    iso_norm  = df["iso_score_0_100"] / 100.0
    ae_norm   = df["ae_score_0_100"] / 100.0
    comb_norm = df["combined_anomaly_0_100"] / 100.0
    fraud_norm = df["fraud_prob_0_100"] / 100.0

    # Give supervised models slightly higher overall weight as they are
    # precision-oriented, while anomaly models add sensitivity.[web:29]
    # Here: (IF + AE) 40%, XGB 35%, RF 25% (RF assumed embedded in fraud_prob).[web:29]
    unsup_part = (iso_norm * WEIGHT_IFOREST) + (ae_norm * WEIGHT_AE)
    sup_part   = fraud_norm * (WEIGHT_XGB + WEIGHT_RF)

    hybrid_0_1 = unsup_part + sup_part
    # Clip to [0,1] to guard against any numerical drift.[web:65]
    hybrid_0_1 = hybrid_0_1.clip(0, 1)

    df["hybrid_risk_score"] = (hybrid_0_1 * 100.0).round(2)

    print_success(
        f"Hybrid score range: {df['hybrid_risk_score'].min():.2f} – "
        f"{df['hybrid_risk_score'].max():.2f}"
    )
    print_info(f"Mean hybrid score: {df['hybrid_risk_score'].mean():.2f}")

    return df

# ============================================================================
# RISK BANDING
# ============================================================================

def assign_risk_band(score: float) -> str:
    """
    Map 0–100 hybrid risk score to human-friendly bands.
    Pattern similar to many fraud-score systems (e.g. 0–30 low, 31–60 medium, 61–100 high),
    but with finer granularity for this hackathon.[web:59][web:63][web:66]
    """
    if score >= 90:
        return "Critical"
    elif score >= 75:
        return "Very High"
    elif score >= 55:
        return "High"
    elif score >= 35:
        return "Medium"
    elif score >= 15:
        return "Low"
    else:
        return "Very Low"

def apply_risk_bands(df):
    """Apply risk banding to hybrid scores."""
    print_section("STEP 4: ASSIGNING FINAL RISK BANDS")

    df["hybrid_risk_band"] = df["hybrid_risk_score"].apply(assign_risk_band)

    counts = Counter(df["hybrid_risk_band"])
    print("Hybrid Risk Band Distribution:")
    for band in ["Critical", "Very High", "High", "Medium", "Low", "Very Low"]:
        if band in counts:
            n = counts[band]
            pct = (n / len(df)) * 100
            print(f"  {band:10s}: {n:7,} ({pct:5.2f}%)")

    print_success("Risk bands assigned.")
    return df

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(df):
    """Create comparison plots for all model scores and hybrid output."""
    print_section("STEP 5: CREATING VISUALIZATIONS")

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    fig.suptitle(
        "Hybrid Fraud & Anomaly Risk Scoring\n"
        "(Isolation Forest + Autoencoder + XGBoost + Random Forest)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # 1. Score distributions
    ax1 = fig.add_subplot(gs[0, 0])
    for col, color, label in [
        ("iso_score_0_100", "steelblue", "Isolation Forest"),
        ("ae_score_0_100", "coral", "Autoencoder"),
        ("fraud_prob_0_100", "darkgreen", "Fraud Probability"),
        ("hybrid_risk_score", "purple", "Hybrid Score"),
    ]:
        sns.kdeplot(
            df[col],
            ax=ax1,
            label=label,
            linewidth=2,
            fill=False,
            color=color,
            alpha=0.8,
        )
    ax1.set_title("Score Distributions (0–100)", fontweight="bold")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Hybrid vs Fraud Probability
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(
        x="fraud_prob_0_100",
        y="hybrid_risk_score",
        data=df.sample(min(10000, len(df)), random_state=42),
        alpha=0.4,
        s=10,
        color="mediumpurple",
        edgecolor=None,
        ax=ax2,
    )
    ax2.set_title("Hybrid Score vs Fraud Probability", fontweight="bold")
    ax2.set_xlabel("Fraud Probability (0–100)")
    ax2.set_ylabel("Hybrid Risk Score (0–100)")
    ax2.grid(True, alpha=0.3)

    # 3. Hybrid vs Combined Anomaly
    ax3 = fig.add_subplot(gs[0, 2])
    sns.scatterplot(
        x="combined_anomaly_0_100",
        y="hybrid_risk_score",
        data=df.sample(min(10000, len(df)), random_state=42),
        alpha=0.4,
        s=10,
        color="teal",
        edgecolor=None,
        ax=ax3,
    )
    ax3.set_title("Hybrid Score vs Combined Anomaly", fontweight="bold")
    ax3.set_xlabel("Combined Anomaly Score (0–100)")
    ax3.set_ylabel("Hybrid Risk Score (0–100)")
    ax3.grid(True, alpha=0.3)

    # 4. Risk band distribution
    ax4 = fig.add_subplot(gs[1, 0])
    band_order = ["Very Low", "Low", "Medium", "High", "Very High", "Critical"]
    band_counts = df["hybrid_risk_band"].value_counts().reindex(band_order)
    band_counts.plot(kind="bar", color="orange", ax=ax4)
    ax4.set_title("Hybrid Risk Band Distribution", fontweight="bold")
    ax4.set_xlabel("Risk Band")
    ax4.set_ylabel("Count")
    for idx, val in enumerate(band_counts.values):
        if not np.isnan(val):
            ax4.text(idx, val, f"{int(val):,}", ha="center", va="bottom", fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. State-wise average hybrid score (if state exists)
    ax5 = fig.add_subplot(gs[1, 1])
    if any(c.startswith("state") for c in df.columns):
        state_col = [c for c in df.columns if c.startswith("state")][0]
        state_avg = (
            df.groupby(state_col)["hybrid_risk_score"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        state_avg.plot(kind="barh", ax=ax5, color="salmon")
        ax5.invert_yaxis()
        ax5.set_title("Top 10 States by Avg Hybrid Score", fontweight="bold")
        ax5.set_xlabel("Average Hybrid Score")
        for i, v in enumerate(state_avg.values):
            ax5.text(v, i, f"{v:.1f}", va="center", fontsize=8)
        ax5.grid(True, alpha=0.3, axis="x")
    else:
        ax5.text(
            0.5,
            0.5,
            "State column not available",
            ha="center",
            va="center",
            transform=ax5.transAxes,
        )
        ax5.axis("off")

    # 6. Age vs hybrid score (if age exists)
    ax6 = fig.add_subplot(gs[1, 2])
    age_cols = [c for c in df.columns if "age" == c or c.endswith("_age")]
    if age_cols:
        age_col = age_cols[0]
        sns.scatterplot(
            x=age_col,
            y="hybrid_risk_score",
            data=df.sample(min(10000, len(df)), random_state=42),
            alpha=0.3,
            s=10,
            color="darkred",
            edgecolor=None,
            ax=ax6,
        )
        ax6.set_title("Hybrid Score vs Age", fontweight="bold")
        ax6.set_xlabel("Age")
        ax6.set_ylabel("Hybrid Risk Score")
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(
            0.5,
            0.5,
            "Age column not available",
            ha="center",
            va="center",
            transform=ax6.transAxes,
        )
        ax6.axis("off")

    # 7–8. Boxplots by risk band
    ax7 = fig.add_subplot(gs[2, 0])
    sns.boxplot(
        x="hybrid_risk_band",
        y="fraud_prob_0_100",
        data=df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["hybrid_risk_band", "fraud_prob_0_100"]
        ),
        order=band_order,
        ax=ax7,
        color="lightblue",
    )
    ax7.set_title("Fraud Probability by Hybrid Band", fontweight="bold")
    ax7.set_xlabel("Hybrid Risk Band")
    ax7.set_ylabel("Fraud Probability (0–100)")
    ax7.grid(True, alpha=0.3, axis="y")

    ax8 = fig.add_subplot(gs[2, 1])
    sns.boxplot(
        x="hybrid_risk_band",
        y="combined_anomaly_0_100",
        data=df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["hybrid_risk_band", "combined_anomaly_0_100"]
        ),
        order=band_order,
        ax=ax8,
        color="lightgreen",
    )
    ax8.set_title("Combined Anomaly by Hybrid Band", fontweight="bold")
    ax8.set_xlabel("Hybrid Risk Band")
    ax8.set_ylabel("Combined Anomaly Score (0–100)")
    ax8.grid(True, alpha=0.3, axis="y")

    # 9. Text summary panel
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")

    text_lines = []
    text_lines.append("HYBRID MODEL SUMMARY")
    text_lines.append("────────────────────")
    text_lines.append(f"Records: {len(df):,}")
    text_lines.append(
        f"Hybrid score: {df['hybrid_risk_score'].min():.1f} – "
        f"{df['hybrid_risk_score'].max():.1f}"
    )
    text_lines.append(
        f"Mean: {df['hybrid_risk_score'].mean():.1f} | "
        f"Std: {df['hybrid_risk_score'].std():.1f}"
    )
    text_lines.append("")
    text_lines.append("Risk Bands:")
    counts = Counter(df["hybrid_risk_band"])
    for band in ["Critical", "Very High", "High", "Medium", "Low", "Very Low"]:
        if band in counts:
            n = counts[band]
            pct = (n / len(df)) * 100
            text_lines.append(f"  {band:10s}: {n:7,} ({pct:5.2f}%)")
    text_lines.append("")
    text_lines.append("Weights:")
    text_lines.append(f"  Isolation Forest : {WEIGHT_IFOREST:0.2f}")
    text_lines.append(f"  Autoencoder      : {WEIGHT_AE:0.2f}")
    text_lines.append(f"  XGBoost + RF     : {WEIGHT_XGB + WEIGHT_RF:0.2f}")

    ax9.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        fontsize=9,
        va="top",
        family="monospace",
    )

    output_path = VIZ_DIR / "04_hybrid_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print_success(f"Saved visualization: {output_path}")

# ============================================================================
# SAVE RESULTS & REPORT
# ============================================================================

def save_outputs(df, join_cols):
    """Save full and compact hybrid outputs plus top-k records."""
    print_section("STEP 6: SAVING HYBRID OUTPUTS")

    # Full file
    df.to_csv(HYBRID_FULL_FILE, index=False)
    print_success(f"Full hybrid results saved: {HYBRID_FULL_FILE}")

    # Compact table for dashboards / BI
    id_cols = [c for c in ["enrolment_id", "row_id"] if c in df.columns]
    compact_cols = (
        id_cols
        + [
            "iso_score_0_100",
            "ae_score_0_100",
            "combined_anomaly_0_100",
            "fraud_prob_0_100",
            "hybrid_risk_score",
            "hybrid_risk_band",
        ]
    )
    # Add state / age if present
    for c in ["state", "age", "gender"]:
        if c in df.columns and c not in compact_cols:
            compact_cols.append(c)

    compact_df = df[compact_cols].copy()
    compact_df.to_csv(HYBRID_COMPACT_FILE, index=False)
    print_success(f"Compact hybrid predictions saved: {HYBRID_COMPACT_FILE}")

    # Top 100 high-risk
    top_100 = df.sort_values("hybrid_risk_score", ascending=False).head(100)
    top_100.to_csv(TOP_100_FILE, index=False)
    print_success(f"Top 100 high-risk records saved: {TOP_100_FILE}")

    return compact_df, top_100

def generate_report(df):
    """Generate text report summarizing hybrid ensemble behaviour."""
    print_section("STEP 7: GENERATING REPORT")

    report = []
    report.append("=" * 80)
    report.append("UIDAI DATA HACKATHON 2026 - HYBRID MODEL REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("Script: 06_hybrid_model.py")
    report.append("\n" + "=" * 80)

    report.append("\n1. DATASET SUMMARY")
    report.append("─" * 80)
    report.append(f"Records: {len(df):,}")
    report.append(
        f"Hybrid Risk Score Range: {df['hybrid_risk_score'].min():.2f} – "
        f"{df['hybrid_risk_score'].max():.2f}"
    )
    report.append(
        f"Mean: {df['hybrid_risk_score'].mean():.2f} | "
        f"Std: {df['hybrid_risk_score'].std():.2f}"
    )

    report.append("\n2. ENSEMBLE WEIGHTS")
    report.append("─" * 80)
    report.append(f"Isolation Forest (unsupervised) : {WEIGHT_IFOREST:0.3f}")
    report.append(f"Autoencoder (unsupervised)      : {WEIGHT_AE:0.3f}")
    report.append(
        f"XGBoost + Random Forest (supervised) : {WEIGHT_XGB + WEIGHT_RF:0.3f}"
    )
    report.append(
        "Rationale: supervised models anchor precision, unsupervised models "
        "increase sensitivity to novel patterns.[web:29]"
    )

    report.append("\n3. RISK BAND DISTRIBUTION")
    report.append("─" * 80)
    counts = Counter(df["hybrid_risk_band"])
    for band in ["Critical", "Very High", "High", "Medium", "Low", "Very Low"]:
        if band in counts:
            n = counts[band]
            pct = (n / len(df)) * 100
            report.append(f"  {band:10s}: {n:7,} ({pct:5.2f}%)")

    report.append("\n4. INTERPRETATION GUIDELINES")
    report.append("─" * 80)
    report.append("  • 90–100  (Critical): Auto-block or mandatory manual review.[web:56][web:61]")
    report.append("  • 75–89   (Very High): Strong review, enhanced KYC/OTP.[web:56][web:63]")
    report.append("  • 55–74   (High): Queue for manual review, limit activity.[web:59]")
    report.append("  • 35–54   (Medium): Step-up authentication, monitor closely.[web:56]")
    report.append("  • 15–34   (Low): Auto-approve but log for monitoring.[web:63][web:66]")
    report.append("  •   0–14  (Very Low): Normal traffic, minimal friction.[web:59]")

    report.append("\n5. RECOMMENDED NEXT ACTIONS")
    report.append("─" * 80)
    report.append("  1. Integrate hybrid_risk_score into dashboards and case tools.")
    report.append("  2. Calibrate band thresholds with domain experts.")
    report.append("  3. Periodically retrain models as fraud patterns evolve.[web:31]")
    report.append("  4. Backtest hybrid vs single-model performance on labelled data.[web:55]")

    report.append("\n" + "=" * 80)
    report.append("END OF HYBRID MODEL REPORT")
    report.append("=" * 80)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print_success(f"Report saved: {REPORT_FILE}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("UIDAI DATA HACKATHON 2026 - HYBRID ENSEMBLE MODEL")

    start_time = datetime.now()

    # 1. Load and merge inputs
    df_merged, join_cols = load_inputs()

    # 2. Normalize component scores
    df_norm = normalize_scores(df_merged)

    # 3. Compute hybrid risk score
    df_hybrid = compute_hybrid_score(df_norm)

    # 4. Assign final risk bands
    df_hybrid = apply_risk_bands(df_hybrid)

    # 5. Visualizations
    create_visualizations(df_hybrid)

    # 6. Save outputs
    compact_df, top_100 = save_outputs(df_hybrid, join_cols)

    # 7. Report
    generate_report(df_hybrid)

    # Summary
    print_section("EXECUTION SUMMARY")
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Start Time : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time   : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration   : {duration:.2f} seconds ({duration/60:.1f} minutes)")
    print(f"Records    : {len(df_hybrid):,}")
    print(
        f"Critical+Very High: "
        f"{((df_hybrid['hybrid_risk_band'].isin(['Critical','Very High']))).sum():,}"
    )

    print_header("✅ HYBRID MODEL COMPLETE!")
    print_info("Next: 07_visualization.py for dashboards or 08_report_generation.py for final reports.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
