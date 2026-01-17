#!/usr/bin/env python3
"""
============================================================================
UIDAI DATA HACKATHON 2026 - ADVANCED VISUALIZATION & DASHBOARDS
============================================================================
File: 07_visualization.py
Purpose: Create comprehensive visualizations, charts, and interactive dashboards
Author: Generated with AI Assistance
Date: January 2026
============================================================================
This script:
1. Loads final hybrid results from 06_hybrid_model.py
2. Creates 10+ advanced visualizations:
   - Geographic heatmaps (state-wise risk distribution)
   - Demographic analysis (age, gender patterns)
   - Temporal patterns (time-based trends)
   - Risk correlation matrices
   - Feature importance comparisons
   - Model performance comparison
   - Interactive HTML dashboard
3. Saves all visualizations as high-quality PNG files
4. Generates interactive Plotly HTML dashboard
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'outputs' / 'data'
VIZ_DIR = PROJECT_ROOT / 'outputs' / 'visualizations'
REPORTS_DIR = PROJECT_ROOT / 'outputs' / 'reports'

VIZ_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / '06_hybrid_results.csv'
DASHBOARD_FILE = VIZ_DIR / 'dashboard.html'

# Color schemes
RISK_COLORS = {
    'Critical': '#8B0000',  # Dark red
    'Very High': '#FF4500',  # Orange red
    'High': '#FF8C00',  # Dark orange
    'Medium': '#FFD700',  # Gold
    'Low': '#90EE90',  # Light green
    'Very Low': '#32CD32'  # Lime green
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_section(text):
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print('─' * 80)


def print_success(text):
    print(f"✅ {text}")


def print_info(text):
    print(f"ℹ️  {text}")


def print_warning(text):
    print(f"⚠️  {text}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load hybrid results"""
    print_section("LOADING DATA")

    try:
        df = pd.read_csv(INPUT_FILE)
        print_success(f"Loaded {len(df):,} records with {df.shape[1]} columns")

        # Convert date columns if present
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass

        return df
    except FileNotFoundError:
        print_warning(f"File not found: {INPUT_FILE}")
        print_info("Please run 06_hybrid_model.py first")
        sys.exit(1)


# ============================================================================
# VISUALIZATION 1: GEOGRAPHIC HEATMAP
# ============================================================================

def create_geographic_heatmap(df):
    """Create state-wise risk heatmap"""
    print_section("VIZ 1: GEOGRAPHIC HEATMAP")

    # Check if state column exists
    state_cols = [col for col in df.columns if 'state' in col.lower()]
    if not state_cols:
        print_info("State column not found. Skipping geographic heatmap.")
        return

    state_col = state_cols[0]

    # Calculate state-wise metrics
    state_stats = df.groupby(state_col).agg({
        'hybrid_risk_score': ['mean', 'median', 'count'],
        'hybrid_risk_band': lambda x: (x.isin(['Critical', 'Very High'])).sum()
    }).reset_index()

    state_stats.columns = ['state', 'avg_risk', 'median_risk', 'count', 'high_risk_count']
    state_stats['high_risk_pct'] = (state_stats['high_risk_count'] / state_stats['count'] * 100).round(2)
    state_stats = state_stats.sort_values('avg_risk', ascending=False)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Geographic Risk Analysis by State', fontsize=16, fontweight='bold')

    # 1. Average risk score by state (top 15)
    ax1 = axes[0, 0]
    top_states = state_stats.head(15)
    ax1.barh(range(len(top_states)), top_states['avg_risk'], color='coral')
    ax1.set_yticks(range(len(top_states)))
    ax1.set_yticklabels(top_states['state'], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Average Hybrid Risk Score')
    ax1.set_title('Top 15 States by Average Risk Score', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    for i, v in enumerate(top_states['avg_risk'].values):
        ax1.text(v, i, f'{v:.1f}', va='center', fontsize=8)

    # 2. High-risk percentage by state (top 15)
    ax2 = axes[0, 1]
    top_high_risk = state_stats.nlargest(15, 'high_risk_pct')
    ax2.barh(range(len(top_high_risk)), top_high_risk['high_risk_pct'], color='darkred')
    ax2.set_yticks(range(len(top_high_risk)))
    ax2.set_yticklabels(top_high_risk['state'], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('% Critical + Very High Risk')
    ax2.set_title('Top 15 States by High-Risk %', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    for i, v in enumerate(top_high_risk['high_risk_pct'].values):
        ax2.text(v, i, f'{v:.1f}%', va='center', fontsize=8)

    # 3. Record volume by state (top 15)
    ax3 = axes[1, 0]
    top_volume = state_stats.nlargest(15, 'count')
    ax3.bar(range(len(top_volume)), top_volume['count'], color='steelblue')
    ax3.set_xticks(range(len(top_volume)))
    ax3.set_xticklabels(top_volume['state'], rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Record Count')
    ax3.set_title('Top 15 States by Volume', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Risk distribution heatmap
    ax4 = axes[1, 1]

    # Create risk band distribution by state (top 10 states)
    top_10_states = state_stats.head(10)['state'].tolist()
    df_top = df[df[state_col].isin(top_10_states)]

    risk_pivot = pd.crosstab(
        df_top[state_col],
        df_top['hybrid_risk_band'],
        normalize='index'
    ) * 100

    risk_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Critical']
    risk_pivot = risk_pivot.reindex(columns=risk_order, fill_value=0)

    sns.heatmap(risk_pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=ax4, cbar_kws={'label': '% of Records'})
    ax4.set_title('Risk Distribution Heatmap (Top 10 States)', fontweight='bold')
    ax4.set_xlabel('Risk Band')
    ax4.set_ylabel('State')

    plt.tight_layout()
    output_path = VIZ_DIR / '05_geographic_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print_success(f"Saved: {output_path}")


# ============================================================================
# VISUALIZATION 2: DEMOGRAPHIC ANALYSIS
# ============================================================================

def create_demographic_analysis(df):
    """Create demographic pattern analysis"""
    print_section("VIZ 2: DEMOGRAPHIC ANALYSIS")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Demographic Risk Analysis', fontsize=16, fontweight='bold')

    # 1. Age distribution by risk band
    ax1 = axes[0, 0]
    age_cols = [col for col in df.columns if col == 'age' or col.endswith('_age')]
    if age_cols:
        age_col = age_cols[0]
        risk_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Critical']

        for risk in risk_order:
            if risk in df['hybrid_risk_band'].values:
                data = df[df['hybrid_risk_band'] == risk][age_col].dropna()
                if len(data) > 0:
                    ax1.hist(data, bins=20, alpha=0.6, label=risk,
                             color=RISK_COLORS.get(risk, 'gray'))

        ax1.set_xlabel('Age')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Age Distribution by Risk Band', fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Age data not available',
                 ha='center', va='center', transform=ax1.transAxes)
        ax1.axis('off')

    # 2. Average risk by age group
    ax2 = axes[0, 1]
    if age_cols:
        age_col = age_cols[0]
        df['age_group'] = pd.cut(df[age_col], bins=[0, 18, 35, 50, 60, 100],
                                 labels=['0-18', '19-35', '36-50', '51-60', '60+'])

        age_risk = df.groupby('age_group')['hybrid_risk_score'].agg(['mean', 'count']).reset_index()

        ax2.bar(range(len(age_risk)), age_risk['mean'], color='salmon')
        ax2.set_xticks(range(len(age_risk)))
        ax2.set_xticklabels(age_risk['age_group'])
        ax2.set_ylabel('Average Risk Score')
        ax2.set_title('Average Risk by Age Group', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        for i, (mean_val, count_val) in enumerate(zip(age_risk['mean'], age_risk['count'])):
            ax2.text(i, mean_val, f'{mean_val:.1f}\n(n={count_val:,})',
                     ha='center', va='bottom', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'Age data not available',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')

    # 3. Gender analysis
    ax3 = axes[0, 2]
    gender_cols = [col for col in df.columns if 'gender' in col.lower()]
    if gender_cols:
        gender_col = gender_cols[0]
        gender_risk = df.groupby(gender_col)['hybrid_risk_score'].agg(['mean', 'count']).reset_index()

        ax3.bar(range(len(gender_risk)), gender_risk['mean'],
                color=['lightblue', 'lightpink', 'lightgray'][:len(gender_risk)])
        ax3.set_xticks(range(len(gender_risk)))
        ax3.set_xticklabels(gender_risk[gender_col])
        ax3.set_ylabel('Average Risk Score')
        ax3.set_title('Average Risk by Gender', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        for i, (mean_val, count_val) in enumerate(zip(gender_risk['mean'], gender_risk['count'])):
            ax3.text(i, mean_val, f'{mean_val:.1f}\n(n={count_val:,})',
                     ha='center', va='bottom', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'Gender data not available',
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.axis('off')

    # 4. Biometric quality vs risk
    ax4 = axes[1, 0]
    quality_cols = [col for col in df.columns if 'biometric_quality' in col.lower()]
    if quality_cols:
        quality_col = quality_cols[0]

        # Sample for performance
        sample_df = df.sample(min(10000, len(df)), random_state=42)
        scatter = ax4.scatter(sample_df[quality_col], sample_df['hybrid_risk_score'],
                              c=sample_df['hybrid_risk_score'], cmap='Reds',
                              alpha=0.4, s=10)
        ax4.set_xlabel('Biometric Quality')
        ax4.set_ylabel('Hybrid Risk Score')
        ax4.set_title('Biometric Quality vs Risk Score', fontweight='bold')
        plt.colorbar(scatter, ax=ax4, label='Risk Score')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Biometric quality data not available',
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')

    # 5. Risk band by demographic segment
    ax5 = axes[1, 1]
    if age_cols and gender_cols:
        # Create demographic segments
        demo_risk = df.groupby([gender_col, 'age_group'])['hybrid_risk_score'].mean().reset_index()
        demo_pivot = demo_risk.pivot(index='age_group', columns=gender_col, values='hybrid_risk_score')

        demo_pivot.plot(kind='bar', ax=ax5, rot=0)
        ax5.set_xlabel('Age Group')
        ax5.set_ylabel('Average Risk Score')
        ax5.set_title('Risk by Age & Gender', fontweight='bold')
        ax5.legend(title='Gender', fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'Demographic data not available',
                 ha='center', va='center', transform=ax5.transAxes)
        ax5.axis('off')

    # 6. High-risk demographic profile
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Calculate high-risk demographics
    high_risk_df = df[df['hybrid_risk_band'].isin(['Critical', 'Very High'])]

    text_lines = []
    text_lines.append("HIGH-RISK PROFILE")
    text_lines.append("═" * 30)
    text_lines.append(f"Total High-Risk: {len(high_risk_df):,}")
    text_lines.append(f"% of Total: {len(high_risk_df) / len(df) * 100:.2f}%")
    text_lines.append("")

    if age_cols:
        age_col = age_cols[0]
        text_lines.append("Age Distribution:")
        age_dist = high_risk_df['age_group'].value_counts()
        for age_grp, count in age_dist.head(3).items():
            pct = count / len(high_risk_df) * 100
            text_lines.append(f"  {age_grp}: {pct:.1f}%")

    if gender_cols:
        gender_col = gender_cols[0]
        text_lines.append("\nGender Distribution:")
        gender_dist = high_risk_df[gender_col].value_counts()
        for gender, count in gender_dist.items():
            pct = count / len(high_risk_df) * 100
            text_lines.append(f"  {gender}: {pct:.1f}%")

    if quality_cols:
        quality_col = quality_cols[0]
        avg_quality = high_risk_df[quality_col].mean()
        text_lines.append(f"\nAvg Biometric Quality: {avg_quality:.1f}")

    ax6.text(0.05, 0.95, '\n'.join(text_lines),
             fontsize=9, va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = VIZ_DIR / '06_demographic_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print_success(f"Saved: {output_path}")


# ============================================================================
# VISUALIZATION 3: TEMPORAL PATTERNS
# ============================================================================

def create_temporal_analysis(df):
    """Create temporal pattern analysis"""
    print_section("VIZ 3: TEMPORAL PATTERNS")

    # Check for temporal columns
    temporal_cols = ['enrolment_year', 'enrolment_month', 'enrolment_day_of_week', 'enrolment_hour']
    has_temporal = any(col in df.columns for col in temporal_cols)

    if not has_temporal:
        print_info("Temporal columns not found. Skipping temporal analysis.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Temporal Risk Patterns', fontsize=16, fontweight='bold')

    # 1. Risk by month
    ax1 = axes[0, 0]
    if 'enrolment_month' in df.columns:
        monthly_risk = df.groupby('enrolment_month')['hybrid_risk_score'].agg(['mean', 'count']).reset_index()

        ax1.plot(monthly_risk['enrolment_month'], monthly_risk['mean'],
                 marker='o', linewidth=2, markersize=8, color='steelblue')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Average Risk Score')
        ax1.set_title('Average Risk by Month', fontweight='bold')
        ax1.set_xticks(range(1, 13))
        ax1.grid(True, alpha=0.3)

        # Add volume as secondary axis
        ax1_twin = ax1.twinx()
        ax1_twin.bar(monthly_risk['enrolment_month'], monthly_risk['count'],
                     alpha=0.3, color='lightcoral')
        ax1_twin.set_ylabel('Record Count', color='lightcoral')

    # 2. Risk by day of week
    ax2 = axes[0, 1]
    if 'enrolment_day_of_week' in df.columns:
        dow_risk = df.groupby('enrolment_day_of_week')['hybrid_risk_score'].mean().reset_index()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        colors = ['lightgreen' if i < 5 else 'lightcoral' for i in range(7)]
        ax2.bar(range(7), dow_risk['hybrid_risk_score'], color=colors)
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(dow_names)
        ax2.set_ylabel('Average Risk Score')
        ax2.set_title('Average Risk by Day of Week', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    # 3. Risk by hour
    ax3 = axes[1, 0]
    if 'enrolment_hour' in df.columns:
        hourly_risk = df.groupby('enrolment_hour')['hybrid_risk_score'].mean().reset_index()

        ax3.plot(hourly_risk['enrolment_hour'], hourly_risk['hybrid_risk_score'],
                 marker='o', linewidth=2, color='darkgreen')
        ax3.axvspan(0, 6, alpha=0.2, color='red', label='Night (0-6)')
        ax3.axvspan(22, 24, alpha=0.2, color='red')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Average Risk Score')
        ax3.set_title('Average Risk by Hour', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Heatmap: Day of week × Hour
    ax4 = axes[1, 1]
    if 'enrolment_day_of_week' in df.columns and 'enrolment_hour' in df.columns:
        # Sample data for performance
        sample_df = df.sample(min(50000, len(df)), random_state=42)

        heatmap_data = sample_df.groupby(['enrolment_day_of_week', 'enrolment_hour'])[
            'hybrid_risk_score'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='enrolment_day_of_week',
                                           columns='enrolment_hour',
                                           values='hybrid_risk_score')

        sns.heatmap(heatmap_pivot, cmap='RdYlGn_r', annot=False, fmt='.1f',
                    cbar_kws={'label': 'Avg Risk Score'}, ax=ax4)
        ax4.set_yticklabels(dow_names, rotation=0)
        ax4.set_xlabel('Hour')
        ax4.set_ylabel('Day of Week')
        ax4.set_title('Risk Heatmap: Day × Hour', fontweight='bold')

    plt.tight_layout()
    output_path = VIZ_DIR / '07_temporal_patterns.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print_success(f"Saved: {output_path}")


# ============================================================================
# VISUALIZATION 4: MODEL COMPARISON
# ============================================================================

def create_model_comparison(df):
    """Create model performance comparison"""
    print_section("VIZ 4: MODEL COMPARISON")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Model Score Comparison', fontsize=16, fontweight='bold')

    # 1. Score distributions
    ax1 = axes[0, 0]
    scores = ['iso_score_0_100', 'ae_score_0_100', 'fraud_prob_0_100', 'hybrid_risk_score']
    labels = ['Isolation Forest', 'Autoencoder', 'Fraud Probability', 'Hybrid']
    colors = ['steelblue', 'coral', 'darkgreen', 'purple']

    for score, label, color in zip(scores, labels, colors):
        if score in df.columns:
            ax1.hist(df[score], bins=50, alpha=0.5, label=label, color=color)

    ax1.set_xlabel('Score (0-100)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Score Distribution Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Correlation between models
    ax2 = axes[0, 1]
    score_cols = [col for col in scores if col in df.columns]
    if len(score_cols) >= 2:
        corr_matrix = df[score_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, square=True, ax=ax2,
                    xticklabels=labels[:len(score_cols)],
                    yticklabels=labels[:len(score_cols)])
        ax2.set_title('Model Score Correlations', fontweight='bold')

    # 3. Model agreement analysis
    ax3 = axes[1, 0]
    if all(col in df.columns for col in scores[:3]):
        # Define high-risk threshold
        threshold = 70

        df['iso_high'] = (df['iso_score_0_100'] > threshold).astype(int)
        df['ae_high'] = (df['ae_score_0_100'] > threshold).astype(int)
        df['fraud_high'] = (df['fraud_prob_0_100'] > threshold).astype(int)

        df['model_agreement'] = df['iso_high'] + df['ae_high'] + df['fraud_high']

        agreement_counts = df['model_agreement'].value_counts().sort_index()

        ax3.bar(range(len(agreement_counts)), agreement_counts.values,
                color=['green', 'yellow', 'orange', 'red'][:len(agreement_counts)])
        ax3.set_xticks(range(len(agreement_counts)))
        ax3.set_xticklabels([f'{int(idx)} models' for idx in agreement_counts.index])
        ax3.set_ylabel('Count')
        ax3.set_title(f'Model Agreement on High Risk (>  {threshold})', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(agreement_counts.values):
            ax3.text(i, v, f'{v:,}\n({v / len(df) * 100:.1f}%)',
                     ha='center', va='bottom', fontsize=9)

    # 4. Hybrid vs individual models
    ax4 = axes[1, 1]
    if all(col in df.columns for col in scores):
        # Calculate mean absolute difference
        metrics = []
        for score, label in zip(scores[:-1], labels[:-1]):
            diff = np.abs(df[score] - df['hybrid_risk_score']).mean()
            metrics.append({'model': label, 'avg_diff': diff})

        metrics_df = pd.DataFrame(metrics)

        ax4.barh(range(len(metrics_df)), metrics_df['avg_diff'],
                 color=['steelblue', 'coral', 'darkgreen'])
        ax4.set_yticks(range(len(metrics_df)))
        ax4.set_yticklabels(metrics_df['model'])
        ax4.invert_yaxis()
        ax4.set_xlabel('Mean Absolute Difference from Hybrid')
        ax4.set_title('Model Deviation from Hybrid Score', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        for i, v in enumerate(metrics_df['avg_diff'].values):
            ax4.text(v, i, f'{v:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    output_path = VIZ_DIR / '08_confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print_success(f"Saved: {output_path}")


# ============================================================================
# VISUALIZATION 5: FEATURE IMPORTANCE
# ============================================================================

def create_feature_importance_viz(df):
    """Create feature importance visualization"""
    print_section("VIZ 5: FEATURE IMPORTANCE")

    # Calculate correlation with hybrid risk score
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude the target and model scores
    exclude = ['hybrid_risk_score', 'iso_score_0_100', 'ae_score_0_100',
               'fraud_prob_0_100', 'combined_anomaly_0_100']
    feature_cols = [col for col in numeric_cols if col not in exclude][:20]  # Top 20

    if not feature_cols:
        print_info("Not enough features for importance analysis")
        return

    # Calculate correlations
    correlations = []
    for col in feature_cols:
        try:
            corr = df[[col, 'hybrid_risk_score']].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations.append({'feature': col, 'correlation': abs(corr)})
        except:
            pass

    if not correlations:
        print_info("Could not calculate feature correlations")
        return

    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['darkred' if x > 0.5 else 'orange' if x > 0.3 else 'steelblue'
              for x in corr_df['correlation']]

    ax.barh(range(len(corr_df)), corr_df['correlation'], color=colors)
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df['feature'], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Absolute Correlation with Hybrid Risk Score')
    ax.set_title('Top 10 Features by Correlation with Risk Score',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')

    for i, v in enumerate(corr_df['correlation'].values):
        ax.text(v, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    output_path = VIZ_DIR / '10_feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print_success(f"Saved: {output_path}")


# ============================================================================
# INTERACTIVE DASHBOARD (PLOTLY)
# ============================================================================

def create_interactive_dashboard(df):
    """Create interactive HTML dashboard using Plotly"""
    print_section("VIZ 6: INTERACTIVE DASHBOARD")

    # Sample data for performance (max 10k records)
    df_sample = df.sample(min(10000, len(df)), random_state=42)

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Risk Score Distribution',
            'Risk Band Distribution',
            'Geographic Risk Map',
            'Temporal Patterns',
            'Model Score Comparison',
            'Top Risk States'
        ),
        specs=[
            [{'type': 'histogram'}, {'type': 'pie'}],
            [{'type': 'bar'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'bar'}]
        ]
    )

    # 1. Risk Score Distribution
    fig.add_trace(
        go.Histogram(x=df_sample['hybrid_risk_score'], nbinsx=50,
                     name='Hybrid Risk Score', marker_color='purple'),
        row=1, col=1
    )

    # 2. Risk Band Pie Chart
    risk_counts = df['hybrid_risk_band'].value_counts()
    fig.add_trace(
        go.Pie(labels=risk_counts.index, values=risk_counts.values,
               marker=dict(colors=[RISK_COLORS.get(r, 'gray') for r in risk_counts.index])),
        row=1, col=2
    )

    # 3. State-wise risk (if available)
    state_cols = [col for col in df.columns if 'state' in col.lower()]
    if state_cols:
        state_col = state_cols[0]
        state_risk = df.groupby(state_col)['hybrid_risk_score'].mean().sort_values(ascending=False).head(10)

        fig.add_trace(
            go.Bar(x=state_risk.index, y=state_risk.values,
                   marker_color='coral', name='Avg Risk by State'),
            row=2, col=1
        )

    # 4. Model comparison scatter
    if 'fraud_prob_0_100' in df_sample.columns and 'combined_anomaly_0_100' in df_sample.columns:
        fig.add_trace(
            go.Scatter(
                x=df_sample['fraud_prob_0_100'],
                y=df_sample['combined_anomaly_0_100'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df_sample['hybrid_risk_score'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Risk")
                ),
                name='Fraud vs Anomaly'
            ),
            row=2, col=2
        )

    # 5. Time series (if available)
    if 'enrolment_month' in df.columns:
        monthly = df.groupby('enrolment_month')['hybrid_risk_score'].mean()
        fig.add_trace(
            go.Scatter(x=monthly.index, y=monthly.values,
                       mode='lines+markers', name='Monthly Risk',
                       line=dict(color='steelblue', width=3)),
            row=3, col=1
        )

    # 6. Top risk records summary
    if state_cols:
        top_risk_states = df.nlargest(100, 'hybrid_risk_score')[state_col].value_counts().head(10)
        fig.add_trace(
            go.Bar(y=top_risk_states.index, x=top_risk_states.values,
                   orientation='h', marker_color='darkred',
                   name='Top Risk States'),
            row=3, col=2
        )

    # Update layout
    fig.update_layout(
        title_text="UIDAI Fraud & Anomaly Detection - Interactive Dashboard",
        title_font_size=20,
        height=1200,
        showlegend=False
    )

    # Save HTML
    fig.write_html(str(DASHBOARD_FILE))
    print_success(f"Interactive dashboard saved: {DASHBOARD_FILE}")
    print_info("Open in browser to interact with visualizations")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main visualization function"""

    print_header("UIDAI DATA HACKATHON 2026 - ADVANCED VISUALIZATIONS")
    start_time = datetime.now()

    # Load data
    df = load_data()

    # Create visualizations
    create_geographic_heatmap(df)
    create_demographic_analysis(df)
    create_temporal_analysis(df)
    create_model_comparison(df)
    create_feature_importance_viz(df)
    create_interactive_dashboard(df)

    # Summary
    print_section("EXECUTION SUMMARY")
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"\nVisualizations Created: 6 PNG files + 1 HTML dashboard")
    print(f"Output Directory: {VIZ_DIR}")

    print_header("✅ VISUALIZATION COMPLETE!")
    print_info("Next step: Run 08_report_generation.py for final reports")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
