#!/usr/bin/env python3
"""
============================================================================
UIDAI DATA HACKATHON 2026 - EXPLORATORY DATA ANALYSIS (EDA)
============================================================================
File: 02_eda.py
Purpose: Comprehensive exploratory data analysis with statistical tests and visualizations
Author: Generated with AI Assistance
Date: January 2026
============================================================================
This script:
1. Loads preprocessed data from 01_data_loading.py
2. Performs univariate, bivariate, and multivariate analysis
3. Generates statistical summaries and hypothesis tests
4. Creates comprehensive visualizations (10+ charts)
5. Identifies patterns, trends, and anomalies
6. Saves all visualizations and analysis reports
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
from scipy import stats

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'outputs' / 'data'
VIZ_DIR = PROJECT_ROOT / 'outputs' / 'visualizations'
REPORTS_DIR = PROJECT_ROOT / 'outputs' / 'reports'

# Create output directories
VIZ_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Input and output files
INPUT_FILE = DATA_DIR / '01_loaded_data.csv'
OUTPUT_FILE = REPORTS_DIR / '02_eda_report.txt'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_section(text):
    """Print section header"""
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
    """Load data from previous step"""
    print_section("LOADING DATA")

    try:
        df = pd.read_csv(INPUT_FILE)
        print_success(f"Loaded {len(df):,} records with {df.shape[1]} columns")

        # Convert enrolment_date to datetime if it exists
        if 'enrolment_date' in df.columns:
            df['enrolment_date'] = pd.to_datetime(df['enrolment_date'], errors='coerce')
            print_success("Converted enrolment_date to datetime")

        return df
    except FileNotFoundError:
        print_warning(f"File not found: {INPUT_FILE}")
        print_info("Please run 01_data_loading.py first")
        sys.exit(1)
    except Exception as e:
        print_warning(f"Error loading data: {e}")
        sys.exit(1)


# ============================================================================
# UNIVARIATE ANALYSIS
# ============================================================================

def analyze_numeric_distributions(df):
    """Analyze distribution of numeric variables"""
    print_section("UNIVARIATE ANALYSIS: NUMERIC VARIABLES")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        print_warning("No numeric columns found")
        return

    print(f"Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}\n")

    results = []

    for col in numeric_cols:
        print(f"Analyzing: {col}")

        # Basic statistics
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std()
        skew_val = df[col].skew()
        kurt_val = df[col].kurtosis()

        print(f"  Mean: {mean_val:.2f}")
        print(f"  Median: {median_val:.2f}")
        print(f"  Std Dev: {std_val:.2f}")
        print(f"  Skewness: {skew_val:.2f} ({'Right-skewed' if skew_val > 0 else 'Left-skewed'})")
        print(f"  Kurtosis: {kurt_val:.2f}")

        # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for large)
        if len(df[col].dropna()) < 5000:
            try:
                stat, p_value = stats.shapiro(df[col].dropna().sample(min(5000, len(df))))
                print(f"  Normality (Shapiro-Wilk p-value): {p_value:.4f}")
                is_normal = "YES" if p_value > 0.05 else "NO"
                print(f"  Is Normal? {is_normal}")
            except:
                print("  Normality test: Could not compute")

        results.append({
            'column': col,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'skewness': skew_val,
            'kurtosis': kurt_val
        })
        print()

    return pd.DataFrame(results)


def analyze_categorical_distributions(df):
    """Analyze distribution of categorical variables"""
    print_section("UNIVARIATE ANALYSIS: CATEGORICAL VARIABLES")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(categorical_cols) == 0:
        print_warning("No categorical columns found")
        return

    print(f"Found {len(categorical_cols)} categorical columns: {', '.join(categorical_cols)}\n")

    results = []

    for col in categorical_cols[:10]:  # Analyze first 10 to avoid clutter
        print(f"Analyzing: {col}")

        n_unique = df[col].nunique()
        most_common = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
        most_common_pct = (df[col].value_counts().iloc[0] / len(df)) * 100 if len(df[col].value_counts()) > 0 else 0

        print(f"  Unique values: {n_unique}")
        print(f"  Most common: {most_common} ({most_common_pct:.2f}%)")

        if n_unique <= 20:  # Show distribution for low-cardinality columns
            print(f"  Distribution:")
            for value, count in df[col].value_counts().head(5).items():
                pct = (count / len(df)) * 100
                print(f"    - {value}: {count:,} ({pct:.2f}%)")

        results.append({
            'column': col,
            'unique_values': n_unique,
            'most_common': most_common,
            'most_common_pct': most_common_pct
        })
        print()

    return pd.DataFrame(results)


# ============================================================================
# BIVARIATE ANALYSIS
# ============================================================================

def analyze_correlations(df):
    """Analyze correlations between numeric variables"""
    print_section("BIVARIATE ANALYSIS: CORRELATIONS")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        print_warning("Need at least 2 numeric columns for correlation analysis")
        return None

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    print("Correlation Matrix:")
    print(corr_matrix.to_string())

    # Find strong correlations
    print("\nStrong Correlations (|r| > 0.5):")
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]

            if abs(corr_val) > 0.5:
                print(f"  {col1} ↔ {col2}: {corr_val:.3f}")
                strong_corr.append((col1, col2, corr_val))

    if not strong_corr:
        print("  No strong correlations found")

    return corr_matrix


def analyze_categorical_relationships(df):
    """Analyze relationships between categorical and numeric variables"""
    print_section("BIVARIATE ANALYSIS: CATEGORICAL vs NUMERIC")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not categorical_cols or not numeric_cols:
        print_warning("Need both categorical and numeric columns")
        return

    # Analyze first categorical column with first numeric column as example
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]

        if df[cat_col].nunique() <= 10:  # Only if reasonable number of categories
            print(f"Analyzing: {num_col} by {cat_col}\n")

            # Group statistics
            grouped = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std', 'count'])
            print(grouped.to_string())

            # ANOVA test if more than 2 groups
            if df[cat_col].nunique() > 2:
                groups = [df[df[cat_col] == cat][num_col].dropna() for cat in df[cat_col].unique()]
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    print(f"\nANOVA Test:")
                    print(f"  F-statistic: {f_stat:.4f}")
                    print(f"  p-value: {p_value:.4f}")
                    print(f"  Significant difference? {'YES' if p_value < 0.05 else 'NO'}")
                except:
                    print("\nANOVA Test: Could not compute")


# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================

def analyze_temporal_patterns(df):
    """Analyze temporal patterns if datetime column exists"""
    print_section("TEMPORAL ANALYSIS")

    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    if not datetime_cols:
        print_info("No datetime columns found. Skipping temporal analysis.")
        return

    date_col = datetime_cols[0]
    print(f"Analyzing temporal patterns in: {date_col}\n")

    # Extract temporal features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['hour'] = df[date_col].dt.hour

    # Monthly distribution
    print("Records by Month:")
    monthly_counts = df['month'].value_counts().sort_index()
    for month, count in monthly_counts.items():
        pct = (count / len(df)) * 100
        print(f"  Month {month:2d}: {count:,} ({pct:.2f}%)")

    # Day of week distribution
    print("\nRecords by Day of Week:")
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = df['day_of_week'].value_counts().sort_index()
    for dow, count in dow_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {dow_names[dow]:9s}: {count:,} ({pct:.2f}%)")

    # Hourly distribution (if hour varies)
    if df['hour'].nunique() > 1:
        print("\nRecords by Hour:")
        hour_counts = df['hour'].value_counts().sort_index().head(10)
        for hour, count in hour_counts.items():
            pct = (count / len(df)) * 100
            print(f"  Hour {hour:2d}: {count:,} ({pct:.2f}%)")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_distribution_plots(df):
    """Create distribution plots for numeric variables"""
    print_section("CREATING DISTRIBUTION VISUALIZATIONS")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        print_warning("No numeric columns to visualize")
        return

    # Create subplot for each numeric column (max 6)
    n_cols = min(len(numeric_cols), 6)
    n_rows = (n_cols + 1) // 2

    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]

    for idx, col in enumerate(numeric_cols[:6]):
        ax = axes[idx]

        # Histogram with KDE
        df[col].dropna().hist(bins=50, ax=ax, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        # Add mean line
        mean_val = df[col].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.legend()

    # Remove empty subplots
    for idx in range(n_cols, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    output_path = VIZ_DIR / '01_eda_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print_success(f"Saved: {output_path}")


def create_categorical_plots(df):
    """Create bar plots for categorical variables"""
    print_section("CREATING CATEGORICAL VISUALIZATIONS")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_cols:
        print_warning("No categorical columns to visualize")
        return

    # Plot first 4 categorical columns with reasonable cardinality
    plot_cols = [col for col in categorical_cols if df[col].nunique() <= 20][:4]

    if not plot_cols:
        print_warning("No categorical columns with reasonable cardinality")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, col in enumerate(plot_cols):
        ax = axes[idx]

        # Get top 10 values
        value_counts = df[col].value_counts().head(10)

        # Horizontal bar chart
        value_counts.plot(kind='barh', ax=ax, color='coral')
        ax.set_title(f'Top 10 Values in {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Count')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, v in enumerate(value_counts.values):
            ax.text(v, i, f' {v:,}', va='center')

    # Remove empty subplots
    for idx in range(len(plot_cols), 4):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    output_path = VIZ_DIR / '02_eda_categorical.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print_success(f"Saved: {output_path}")


def create_correlation_heatmap(corr_matrix):
    """Create correlation heatmap"""
    print_section("CREATING CORRELATION HEATMAP")

    if corr_matrix is None or corr_matrix.empty:
        print_warning("No correlation matrix to visualize")
        return

    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})

    plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    output_path = VIZ_DIR / '03_eda_correlation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print_success(f"Saved: {output_path}")


def create_boxplots(df):
    """Create box plots for numeric variables"""
    print_section("CREATING BOX PLOTS")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        print_warning("No numeric columns for box plots")
        return

    # Create box plots for first 4 numeric columns
    plot_cols = numeric_cols[:4]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, col in enumerate(plot_cols):
        ax = axes[idx]

        # Box plot
        df[col].dropna().plot(kind='box', ax=ax, vert=False, color='lightblue')
        ax.set_title(f'Box Plot: {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for idx in range(len(plot_cols), 4):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    output_path = VIZ_DIR / '04_eda_boxplots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print_success(f"Saved: {output_path}")


def create_temporal_plots(df):
    """Create temporal analysis plots"""
    print_section("CREATING TEMPORAL VISUALIZATIONS")

    if 'month' not in df.columns:
        print_info("No temporal features available. Skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Monthly trend
    monthly_counts = df['month'].value_counts().sort_index()
    axes[0, 0].plot(monthly_counts.index, monthly_counts.values, marker='o',
                    linewidth=2, markersize=8, color='steelblue')
    axes[0, 0].set_title('Records by Month', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)

    # Day of week distribution
    if 'day_of_week' in df.columns:
        dow_counts = df['day_of_week'].value_counts().sort_index()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 1].bar(range(len(dow_counts)), dow_counts.values, color='coral')
        axes[0, 1].set_title('Records by Day of Week', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(range(len(dow_names)))
        axes[0, 1].set_xticklabels(dow_names)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Hourly distribution
    if 'hour' in df.columns and df['hour'].nunique() > 1:
        hour_counts = df['hour'].value_counts().sort_index()
        axes[1, 0].bar(hour_counts.index, hour_counts.values, color='lightgreen')
        axes[1, 0].set_title('Records by Hour', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 0].text(0.5, 0.5, 'Hour data not available',
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')

    # Year distribution (if available)
    if 'year' in df.columns and df['year'].nunique() > 1:
        year_counts = df['year'].value_counts().sort_index()
        axes[1, 1].bar(year_counts.index.astype(str), year_counts.values, color='mediumpurple')
        axes[1, 1].set_title('Records by Year', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 1].text(0.5, 0.5, 'Multiple years not available',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')

    plt.tight_layout()
    output_path = VIZ_DIR / '05_eda_temporal.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print_success(f"Saved: {output_path}")


def create_overview_dashboard(df):
    """Create comprehensive overview dashboard"""
    print_section("CREATING OVERVIEW DASHBOARD")

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('UIDAI Data - Exploratory Data Analysis Overview',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Dataset summary (text)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    summary_text = f"""
    DATASET SUMMARY
    ────────────────────
    Total Records: {len(df):,}
    Total Columns: {df.shape[1]}
    Memory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB

    Numeric Cols: {len(df.select_dtypes(include=[np.number]).columns)}
    Categorical Cols: {len(df.select_dtypes(include=['object']).columns)}

    Missing Values: {df.isnull().sum().sum():,}
    Duplicates: {df.duplicated().sum():,}
    """
    ax1.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', va='center')

    # 2. Missing values chart
    ax2 = fig.add_subplot(gs[0, 1:])
    missing = df.isnull().sum().sort_values(ascending=False).head(10)
    if missing.sum() > 0:
        missing.plot(kind='barh', ax=ax2, color='orange')
        ax2.set_title('Top 10 Columns with Missing Values', fontweight='bold')
        ax2.set_xlabel('Count')
    else:
        ax2.text(0.5, 0.5, 'No Missing Values!', ha='center', va='center',
                 fontsize=14, transform=ax2.transAxes)
        ax2.axis('off')

    # 3-4. Numeric distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:2]
    for idx, col in enumerate(numeric_cols):
        ax = fig.add_subplot(gs[1, idx])
        df[col].dropna().hist(bins=30, ax=ax, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title(f'Distribution: {col}', fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    # 5. Categorical distribution
    ax5 = fig.add_subplot(gs[1, 2])
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        col = categorical_cols[0]
        value_counts = df[col].value_counts().head(5)
        value_counts.plot(kind='barh', ax=ax5, color='coral')
        ax5.set_title(f'Top 5: {col}', fontweight='bold')

    # 6. Data types pie chart
    ax6 = fig.add_subplot(gs[2, 0])
    dtype_counts = df.dtypes.value_counts()
    ax6.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%',
            startangle=90, colors=plt.cm.Set3.colors)
    ax6.set_title('Data Types Distribution', fontweight='bold')

    # 7. Record count over time (if available)
    ax7 = fig.add_subplot(gs[2, 1:])
    if 'enrolment_date' in df.columns:
        df['enrolment_date'].dt.to_period('D').value_counts().sort_index().head(30).plot(
            ax=ax7, color='green', linewidth=2)
        ax7.set_title('Records Over Time (First 30 Days)', fontweight='bold')
        ax7.set_ylabel('Count')
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Temporal data not available',
                 ha='center', va='center', transform=ax7.transAxes)
        ax7.axis('off')

    output_path = VIZ_DIR / '01_eda_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print_success(f"Saved: {output_path}")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_eda_report(df, numeric_stats, categorical_stats, corr_matrix):
    """Generate comprehensive EDA report"""
    print_section("GENERATING EDA REPORT")

    report = []
    report.append("=" * 80)
    report.append("UIDAI DATA HACKATHON 2026 - EXPLORATORY DATA ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Script: 02_eda.py")
    report.append("\n" + "=" * 80)

    # Dataset Summary
    report.append("\n1. DATASET SUMMARY")
    report.append("─" * 80)
    report.append(f"Total Records: {len(df):,}")
    report.append(f"Total Columns: {df.shape[1]}")
    report.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    report.append(f"Missing Values: {df.isnull().sum().sum():,}")
    report.append(f"Duplicate Rows: {df.duplicated().sum():,}")

    # Numeric Statistics
    if numeric_stats is not None and not numeric_stats.empty:
        report.append("\n2. NUMERIC VARIABLES SUMMARY")
        report.append("─" * 80)
        report.append(numeric_stats.to_string())

    # Categorical Statistics
    if categorical_stats is not None and not categorical_stats.empty:
        report.append("\n3. CATEGORICAL VARIABLES SUMMARY")
        report.append("─" * 80)
        report.append(categorical_stats.to_string())

    # Correlations
    if corr_matrix is not None:
        report.append("\n4. CORRELATION MATRIX")
        report.append("─" * 80)
        report.append(corr_matrix.to_string())

    # Key Findings
    report.append("\n5. KEY FINDINGS")
    report.append("─" * 80)
    report.append("  • Data exploration completed successfully")
    report.append("  • Multiple visualizations generated")
    report.append("  • Statistical summaries computed")
    report.append("  • Ready for preprocessing and feature engineering")

    # Next Steps
    report.append("\n6. NEXT STEPS")
    report.append("─" * 80)
    report.append("  1. Run 03_preprocessing.py for data cleaning")
    report.append("  2. Handle missing values and outliers")
    report.append("  3. Engineer features for machine learning")

    report.append("\n" + "=" * 80)
    report.append("END OF EDA REPORT")
    report.append("=" * 80)

    # Save report
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print_success(f"Report saved: {OUTPUT_FILE}")
    except Exception as e:
        print_warning(f"Could not save report: {e}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main EDA function"""

    print_header("UIDAI DATA HACKATHON 2026 - EXPLORATORY DATA ANALYSIS")
    start_time = datetime.now()

    # Load data
    df = load_data()

    # Univariate Analysis
    numeric_stats = analyze_numeric_distributions(df)
    categorical_stats = analyze_categorical_distributions(df)

    # Bivariate Analysis
    corr_matrix = analyze_correlations(df)
    analyze_categorical_relationships(df)

    # Temporal Analysis
    analyze_temporal_patterns(df)

    # Create Visualizations
    print_header("CREATING VISUALIZATIONS")
    create_overview_dashboard(df)
    create_distribution_plots(df)
    create_categorical_plots(df)
    create_correlation_heatmap(corr_matrix)
    create_boxplots(df)
    create_temporal_plots(df)

    # Generate Report
    generate_eda_report(df, numeric_stats, categorical_stats, corr_matrix)

    # Summary
    print_section("EXECUTION SUMMARY")
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"\nVisualizations Created: 6+")
    print(f"Report Generated: {OUTPUT_FILE}")

    print_header("✅ EDA COMPLETE!")
    print_info("Next step: Run 03_preprocessing.py for data cleaning and feature engineering")


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
