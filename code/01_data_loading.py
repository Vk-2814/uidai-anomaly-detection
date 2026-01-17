#!/usr/bin/env python3
"""
============================================================================
UIDAI DATA HACKATHON 2026 - DATA LOADING & INITIAL EXPLORATION
============================================================================
File: 01_data_loading.py
Purpose: Load UIDAI datasets, perform initial exploration, and save processed data
Author: Generated with AI Assistance
Date: January 2026
============================================================================
This script:
1. Loads UIDAI Aadhaar enrolment and update datasets
2. Performs initial data quality checks
3. Displays basic statistics and information
4. Identifies data issues (missing values, duplicates, outliers)
5. Saves loaded data to outputs folder
6. Generates initial data report
============================================================================
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'data'
REPORTS_DIR = PROJECT_ROOT / 'outputs' / 'reports'
LOGS_DIR = PROJECT_ROOT / 'outputs' / 'logs'

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# File paths
ENROLMENT_FILE = DATA_DIR / 'aadhaar_enrolment.csv'
UPDATES_FILE = DATA_DIR / 'aadhaar_updates.csv'

# Output file
OUTPUT_FILE = OUTPUT_DIR / '01_loaded_data.csv'
REPORT_FILE = REPORTS_DIR / '01_data_loading_report.txt'
LOG_FILE = LOGS_DIR / 'execution_log.txt'


# ============================================================================
# LOGGING SETUP
# ============================================================================

def log_message(message, level="INFO"):
    """Write message to log file and print to console"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] [{level}] {message}"

    # Print to console
    print(log_entry)

    # Write to log file
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")


# ============================================================================
# PRINT FORMATTING FUNCTIONS
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
    """Print success message"""
    print(f"✅ {text}")


def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")


def print_error(text):
    """Print error message"""
    print(f"❌ {text}")


def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def check_file_exists(file_path):
    """Check if data file exists"""
    if file_path.exists():
        file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
        print_success(f"Found: {file_path.name} ({file_size:.2f} MB)")
        return True
    else:
        print_error(f"NOT FOUND: {file_path}")
        return False


def load_csv_file(file_path, description="data"):
    """Load CSV file with error handling"""
    print_info(f"Loading {description}...")

    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                print_success(f"Loaded {description} with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue

        # If all encodings fail, try with error handling
        df = pd.read_csv(file_path, encoding='utf-8', errors='replace', low_memory=False)
        print_warning(f"Loaded {description} with character replacement")
        return df

    except FileNotFoundError:
        print_error(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print_error(f"File is empty: {file_path}")
        return None
    except Exception as e:
        print_error(f"Error loading {description}: {str(e)}")
        return None


def create_sample_data():
    """Create sample UIDAI data for testing if real data not available"""
    print_warning("Creating sample data for testing purposes...")

    np.random.seed(42)
    n_samples = 100000  # 100K sample records

    # Generate sample enrolment data
    df = pd.DataFrame({
        'enrolment_id': [f'UID{str(i).zfill(12)}' for i in range(1, n_samples + 1)],
        'enrolment_date': pd.date_range('2020-01-01', periods=n_samples, freq='5min'),
        'state': np.random.choice([
            'Maharashtra', 'Uttar Pradesh', 'Bihar', 'West Bengal', 'Madhya Pradesh',
            'Tamil Nadu', 'Rajasthan', 'Karnataka', 'Gujarat', 'Andhra Pradesh',
            'Odisha', 'Telangana', 'Kerala', 'Jharkhand', 'Assam', 'Punjab',
            'Chhattisgarh', 'Haryana', 'Delhi', 'Jammu and Kashmir'
        ], n_samples),
        'district': [f'District_{i % 100}' for i in range(n_samples)],
        'age': np.random.randint(1, 100, n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.52, 0.47, 0.01]),
        'biometric_quality': np.random.randint(30, 100, n_samples),
        'enrolment_agency': np.random.choice([f'AGENCY_{i}' for i in range(1, 21)], n_samples),
        'operator_id': [f'OP{str(i).zfill(6)}' for i in np.random.randint(1, 1000, n_samples)],
        'iris_captured': np.random.choice([True, False], n_samples, p=[0.95, 0.05]),
        'fingerprint_captured': np.random.choice([True, False], n_samples, p=[0.98, 0.02]),
        'photo_captured': np.random.choice([True, False], n_samples, p=[0.99, 0.01]),
    })

    # Add some intentional issues for testing
    # 1. Missing values (2% random)
    mask = np.random.random(n_samples) < 0.02
    df.loc[mask, 'biometric_quality'] = np.nan

    # 2. Duplicates (0.5%)
    n_duplicates = int(n_samples * 0.005)
    duplicate_indices = np.random.choice(df.index, n_duplicates, replace=False)
    df = pd.concat([df, df.loc[duplicate_indices]], ignore_index=True)

    # 3. Outliers
    outlier_mask = np.random.random(len(df)) < 0.01
    df.loc[outlier_mask, 'age'] = np.random.choice([0, 150, 200], outlier_mask.sum())

    # 4. Low quality biometrics (fraud indicator)
    fraud_mask = np.random.random(len(df)) < 0.05
    df.loc[fraud_mask, 'biometric_quality'] = np.random.randint(0, 40, fraud_mask.sum())

    print_success(f"Created sample dataset with {len(df):,} records")
    print_warning("Remember: This is SAMPLE DATA. Replace with real UIDAI data!")

    return df


# ============================================================================
# DATA EXPLORATION FUNCTIONS
# ============================================================================

def display_basic_info(df, name="Dataset"):
    """Display basic information about the dataset"""
    print_section(f"BASIC INFORMATION: {name}")

    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    print("\nColumn Names and Data Types:")
    print("─" * 60)
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        print(f"  {col:30s} → {str(dtype):15s} ({non_null:,} non-null, {null_pct:.1f}% null)")

    print("\nFirst 3 Rows:")
    print("─" * 60)
    print(df.head(3).to_string())

    print("\nLast 3 Rows:")
    print("─" * 60)
    print(df.tail(3).to_string())


def check_data_quality(df):
    """Perform comprehensive data quality checks"""
    print_section("DATA QUALITY ASSESSMENT")

    issues_found = []

    # 1. Missing Values
    print("\n1. MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    has_missing = False
    for col in df.columns:
        if missing[col] > 0:
            has_missing = True
            print(f"   ⚠️  {col:30s}: {missing[col]:,} ({missing_pct[col]:.2f}%)")
            issues_found.append(f"Missing values in {col}")

    if not has_missing:
        print_success("   No missing values found!")

    # 2. Duplicate Rows
    print("\n2. DUPLICATE ROWS:")
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        dup_pct = (n_duplicates / len(df)) * 100
        print_warning(f"   Found {n_duplicates:,} duplicate rows ({dup_pct:.2f}%)")
        issues_found.append(f"{n_duplicates} duplicate rows")
    else:
        print_success("   No duplicate rows found!")

    # 3. Data Type Consistency
    print("\n3. DATA TYPES:")
    print(f"   Numeric columns: {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"   Categorical columns: {df.select_dtypes(include=['object']).shape[1]}")
    print(f"   Datetime columns: {df.select_dtypes(include=['datetime64']).shape[1]}")

    # 4. Value Ranges (for numeric columns)
    print("\n4. NUMERIC COLUMN RANGES:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:  # Show first 5 numeric columns
        col_min = df[col].min()
        col_max = df[col].max()
        col_mean = df[col].mean()
        print(f"   {col:30s}: [{col_min:.2f}, {col_max:.2f}] (mean: {col_mean:.2f})")

    # 5. Outliers Detection (Simple IQR method)
    print("\n5. POTENTIAL OUTLIERS (IQR method):")
    for col in numeric_cols[:3]:  # Check first 3 numeric columns
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        if len(outliers) > 0:
            outlier_pct = (len(outliers) / len(df)) * 100
            print(f"   ⚠️  {col:30s}: {len(outliers):,} outliers ({outlier_pct:.2f}%)")
            issues_found.append(f"Outliers in {col}")

    return issues_found


def calculate_statistics(df):
    """Calculate and display statistical summary"""
    print_section("STATISTICAL SUMMARY")

    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("\nNUMERIC COLUMNS:")
        print("─" * 80)
        print(df[numeric_cols].describe().to_string())

    # Categorical columns summary
    print("\n\nCATEGORICAL COLUMNS (Top 5 values):")
    print("─" * 80)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:5]:  # First 5 categorical columns
        print(f"\n{col}:")
        value_counts = df[col].value_counts().head(5)
        for idx, (value, count) in enumerate(value_counts.items(), 1):
            pct = (count / len(df)) * 100
            print(f"  {idx}. {value:30s}: {count:,} ({pct:.2f}%)")


def analyze_datetime_columns(df):
    """Analyze datetime columns if present"""
    datetime_cols = df.select_dtypes(include=['datetime64']).columns

    if len(datetime_cols) > 0:
        print_section("DATETIME ANALYSIS")

        for col in datetime_cols:
            print(f"\n{col}:")
            print(f"  Earliest: {df[col].min()}")
            print(f"  Latest: {df[col].max()}")
            print(f"  Range: {(df[col].max() - df[col].min()).days} days")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(df, issues_found):
    """Generate comprehensive data loading report"""
    print_section("GENERATING REPORT")

    report = []
    report.append("=" * 80)
    report.append("UIDAI DATA HACKATHON 2026 - DATA LOADING REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Script: 01_data_loading.py")
    report.append("\n" + "=" * 80)

    # Dataset Overview
    report.append("\n1. DATASET OVERVIEW")
    report.append("─" * 80)
    report.append(f"Total Records: {len(df):,}")
    report.append(f"Total Columns: {df.shape[1]}")
    report.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    # Column Information
    report.append("\n2. COLUMN INFORMATION")
    report.append("─" * 80)
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        report.append(f"  {col:30s} → {str(dtype):15s} ({null_pct:.1f}% null)")

    # Data Quality Issues
    report.append("\n3. DATA QUALITY ISSUES")
    report.append("─" * 80)
    if issues_found:
        for issue in issues_found:
            report.append(f"  ⚠️  {issue}")
    else:
        report.append("  ✅ No major issues found!")

    # Statistical Summary
    report.append("\n4. STATISTICAL SUMMARY")
    report.append("─" * 80)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report.append("\n" + df[numeric_cols].describe().to_string())

    # Next Steps
    report.append("\n5. NEXT STEPS")
    report.append("─" * 80)
    report.append("  1. Run 02_eda.py for exploratory data analysis")
    report.append("  2. Run 03_preprocessing.py for data cleaning")
    report.append("  3. Review missing values and decide handling strategy")
    report.append("  4. Check for anomalies and outliers")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Save to file
    try:
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print_success(f"Report saved: {REPORT_FILE}")
    except Exception as e:
        print_warning(f"Could not save report: {e}")

    return report


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main data loading function"""

    print_header("UIDAI DATA HACKATHON 2026 - DATA LOADING")
    log_message("=" * 80)
    log_message("Starting data loading process")
    log_message("=" * 80)

    start_time = datetime.now()

    # Step 1: Check if data files exist
    print_section("STEP 1: CHECKING DATA FILES")

    enrolment_exists = check_file_exists(ENROLMENT_FILE)

    # Step 2: Load data
    print_section("STEP 2: LOADING DATA")

    if enrolment_exists:
        df = load_csv_file(ENROLMENT_FILE, "UIDAI Enrolment Data")
    else:
        print_warning("Real data not found. Creating sample data...")
        print_info("To use real data:")
        print_info(f"  1. Download UIDAI data from data.gov.in")
        print_info(f"  2. Place in: {DATA_DIR}")
        print_info(f"  3. Name it: aadhaar_enrolment.csv")
        df = create_sample_data()

    if df is None:
        print_error("Failed to load data. Exiting.")
        log_message("Failed to load data", "ERROR")
        sys.exit(1)

    log_message(f"Loaded dataset with {len(df):,} records", "SUCCESS")

    # Step 3: Basic Information
    display_basic_info(df, "UIDAI Enrolment Data")

    # Step 4: Data Quality Check
    issues_found = check_data_quality(df)

    # Step 5: Statistical Summary
    calculate_statistics(df)

    # Step 6: Datetime Analysis
    analyze_datetime_columns(df)

    # Step 7: Save loaded data
    print_section("STEP 3: SAVING PROCESSED DATA")

    try:
        df.to_csv(OUTPUT_FILE, index=False)
        print_success(f"Saved: {OUTPUT_FILE}")
        print_info(f"File size: {OUTPUT_FILE.stat().st_size / 1024 ** 2:.2f} MB")
        log_message(f"Data saved to {OUTPUT_FILE}", "SUCCESS")
    except Exception as e:
        print_error(f"Failed to save data: {e}")
        log_message(f"Failed to save data: {e}", "ERROR")

    # Step 8: Generate Report
    report = generate_report(df, issues_found)

    # Step 9: Summary
    print_section("EXECUTION SUMMARY")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"\nRecords Loaded: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Issues Found: {len(issues_found)}")

    print_header("✅ DATA LOADING COMPLETE!")
    print_info("Next step: Run 02_eda.py for exploratory data analysis")

    log_message("=" * 80)
    log_message(f"Data loading completed in {duration:.2f} seconds", "SUCCESS")
    log_message("=" * 80)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user.")
        log_message("Script interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        log_message(f"Unexpected error: {str(e)}", "ERROR")
        import traceback

        traceback.print_exc()
        log_message(traceback.format_exc(), "ERROR")
        sys.exit(1)

