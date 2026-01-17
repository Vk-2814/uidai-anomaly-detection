#!/usr/bin/env python3
"""
============================================================================
UIDAI DATA HACKATHON 2026 - DATA PREPROCESSING & FEATURE ENGINEERING
============================================================================
File: 03_preprocessing.py
Purpose: Clean data, handle missing values, engineer features for ML models
Author: Generated with AI Assistance
Date: January 2026
============================================================================
This script:
1. Loads data from 01_data_loading.py
2. Handles missing values (imputation strategies)
3. Removes duplicates and invalid records
4. Detects and treats outliers
5. Engineers 15+ new features (temporal, demographic, statistical)
6. Normalizes and scales features
7. Prepares data for machine learning models
8. Saves clean, feature-rich dataset
============================================================================
"""

import pandas as pd
import numpy as np
import sys
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'outputs' / 'data'
MODELS_DIR = PROJECT_ROOT / 'outputs' / 'models'
REPORTS_DIR = PROJECT_ROOT / 'outputs' / 'reports'

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Input and output files
INPUT_FILE = DATA_DIR / '01_loaded_data.csv'
OUTPUT_FILE = DATA_DIR / '02_preprocessed_data.csv'
FEATURES_FILE = DATA_DIR / '03_features_engineered.csv'
SCALER_FILE = MODELS_DIR / 'scaler.pkl'
REPORT_FILE = REPORTS_DIR / '03_preprocessing_report.txt'


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
    print_section("STEP 1: LOADING DATA")

    try:
        df = pd.read_csv(INPUT_FILE)
        print_success(f"Loaded {len(df):,} records with {df.shape[1]} columns")

        # Convert date columns
        date_columns = ['enrolment_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print_success(f"Converted {col} to datetime")

        return df
    except FileNotFoundError:
        print_warning(f"File not found: {INPUT_FILE}")
        print_info("Please run 01_data_loading.py first")
        sys.exit(1)
    except Exception as e:
        print_warning(f"Error loading data: {e}")
        sys.exit(1)


# ============================================================================
# DATA CLEANING
# ============================================================================

def handle_missing_values(df):
    """Handle missing values with appropriate strategies"""
    print_section("STEP 2: HANDLING MISSING VALUES")

    initial_missing = df.isnull().sum().sum()
    print(f"Initial missing values: {initial_missing:,}\n")

    # Strategy 1: Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print_success(f"{col}: Filled {missing_count} missing with median ({median_val:.2f})")

    # Strategy 2: Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            print_success(f"{col}: Filled {missing_count} missing with mode ({mode_val})")

    # Strategy 3: Fill boolean columns with False (conservative)
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col].fillna(False, inplace=True)
            print_success(f"{col}: Filled {missing_count} missing with False")

    final_missing = df.isnull().sum().sum()
    print(f"\nFinal missing values: {final_missing:,}")
    print_success(f"Handled {initial_missing - final_missing:,} missing values")

    return df


def remove_duplicates(df):
    """Remove duplicate records"""
    print_section("STEP 3: REMOVING DUPLICATES")

    initial_rows = len(df)
    df = df.drop_duplicates()
    final_rows = len(df)
    removed = initial_rows - final_rows

    if removed > 0:
        print_warning(f"Removed {removed:,} duplicate rows ({removed / initial_rows * 100:.2f}%)")
    else:
        print_success("No duplicate rows found")

    return df


def handle_outliers(df):
    """Detect and handle outliers using IQR method"""
    print_section("STEP 4: HANDLING OUTLIERS")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_handled = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        n_outliers = len(outliers)

        if n_outliers > 0:
            outlier_pct = (n_outliers / len(df)) * 100

            # Strategy: Cap outliers at boundaries instead of removing
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

            print_success(
                f"{col}: Capped {n_outliers:,} outliers ({outlier_pct:.2f}%) to [{lower_bound:.2f}, {upper_bound:.2f}]")
            outliers_handled[col] = n_outliers

    if not outliers_handled:
        print_info("No outliers detected")

    return df


def validate_data_quality(df):
    """Validate data quality after cleaning"""
    print_section("STEP 5: DATA QUALITY VALIDATION")

    issues = []

    # Check 1: Missing values
    missing = df.isnull().sum().sum()
    if missing == 0:
        print_success(f"Missing values: {missing} ✓")
    else:
        print_warning(f"Missing values: {missing}")
        issues.append(f"Still has {missing} missing values")

    # Check 2: Duplicates
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        print_success(f"Duplicate rows: {duplicates} ✓")
    else:
        print_warning(f"Duplicate rows: {duplicates}")
        issues.append(f"Still has {duplicates} duplicates")

    # Check 3: Data types
    print_success(f"Data types: {len(df.dtypes.unique())} types")

    # Check 4: Value ranges for specific columns
    if 'age' in df.columns:
        age_min, age_max = df['age'].min(), df['age'].max()
        if 0 <= age_min and age_max <= 120:
            print_success(f"Age range: [{age_min:.0f}, {age_max:.0f}] ✓")
        else:
            print_warning(f"Age range unusual: [{age_min:.0f}, {age_max:.0f}]")

    if 'biometric_quality' in df.columns:
        qual_min, qual_max = df['biometric_quality'].min(), df['biometric_quality'].max()
        if 0 <= qual_min and qual_max <= 100:
            print_success(f"Quality range: [{qual_min:.0f}, {qual_max:.0f}] ✓")
        else:
            print_warning(f"Quality range unusual: [{qual_min:.0f}, {qual_max:.0f}]")

    if issues:
        print_warning(f"\nFound {len(issues)} quality issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print_success("\n✓ All data quality checks passed!")

    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_temporal_features(df):
    """Create temporal features from datetime columns"""
    print_section("STEP 6: CREATING TEMPORAL FEATURES")

    if 'enrolment_date' not in df.columns:
        print_info("No enrolment_date column. Skipping temporal features.")
        return df

    # Extract temporal components
    df['enrolment_year'] = df['enrolment_date'].dt.year
    df['enrolment_month'] = df['enrolment_date'].dt.month
    df['enrolment_day'] = df['enrolment_date'].dt.day
    df['enrolment_hour'] = df['enrolment_date'].dt.hour
    df['enrolment_day_of_week'] = df['enrolment_date'].dt.dayofweek
    df['enrolment_quarter'] = df['enrolment_date'].dt.quarter

    print_success("Created: year, month, day, hour, day_of_week, quarter")

    # Days since enrolment
    df['days_since_enrolment'] = (pd.Timestamp.now() - df['enrolment_date']).dt.days
    print_success("Created: days_since_enrolment")

    # Temporal flags
    df['is_weekend'] = (df['enrolment_day_of_week'] >= 5).astype(int)
    df['is_unusual_time'] = ((df['enrolment_hour'] < 6) | (df['enrolment_hour'] > 22)).astype(int)
    df['is_business_hours'] = ((df['enrolment_hour'] >= 9) & (df['enrolment_hour'] <= 17)).astype(int)

    print_success("Created: is_weekend, is_unusual_time, is_business_hours")

    # Season (for Northern Hemisphere - India)
    df['season'] = df['enrolment_month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    print_success("Created: season")

    print_info(f"Total temporal features: 12")

    return df


def create_demographic_features(df):
    """Create demographic-based features"""
    print_section("STEP 7: CREATING DEMOGRAPHIC FEATURES")

    features_created = []

    # Age-based features
    if 'age' in df.columns:
        # Age groups
        df['age_group'] = pd.cut(df['age'],
                                 bins=[0, 5, 18, 35, 50, 60, 100],
                                 labels=['Infant', 'Minor', 'Young_Adult', 'Adult', 'Senior', 'Elderly'])
        features_created.append('age_group')

        # Age flags
        df['is_minor'] = (df['age'] < 18).astype(int)
        df['is_senior'] = (df['age'] >= 60).astype(int)
        df['is_working_age'] = ((df['age'] >= 18) & (df['age'] < 60)).astype(int)
        features_created.extend(['is_minor', 'is_senior', 'is_working_age'])

        # Age anomaly (too young or too old)
        df['age_anomaly'] = ((df['age'] < 5) | (df['age'] > 100)).astype(int)
        features_created.append('age_anomaly')

        print_success(
            f"Created age features: {', '.join(['age_group', 'is_minor', 'is_senior', 'is_working_age', 'age_anomaly'])}")

    # Gender encoding (if exists)
    if 'gender' in df.columns:
        df['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
        features_created.append('gender_encoded')
        print_success("Created: gender_encoded")

    print_info(f"Total demographic features: {len(features_created)}")

    return df


def create_biometric_features(df):
    """Create biometric quality-based features"""
    print_section("STEP 8: CREATING BIOMETRIC FEATURES")

    features_created = []

    if 'biometric_quality' in df.columns:
        # Quality categories
        df['quality_category'] = pd.cut(df['biometric_quality'],
                                        bins=[0, 40, 60, 80, 100],
                                        labels=['Poor', 'Fair', 'Good', 'Excellent'])
        features_created.append('quality_category')

        # Quality flags
        df['is_low_quality'] = (df['biometric_quality'] < 50).astype(int)
        df['is_high_quality'] = (df['biometric_quality'] >= 80).astype(int)
        features_created.extend(['is_low_quality', 'is_high_quality'])

        # Quality score normalized (0-1)
        df['quality_normalized'] = df['biometric_quality'] / 100
        features_created.append('quality_normalized')

        print_success(f"Created biometric features: {', '.join(features_created)}")

    # Biometric capture completeness
    biometric_cols = ['iris_captured', 'fingerprint_captured', 'photo_captured']
    if all(col in df.columns for col in biometric_cols):
        df['biometric_completeness'] = df[biometric_cols].sum(axis=1)
        df['all_biometrics_captured'] = (df['biometric_completeness'] == 3).astype(int)
        features_created.extend(['biometric_completeness', 'all_biometrics_captured'])
        print_success("Created: biometric_completeness, all_biometrics_captured")

    print_info(f"Total biometric features: {len(features_created)}")

    return df


def create_geographic_features(df):
    """Create geography-based features"""
    print_section("STEP 9: CREATING GEOGRAPHIC FEATURES")

    features_created = []

    if 'state' in df.columns:
        # State-level aggregations
        state_stats = df.groupby('state').agg({
            'biometric_quality': ['mean', 'std', 'count'] if 'biometric_quality' in df.columns else 'count',
            'age': 'mean' if 'age' in df.columns else 'count'
        }).reset_index()

        # Flatten column names
        if 'biometric_quality' in df.columns:
            state_stats.columns = ['state', 'state_avg_quality', 'state_std_quality', 'state_enrolment_count',
                                   'state_avg_age']

            # Merge back to main dataframe
            df = df.merge(state_stats[['state', 'state_avg_quality', 'state_enrolment_count']],
                          on='state', how='left')

            # Quality deviation from state mean
            df['quality_vs_state_mean'] = df['biometric_quality'] - df['state_avg_quality']

            features_created.extend(['state_avg_quality', 'state_enrolment_count', 'quality_vs_state_mean'])
            print_success("Created: state_avg_quality, state_enrolment_count, quality_vs_state_mean")
        else:
            state_stats.columns = ['state', 'state_enrolment_count']
            df = df.merge(state_stats, on='state', how='left')
            features_created.append('state_enrolment_count')
            print_success("Created: state_enrolment_count")

        # High-volume state flag
        median_count = df['state_enrolment_count'].median()
        df['is_high_volume_state'] = (df['state_enrolment_count'] > median_count).astype(int)
        features_created.append('is_high_volume_state')
        print_success("Created: is_high_volume_state")

    print_info(f"Total geographic features: {len(features_created)}")

    return df


def create_statistical_features(df):
    """Create statistical deviation features"""
    print_section("STEP 10: CREATING STATISTICAL FEATURES")

    features_created = []

    numeric_cols = ['age', 'biometric_quality']

    for col in numeric_cols:
        if col in df.columns:
            # Z-score (standardized score)
            mean_val = df[col].mean()
            std_val = df[col].std()
            df[f'{col}_zscore'] = (df[col] - mean_val) / std_val
            features_created.append(f'{col}_zscore')

            # Percentile rank
            df[f'{col}_percentile'] = df[col].rank(pct=True) * 100
            features_created.append(f'{col}_percentile')

            print_success(f"Created: {col}_zscore, {col}_percentile")

    print_info(f"Total statistical features: {len(features_created)}")

    return df


def create_interaction_features(df):
    """Create interaction features between variables"""
    print_section("STEP 11: CREATING INTERACTION FEATURES")

    features_created = []

    # Age × Quality interaction
    if 'age' in df.columns and 'biometric_quality' in df.columns:
        df['age_quality_interaction'] = df['age'] * df['biometric_quality']
        features_created.append('age_quality_interaction')
        print_success("Created: age_quality_interaction")

    # Senior with low quality (high risk)
    if 'is_senior' in df.columns and 'is_low_quality' in df.columns:
        df['senior_low_quality'] = df['is_senior'] * df['is_low_quality']
        features_created.append('senior_low_quality')
        print_success("Created: senior_low_quality")

    # Weekend + unusual time (suspicious pattern)
    if 'is_weekend' in df.columns and 'is_unusual_time' in df.columns:
        df['weekend_unusual_time'] = df['is_weekend'] * df['is_unusual_time']
        features_created.append('weekend_unusual_time')
        print_success("Created: weekend_unusual_time")

    print_info(f"Total interaction features: {len(features_created)}")

    return df


# ============================================================================
# FEATURE SCALING
# ============================================================================

def scale_features(df):
    """Scale numeric features for ML models"""
    print_section("STEP 12: FEATURE SCALING")

    # Select numeric columns for scaling
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude ID columns and already normalized features
    exclude_cols = ['enrolment_id'] if 'enrolment_id' in numeric_cols else []
    exclude_cols.extend([col for col in numeric_cols if 'normalized' in col or 'encoded' in col])

    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

    if not cols_to_scale:
        print_info("No columns to scale")
        return df, None

    print(f"Scaling {len(cols_to_scale)} features...")

    # Initialize and fit scaler
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # Save scaler for later use
    joblib.dump(scaler, SCALER_FILE)
    print_success(f"Scaler saved: {SCALER_FILE}")

    # Also save feature names
    feature_names = {
        'numeric_features': cols_to_scale,
        'all_features': df.columns.tolist()
    }
    joblib.dump(feature_names, MODELS_DIR / 'feature_names.pkl')
    print_success(f"Feature names saved: {MODELS_DIR / 'feature_names.pkl'}")

    return df, scaler


# ============================================================================
# SUMMARY & REPORTING
# ============================================================================

def generate_preprocessing_summary(df, initial_shape, final_shape):
    """Generate preprocessing summary"""
    print_section("PREPROCESSING SUMMARY")

    print(f"Initial Shape: {initial_shape[0]:,} rows × {initial_shape[1]} columns")
    print(f"Final Shape: {final_shape[0]:,} rows × {final_shape[1]} columns")
    print(f"Rows Removed: {initial_shape[0] - final_shape[0]:,}")
    print(f"Columns Added: {final_shape[1] - initial_shape[1]}")
    print(f"\nFinal Dataset:")
    print(f"  - Records: {len(df):,}")
    print(f"  - Features: {df.shape[1]}")
    print(f"  - Numeric: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  - Categorical: {len(df.select_dtypes(include=['object', 'category']).columns)}")
    print(f"  - Missing: {df.isnull().sum().sum()}")
    print(f"  - Duplicates: {df.duplicated().sum()}")
    print(f"  - Memory: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")


def generate_preprocessing_report(df, initial_shape):
    """Generate comprehensive preprocessing report"""
    print_section("GENERATING REPORT")

    report = []
    report.append("=" * 80)
    report.append("UIDAI DATA HACKATHON 2026 - PREPROCESSING REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Script: 03_preprocessing.py")
    report.append("\n" + "=" * 80)

    # Data Transformation Summary
    report.append("\n1. DATA TRANSFORMATION SUMMARY")
    report.append("─" * 80)
    report.append(f"Initial Records: {initial_shape[0]:,}")
    report.append(f"Initial Columns: {initial_shape[1]}")
    report.append(f"Final Records: {len(df):,}")
    report.append(f"Final Columns: {df.shape[1]}")
    report.append(f"Records Removed: {initial_shape[0] - len(df):,}")
    report.append(f"Columns Added: {df.shape[1] - initial_shape[1]}")

    # Feature Categories
    report.append("\n2. FEATURE CATEGORIES")
    report.append("─" * 80)
    report.append(f"Temporal Features: 12")
    report.append(f"Demographic Features: 6")
    report.append(f"Biometric Features: 6")
    report.append(f"Geographic Features: 4")
    report.append(f"Statistical Features: 4")
    report.append(f"Interaction Features: 3")
    report.append(f"Total New Features: ~35")

    # Data Quality
    report.append("\n3. DATA QUALITY METRICS")
    report.append("─" * 80)
    report.append(f"Missing Values: {df.isnull().sum().sum()}")
    report.append(f"Duplicate Rows: {df.duplicated().sum()}")
    report.append(f"Numeric Columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    report.append(f"Categorical Columns: {len(df.select_dtypes(include=['object']).columns)}")

    # Column Names
    report.append("\n4. ALL COLUMNS")
    report.append("─" * 80)
    for idx, col in enumerate(df.columns, 1):
        report.append(f"  {idx:2d}. {col}")

    # Next Steps
    report.append("\n5. NEXT STEPS")
    report.append("─" * 80)
    report.append("  1. Run 04_anomaly_detection.py (Isolation Forest + Autoencoder)")
    report.append("  2. Run 05_fraud_classification.py (XGBoost + Random Forest)")
    report.append("  3. Run 06_hybrid_model.py (Combine all models)")

    report.append("\n" + "=" * 80)
    report.append("END OF PREPROCESSING REPORT")
    report.append("=" * 80)

    # Save report
    try:
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print_success(f"Report saved: {REPORT_FILE}")
    except Exception as e:
        print_warning(f"Could not save report: {e}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main preprocessing function"""

    print_header("UIDAI DATA HACKATHON 2026 - DATA PREPROCESSING")
    start_time = datetime.now()

    # Load data
    df = load_data()
    initial_shape = df.shape

    # Data Cleaning
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = validate_data_quality(df)

    # Save cleaned data
    print_section("SAVING CLEANED DATA")
    df.to_csv(OUTPUT_FILE, index=False)
    print_success(f"Saved: {OUTPUT_FILE}")

    # Feature Engineering
    print_header("FEATURE ENGINEERING")
    df = create_temporal_features(df)
    df = create_demographic_features(df)
    df = create_biometric_features(df)
    df = create_geographic_features(df)
    df = create_statistical_features(df)
    df = create_interaction_features(df)

    # Feature Scaling
    df, scaler = scale_features(df)

    # Save feature-engineered data
    print_section("SAVING FEATURE-ENGINEERED DATA")
    df.to_csv(FEATURES_FILE, index=False)
    print_success(f"Saved: {FEATURES_FILE}")

    # Generate Summary
    final_shape = df.shape
    generate_preprocessing_summary(df, initial_shape, final_shape)

    # Generate Report
    generate_preprocessing_report(df, initial_shape)

    # Execution Summary
    print_section("EXECUTION SUMMARY")
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"\nInitial: {initial_shape[0]:,} × {initial_shape[1]}")
    print(f"Final: {final_shape[0]:,} × {final_shape[1]}")
    print(f"New Features: {final_shape[1] - initial_shape[1]}")

    print_header("✅ PREPROCESSING COMPLETE!")
    print_info("Next step: Run 04_anomaly_detection.py for anomaly detection")


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

