#!/usr/bin/env python3
"""
============================================================================
UIDAI DATA HACKATHON 2026 - ANOMALY DETECTION
============================================================================
File: 04_anomaly_detection.py
Purpose: Detect anomalies using Isolation Forest and Autoencoder (Deep Learning)
Author: Generated with AI Assistance
Date: January 2026
============================================================================
This script:
1. Loads feature-engineered data from 03_preprocessing.py
2. Trains Isolation Forest (unsupervised anomaly detection)
3. Trains Autoencoder Neural Network (reconstruction-based anomaly detection)
4. Combines both models for robust anomaly scoring
5. Identifies top anomalous records
6. Saves models and anomaly scores
7. Generates comprehensive visualizations and reports
============================================================================
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import utility functions (create this file if it doesn't exist)
try:
    from utils import get_numeric_features, clean_numeric_data, validate_numeric_data
except ImportError:
    # If utils.py doesn't exist, define functions here
    def get_numeric_features(df, exclude_patterns=None):
        if exclude_patterns is None:
            exclude_patterns = ['id', 'ID', '_id', 'fraud', 'anomaly', 'risk', 'label', 'target', 'prediction']
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = []
        for col in numeric_cols:
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern.lower() in col.lower():
                    should_exclude = True
                    break
            if not should_exclude:
                feature_cols.append(col)
        return feature_cols


    def clean_numeric_data(X):
        import numpy as np
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        X = X.fillna(0)
        X = X.astype(float)
        return X


    def validate_numeric_data(X, name="Data"):
        pass

import warnings

warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN

import tensorflow as tf

tf.get_logger().setLevel('ERROR')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import joblib

# Machine Learning
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Deep Learning
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'outputs' / 'data'
MODELS_DIR = PROJECT_ROOT / 'outputs' / 'models'
VIZ_DIR = PROJECT_ROOT / 'outputs' / 'visualizations'
REPORTS_DIR = PROJECT_ROOT / 'outputs' / 'reports'

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# File paths
INPUT_FILE = DATA_DIR / '03_features_engineered.csv'
OUTPUT_FILE = DATA_DIR / '04_anomaly_scores.csv'
ISO_FOREST_MODEL = MODELS_DIR / 'isolation_forest_model.pkl'
AUTOENCODER_MODEL = MODELS_DIR / 'autoencoder_model.h5'
REPORT_FILE = REPORTS_DIR / '04_anomaly_detection_report.txt'

# ✅ FIX: Add missing configuration variables
N_ESTIMATORS = 100
RANDOM_STATE = 42
CONTAMINATION = 0.05  # Expected proportion of anomalies (5%)
ENCODING_DIM = 10  # Autoencoder bottleneck dimension
EPOCHS = 50
BATCH_SIZE = 256


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
    """Print success message"""
    print(f"✅ {text}")


def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")


def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")


def print_error(text):
    """Print error message"""
    print(f"❌ {text}")


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_data():
    """Load preprocessed data"""
    print_section("STEP 1: LOADING DATA")

    try:
        df = pd.read_csv(INPUT_FILE)
        print_success(f"Loaded {len(df):,} records with {df.shape[1]} features")
        return df
    except FileNotFoundError:
        print_warning(f"File not found: {INPUT_FILE}")
        print_info("Please run 03_preprocessing.py first")
        sys.exit(1)
    except Exception as e:
        print_warning(f"Error loading data: {e}")
        sys.exit(1)


def prepare_features(df):
    """
    Prepare numeric features for anomaly detection
    Automatically excludes string columns, IDs, and target variables
    """
    print_section("STEP 2: PREPARING FEATURES")

    print(f"Original data shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")

    # ============================================================================
    # STEP 1: Get all numeric columns
    # ============================================================================

    print("\nStep 1: Identifying numeric columns...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Found {len(numeric_cols)} numeric columns")

    # Show non-numeric columns (for debugging)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_numeric_cols) > 0:
        print(f"  Excluded {len(non_numeric_cols)} non-numeric columns:")
        for col in non_numeric_cols[:10]:  # Show first 10
            print(f"    - {col} (type: {df[col].dtype})")
        if len(non_numeric_cols) > 10:
            print(f"    ... and {len(non_numeric_cols) - 10} more")

    # ============================================================================
    # STEP 2: Exclude ID columns and target variables
    # ============================================================================

    print("\nStep 2: Filtering out ID and target columns...")

    # Patterns to exclude from column names
    exclude_patterns = [
        # ID columns
        'id', 'ID', 'Id', '_id', '_ID',
        'enrolment_id', 'enrollment_id',
        'operator_id', 'officer_id', 'user_id',
        'uid', 'UID',

        # Target/label columns (if they exist from previous runs)
        'fraud', 'anomaly', 'risk',
        'label', 'target', 'class',

        # Previous model outputs
        'iso_forest', 'isolation',
        'autoencoder', 'ae_score',
        'combined', 'hybrid',
        'xgboost', 'xgb',
        'random_forest', 'rf_',
        'prediction', 'predicted',
        'probability', 'prob'
    ]

    # Filter columns
    feature_cols = []
    excluded_cols = []

    for col in numeric_cols:
        # Check if column name contains any exclude pattern
        should_exclude = False
        col_lower = col.lower()

        for pattern in exclude_patterns:
            if pattern.lower() in col_lower:
                should_exclude = True
                excluded_cols.append(col)
                break

        if not should_exclude:
            feature_cols.append(col)

    print(f"  Kept: {len(feature_cols)} feature columns")
    if len(excluded_cols) > 0:
        print(f"  Excluded: {len(excluded_cols)} columns")
        for col in excluded_cols[:5]:
            print(f"    - {col}")
        if len(excluded_cols) > 5:
            print(f"    ... and {len(excluded_cols) - 5} more")

    # ============================================================================
    # STEP 3: Safety check - need minimum features
    # ============================================================================

    print("\nStep 3: Validating feature count...")

    if len(feature_cols) < 3:
        print_warning(f"Only {len(feature_cols)} features available!")
        print_warning("Adding engineered features to reach minimum...")

        # Add basic engineered features if too few
        engineered = []

        # Age features
        if 'age' in df.columns and pd.api.types.is_numeric_dtype(df['age']):
            if 'age_squared' not in df.columns:
                df['age_squared'] = df['age'] ** 2
                feature_cols.append('age_squared')
                engineered.append('age_squared')

            if 'age_log' not in df.columns:
                df['age_log'] = np.log1p(df['age'])  # log(1+age) to avoid log(0)
                feature_cols.append('age_log')
                engineered.append('age_log')

        # Biometric quality features
        if 'biometric_quality' in df.columns and pd.api.types.is_numeric_dtype(df['biometric_quality']):
            if 'quality_squared' not in df.columns:
                df['quality_squared'] = df['biometric_quality'] ** 2
                feature_cols.append('quality_squared')
                engineered.append('quality_squared')

            if 'quality_log' not in df.columns:
                df['quality_log'] = np.log1p(df['biometric_quality'])
                feature_cols.append('quality_log')
                engineered.append('quality_log')

        # Row-wise statistics (if we have at least 2 numeric columns)
        if len(numeric_cols) >= 2:
            temp_df = df[numeric_cols].select_dtypes(include=[np.number])

            if 'row_mean' not in df.columns:
                df['row_mean'] = temp_df.mean(axis=1)
                feature_cols.append('row_mean')
                engineered.append('row_mean')

            if 'row_std' not in df.columns:
                df['row_std'] = temp_df.std(axis=1)
                feature_cols.append('row_std')
                engineered.append('row_std')

            if 'row_min' not in df.columns:
                df['row_min'] = temp_df.min(axis=1)
                feature_cols.append('row_min')
                engineered.append('row_min')

            if 'row_max' not in df.columns:
                df['row_max'] = temp_df.max(axis=1)
                feature_cols.append('row_max')
                engineered.append('row_max')

        if len(engineered) > 0:
            print(f"  ✓ Added {len(engineered)} engineered features")
            for feat in engineered:
                print(f"    - {feat}")

    print(f"\nFinal feature count: {len(feature_cols)}")

    # ============================================================================
    # STEP 4: Extract feature matrix
    # ============================================================================

    print("\nStep 4: Extracting feature matrix...")
    X = df[feature_cols].copy()
    print(f"  Initial shape: {X.shape[0]:,} rows × {X.shape[1]} columns")

    # ============================================================================
    # STEP 5: Handle missing values
    # ============================================================================

    print("\nStep 5: Handling missing values...")

    # Replace infinity with NaN
    inf_count = np.isinf(X.values).sum()
    if inf_count > 0:
        print(f"  Found {inf_count} infinity values → replacing with NaN")
        X = X.replace([np.inf, -np.inf], np.nan)

    # Count and fill missing values
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"  Found {missing_count} missing values")
        print(f"  Strategy: Fill with column median")

        # Fill with median
        X = X.fillna(X.median())

        # If entire column was NaN, fill with 0
        still_missing = X.isnull().sum().sum()
        if still_missing > 0:
            print(f"  Some columns still have NaN (were entirely empty)")
            print(f"  Filling remaining with 0")
            X = X.fillna(0)
    else:
        print(f"  ✓ No missing values found")

    # ============================================================================
    # STEP 6: Ensure all values are numeric
    # ============================================================================

    print("\nStep 6: Converting to numeric...")

    # Convert all columns to float64
    for col in X.columns:
        try:
            X[col] = X[col].astype(float)
        except Exception as e:
            print_warning(f"Could not convert column '{col}' to float: {e}")
            print_warning(f"Dropping column '{col}'")
            X = X.drop(columns=[col])

    print(f"  ✓ All columns converted to float64")

    # ============================================================================
    # STEP 7: Final validation
    # ============================================================================

    print("\nStep 7: Final validation...")

    # Check for remaining issues
    validation_passed = True

    # Check 1: No NaN
    if X.isnull().any().any():
        print_error("Still have NaN values!")
        nan_cols = X.columns[X.isnull().any()].tolist()
        print(f"  Columns with NaN: {nan_cols}")
        validation_passed = False
    else:
        print("  ✓ No NaN values")

    # Check 2: No infinity
    if np.isinf(X.values).any():
        print_error("Still have infinity values!")
        inf_cols = X.columns[np.isinf(X.values).any(axis=0)].tolist()
        print(f"  Columns with infinity: {inf_cols}")
        validation_passed = False
    else:
        print("  ✓ No infinity values")

    # Check 3: All numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_numeric) > 0:
        print_error("Still have non-numeric columns!")
        print(f"  Non-numeric columns: {non_numeric}")
        validation_passed = False
    else:
        print("  ✓ All columns are numeric")

    # Check 4: Sufficient features
    if X.shape[1] < 3:
        print_error(f"Only {X.shape[1]} features - need at least 3!")
        validation_passed = False
    else:
        print(f"  ✓ Have {X.shape[1]} features")

    # Check 5: Sufficient samples
    if X.shape[0] < 100:
        print_warning(f"Only {X.shape[0]} samples - results may not be reliable")
    else:
        print(f"  ✓ Have {X.shape[0]:,} samples")

    if not validation_passed:
        print_error("Validation failed! Cannot proceed with anomaly detection.")
        print("\nDebug info:")
        print(X.info())
        sys.exit(1)

    # ============================================================================
    # STEP 8: Show statistics
    # ============================================================================

    print("\n" + "=" * 70)
    print("FEATURE MATRIX SUMMARY")
    print("=" * 70)
    print(f"Shape:           {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"Data type:       {X.values.dtype}")
    print(f"Memory usage:    {X.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    print(f"\nStatistics:")
    print(f"  Min value:     {X.min().min():.4f}")
    print(f"  Max value:     {X.max().max():.4f}")
    print(f"  Mean:          {X.mean().mean():.4f}")
    print(f"  Std Dev:       {X.std().mean():.4f}")
    print(f"\nFeatures: {list(X.columns[:10])}")
    if X.shape[1] > 10:
        print(f"         ... and {X.shape[1] - 10} more")
    print("=" * 70)

    print_success(f"Feature matrix ready for anomaly detection!")

    return X


# ============================================================================
# ISOLATION FOREST
# ============================================================================

def train_isolation_forest(X):
    """Train Isolation Forest for anomaly detection"""
    print_section("STEP 3: TRAINING ISOLATION FOREST")

    print(f"Training on {len(X):,} samples with {X.shape[1]} features...")
    print(f"Contamination rate: {CONTAMINATION} ({CONTAMINATION * 100}% expected anomalies)")

    # Initialize model
    iso_forest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_samples='auto',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    # Train
    print("Training Isolation Forest...")
    start_time = time.time()
    iso_forest.fit(X)
    training_time = time.time() - start_time

    print_success(f"Training completed in {training_time:.2f} seconds")

    # Get anomaly scores
    print("Calculating anomaly scores...")

    # Get decision scores (negative scores = anomalies)
    decision_scores = iso_forest.decision_function(X)

    # Convert to 0-100 scale (higher = more anomalous)
    min_score = np.percentile(decision_scores, 1)
    max_score = np.percentile(decision_scores, 99)

    # Invert and scale to 0-100
    iso_scores = 100 - ((decision_scores - min_score) / (max_score - min_score) * 100)
    iso_scores = np.clip(iso_scores, 0, 100)

    # Statistics
    high_anomalies = (iso_scores > 80).sum()
    mean_score = iso_scores.mean()

    print(f"\nIsolation Forest Results:")
    print(f"  Mean anomaly score: {mean_score:.2f}")
    print(f"  High anomalies (>80): {high_anomalies:,} ({high_anomalies / len(X) * 100:.2f}%)")
    print(f"  Score range: {iso_scores.min():.2f} - {iso_scores.max():.2f}")

    # Save model
    iso_model_path = MODELS_DIR / 'isolation_forest_model.pkl'
    joblib.dump(iso_forest, iso_model_path)
    print_success(f"Model saved: {iso_model_path}")

    # ✅ FIX: Return only 2 values (model, scores)
    return iso_forest, iso_scores


# ============================================================================
# AUTOENCODER
# ============================================================================

def train_autoencoder(X):
    """Train autoencoder neural network for anomaly detection"""
    print_section("STEP 4: TRAINING AUTOENCODER NEURAL NETWORK")

    print(f"Training on {len(X):,} samples with {X.shape[1]} features...")

    # Scale features to 0-1 range
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Architecture
    input_dim = X.shape[1]
    encoding_dim = ENCODING_DIM

    print(f"\nAutoencoder Architecture:")
    print(f"  Input layer:  {input_dim} features")
    print(f"  Hidden layer: 64 neurons (ReLU)")
    print(f"  Hidden layer: 32 neurons (ReLU)")
    print(f"  Bottleneck:   {encoding_dim} neurons (ReLU)")
    print(f"  Hidden layer: 32 neurons (ReLU)")
    print(f"  Hidden layer: 64 neurons (ReLU)")
    print(f"  Output layer: {input_dim} features (Linear)")

    # Build model
    model = Sequential([
        # Encoder
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(encoding_dim, activation='relu', name='bottleneck'),

        # Decoder
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(input_dim, activation='linear')
    ])

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print("\nModel compiled successfully")
    print(f"Total parameters: {model.count_params():,}")

    # Train
    print(f"\nTraining autoencoder...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")

    early_stop = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )

    start_time = time.time()

    history = model.fit(
        X_scaled, X_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0  # Silent training
    )

    training_time = time.time() - start_time

    final_loss = history.history['loss'][-1]
    print_success(f"Training completed in {training_time:.2f} seconds")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Epochs completed: {len(history.history['loss'])}")

    # Calculate reconstruction errors (anomaly scores)
    print("\nCalculating reconstruction errors...")
    predictions = model.predict(X_scaled, verbose=0)

    # MSE per sample
    mse = np.mean(np.square(X_scaled - predictions), axis=1)

    # Convert to 0-100 scale
    min_mse = np.percentile(mse, 1)
    max_mse = np.percentile(mse, 99)

    ae_scores = ((mse - min_mse) / (max_mse - min_mse) * 100)
    ae_scores = np.clip(ae_scores, 0, 100)

    # Statistics
    high_anomalies = (ae_scores > 80).sum()
    mean_score = ae_scores.mean()

    print(f"\nAutoencoder Results:")
    print(f"  Mean anomaly score: {mean_score:.2f}")
    print(f"  High anomalies (>80): {high_anomalies:,} ({high_anomalies / len(X) * 100:.2f}%)")
    print(f"  Score range: {ae_scores.min():.2f} - {ae_scores.max():.2f}")

    # Save model
    ae_model_path = MODELS_DIR / 'autoencoder_model.h5'
    model.save(ae_model_path)
    print_success(f"Model saved: {ae_model_path}")

    # ✅ FIX: Return 3 values (model, scores, history)
    return model, ae_scores, history


# ============================================================================
# COMBINE SCORES
# ============================================================================

def combine_scores(iso_scores, ae_scores):
    """Combine Isolation Forest and Autoencoder scores"""
    print_section("STEP 5: COMBINING ANOMALY SCORES")

    print("Combining two methods:")
    print("  1. Isolation Forest (tree-based)")
    print("  2. Autoencoder (neural network-based)")

    # Weighted average (equal weights)
    combined_scores = (iso_scores * 0.5) + (ae_scores * 0.5)

    print(f"\nCombined scores range: [{combined_scores.min():.2f}, {combined_scores.max():.2f}]")
    print(f"Mean combined score: {combined_scores.mean():.2f}")
    print(f"Std combined score: {combined_scores.std():.2f}")

    print_success("Anomaly scores combined successfully")

    return combined_scores


def categorize_anomalies(df, combined_scores):
    """Categorize anomaly risk"""
    df['anomaly_risk'] = pd.cut(combined_scores,
                                bins=[0, 15, 35, 55, 75, 90, 100],
                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Critical'])

    # Print distribution
    from collections import Counter
    risk_counts = Counter(df['anomaly_risk'])

    print("\nAnomaly Risk Distribution:")
    for risk in ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Critical']:
        if risk in risk_counts:
            count = risk_counts[risk]
            pct = (count / len(combined_scores)) * 100
            print(f"  {risk:10s}: {count:6,} ({pct:5.2f}%)")

    return df


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(df, iso_scores, ae_scores, combined_scores):
    """Save anomaly detection results"""
    print_section("STEP 6: SAVING RESULTS")

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print_success(f"Results saved: {OUTPUT_FILE}")
    print_info(f"Columns: iso_forest_score, autoencoder_score, combined_anomaly_score, anomaly_risk")

    # Save top 100 anomalies
    top_100 = df.nlargest(100, 'combined_anomaly_score')
    top_100_file = REPORTS_DIR / 'top_100_anomalies.csv'
    top_100.to_csv(top_100_file, index=False)
    print_success(f"Top 100 anomalies saved: {top_100_file}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(df, iso_scores, ae_scores, combined_scores, history):
    """Create comprehensive anomaly detection visualizations"""
    print_section("STEP 7: CREATING VISUALIZATIONS")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Anomaly Detection Analysis - Isolation Forest + Autoencoder',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Isolation Forest score distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(iso_scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(80, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Isolation Forest Score Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Autoencoder score distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(ae_scores, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(80, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
    ax2.set_xlabel('Anomaly Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Autoencoder Score Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Combined score distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(combined_scores, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(80, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
    ax3.set_xlabel('Combined Anomaly Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Combined Score Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Scatter: Isolation Forest vs Autoencoder
    ax4 = fig.add_subplot(gs[1, 0])
    sample_size = min(10000, len(df))
    sample_indices = np.random.choice(len(df), sample_size, replace=False)

    scatter = ax4.scatter(iso_scores[sample_indices], ae_scores[sample_indices],
                          alpha=0.4, s=10, c=combined_scores[sample_indices],
                          cmap='Reds', edgecolors='none')
    ax4.set_xlabel('Isolation Forest Score')
    ax4.set_ylabel('Autoencoder Score')
    ax4.set_title('Model Score Comparison', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Combined Score')

    # 5. Anomaly risk categories
    ax5 = fig.add_subplot(gs[1, 1])

    risk_counts = df['anomaly_risk'].value_counts()
    risk_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Critical']
    risk_counts = risk_counts.reindex(risk_order, fill_value=0)

    colors_map = {
        'Very Low': '#32CD32',
        'Low': '#90EE90',
        'Medium': '#FFD700',
        'High': '#FF8C00',
        'Very High': '#FF4500',
        'Critical': '#8B0000'
    }
    colors = [colors_map[risk] for risk in risk_order]

    ax5.bar(range(len(risk_counts)), risk_counts.values, color=colors)
    ax5.set_xticks(range(len(risk_counts)))
    ax5.set_xticklabels(risk_order, rotation=45, ha='right')
    ax5.set_ylabel('Count')
    ax5.set_title('Anomaly Risk Distribution', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(risk_counts.values):
        ax5.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)

    # 6. Autoencoder training history
    ax6 = fig.add_subplot(gs[1, 2])
    if history is not None:
        ax6.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
        if 'val_loss' in history.history:
            ax6.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Loss')
        ax6.set_title('Autoencoder Training History', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Training history not available',
                 ha='center', va='center', transform=ax6.transAxes)
        ax6.axis('off')

    # 7. Box plot by risk category
    ax7 = fig.add_subplot(gs[2, 0])

    box_data = []
    for risk in risk_order:
        risk_mask = df['anomaly_risk'] == risk
        risk_scores = combined_scores[risk_mask]
        box_data.append(risk_scores)

    bp = ax7.boxplot(box_data, labels=risk_order, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax7.set_xticklabels(risk_order, rotation=45, ha='right')
    ax7.set_ylabel('Combined Score')
    ax7.set_title('Score Distribution by Risk Category', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Top anomalies scatter
    ax8 = fig.add_subplot(gs[2, 1])

    top_anomalies = df.nlargest(1000, 'combined_anomaly_score')

    ax8.scatter(range(len(top_anomalies)),
                top_anomalies['combined_anomaly_score'].values,
                alpha=0.6, s=20, color='darkred')
    ax8.set_xlabel('Rank')
    ax8.set_ylabel('Anomaly Score')
    ax8.set_title('Top 1000 Anomalies', fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # 9. Score percentiles
    ax9 = fig.add_subplot(gs[2, 2])
    percentiles = np.arange(0, 101, 10)
    score_percentiles = np.percentile(combined_scores, percentiles)

    ax9.plot(percentiles, score_percentiles, marker='o', linewidth=2,
             markersize=8, color='darkgreen')
    ax9.fill_between(percentiles, score_percentiles, alpha=0.3, color='lightgreen')
    ax9.set_xlabel('Percentile')
    ax9.set_ylabel('Anomaly Score')
    ax9.set_title('Score Percentiles', fontweight='bold')
    ax9.grid(True, alpha=0.3)

    for p, s in zip(percentiles[::2], score_percentiles[::2]):
        ax9.text(p, s, f'{s:.1f}', fontsize=8, ha='center', va='bottom')

    # Save figure
    output_path = VIZ_DIR / '02_anomaly_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print_success(f"Saved: {output_path}")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(df, iso_scores, ae_scores, combined_scores):
    """Generate comprehensive anomaly detection report"""
    print_section("STEP 8: GENERATING REPORT")

    report = []
    report.append("=" * 80)
    report.append("UIDAI DATA HACKATHON 2026 - ANOMALY DETECTION REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Script: 04_anomaly_detection.py")
    report.append("\n" + "=" * 80)

    # Results Summary
    report.append("\nDETECTION RESULTS")
    report.append("─" * 80)
    report.append(f"Total Records Analyzed: {len(df):,}")
    report.append(f"\nIsolation Forest:")
    report.append(f"  - Mean Score: {iso_scores.mean():.2f}")
    report.append(f"  - Std Score: {iso_scores.std():.2f}")
    report.append(f"  - High Anomalies (>80): {(iso_scores > 80).sum():,}")

    report.append(f"\nAutoencoder:")
    report.append(f"  - Mean Score: {ae_scores.mean():.2f}")
    report.append(f"  - Std Score: {ae_scores.std():.2f}")
    report.append(f"  - High Anomalies (>80): {(ae_scores > 80).sum():,}")

    report.append(f"\nCombined:")
    report.append(f"  - Mean Score: {combined_scores.mean():.2f}")
    report.append(f"  - Std Score: {combined_scores.std():.2f}")
    report.append(f"  - High Anomalies (>80): {(combined_scores > 80).sum():,}")

    # Risk Distribution
    report.append("\nRISK DISTRIBUTION")
    report.append("─" * 80)
    from collections import Counter
    risk_counts = Counter(df['anomaly_risk'])
    for risk in ['Critical', 'Very High', 'High', 'Medium', 'Low', 'Very Low']:
        if risk in risk_counts:
            count = risk_counts[risk]
            pct = (count / len(df)) * 100
            report.append(f"  {risk:10s}: {count:6,} ({pct:5.2f}%)")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
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
    """Main anomaly detection function"""

    print_header("UIDAI DATA HACKATHON 2026 - ANOMALY DETECTION")
    start_time = datetime.now()

    try:
        # Step 1: Load data
        df = load_data()

        # Step 2: Prepare features
        X = prepare_features(df)

        # Step 3: Train Isolation Forest
        iso_model, iso_scores = train_isolation_forest(X)

        # Step 4: Train Autoencoder
        ae_model, ae_scores, history = train_autoencoder(X)

        # Step 5: Combine scores
        combined_scores = combine_scores(iso_scores, ae_scores)

        # Step 6: Add scores to dataframe
        df['iso_forest_score'] = iso_scores
        df['autoencoder_score'] = ae_scores
        df['combined_anomaly_score'] = combined_scores

        # Step 7: Categorize anomalies
        df = categorize_anomalies(df, combined_scores)

        # Step 8: Save results
        save_results(df, iso_scores, ae_scores, combined_scores)

        # Step 9: Create visualizations
        create_visualizations(df, iso_scores, ae_scores, combined_scores, history)

        # Step 10: Generate report
        generate_report(df, iso_scores, ae_scores, combined_scores)

        # Summary
        print_section("EXECUTION SUMMARY")
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration:.2f} seconds ({duration / 60:.1f} minutes)")

        high_anomalies = (combined_scores > 80).sum()
        print(f"\nDataset: {len(df):,} records")
        print(f"High Anomalies (>80): {high_anomalies:,} ({high_anomalies / len(df) * 100:.2f}%)")

        print_header("✅ ANOMALY DETECTION COMPLETE!")

    except Exception as e:
        print_error(f"Fatal error in anomaly detection: {str(e)}")
        print("\n" + "=" * 80)
        print("DEBUGGING INFORMATION")
        print("=" * 80)
        print("Possible solutions:")
        print("  1. Run: python quick_clean.py")
        print("  2. Run: python code/03_preprocessing.py")
        print("  3. Check if data has numeric features")
        print("  4. Check for string columns in data")
        print("\nFull error traceback:")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
