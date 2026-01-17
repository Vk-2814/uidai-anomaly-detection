#!/usr/bin/env python3
"""
============================================================================
UIDAI DATA HACKATHON 2026 - FRAUD CLASSIFICATION (SUPERVISED LEARNING)
============================================================================
File: 05_fraud_classification.py
Purpose: Train supervised models (XGBoost + Random Forest) for fraud detection
Author: Generated with AI Assistance
Date: January 2026
============================================================================
This script:
1. Loads data with anomaly scores from 04_anomaly_detection.py
2. Creates synthetic fraud labels based on anomaly patterns (for demo)
3. Trains XGBoost Classifier (Gradient Boosting)
4. Trains Random Forest Classifier (Ensemble)
5. Evaluates both models with comprehensive metrics
6. Combines predictions for robust fraud detection
7. Analyzes feature importance
8. Saves models and predictions
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
import joblib
from collections import Counter

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, f1_score,
                             accuracy_score, precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'outputs' / 'data'
MODELS_DIR = PROJECT_ROOT / 'outputs' / 'models'
VIZ_DIR = PROJECT_ROOT / 'outputs' / 'visualizations'
REPORTS_DIR = PROJECT_ROOT / 'outputs' / 'reports'

# Create directories
VIZ_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# File paths
INPUT_FILE = DATA_DIR / '04_anomaly_scores.csv'
OUTPUT_FILE = DATA_DIR / '05_fraud_predictions.csv'
XGBOOST_MODEL = MODELS_DIR / 'xgboost_fraud_model.pkl'
RF_MODEL = MODELS_DIR / 'random_forest_fraud_model.pkl'
REPORT_FILE = REPORTS_DIR / '05_fraud_classification_report.txt'

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS_XGB = 200
N_ESTIMATORS_RF = 100
MAX_DEPTH = 10


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
# DATA LOADING & PREPARATION
# ============================================================================

def load_data():
    """Load data with anomaly scores"""
    print_section("STEP 1: LOADING DATA")

    try:
        df = pd.read_csv(INPUT_FILE)
        print_success(f"Loaded {len(df):,} records with {df.shape[1]} features")

        # Check if anomaly scores exist
        required_cols = ['combined_anomaly_score', 'anomaly_risk']
        if all(col in df.columns for col in required_cols):
            print_success("Anomaly scores found")
        else:
            print_warning("Anomaly scores missing. Please run 04_anomaly_detection.py first")
            sys.exit(1)

        return df
    except FileNotFoundError:
        print_warning(f"File not found: {INPUT_FILE}")
        print_info("Please run 04_anomaly_detection.py first")
        sys.exit(1)
    except Exception as e:
        print_warning(f"Error loading data: {e}")
        sys.exit(1)


def create_fraud_labels(df):
    """
    Create synthetic fraud labels based on multiple risk factors
    In production, you would use real fraud labels from investigations
    """
    print_section("STEP 2: CREATING FRAUD LABELS")

    print_warning("NOTE: Creating synthetic fraud labels for demonstration")
    print_info("In production, use actual fraud investigation results")

    # Initialize fraud probability
    fraud_prob = np.zeros(len(df))

    # Factor 1: High anomaly score (40% weight)
    if 'combined_anomaly_score' in df.columns:
        anomaly_factor = df['combined_anomaly_score'] / 100
        fraud_prob += anomaly_factor * 0.4
        print_success("Factor 1: Anomaly score (40% weight)")

    # Factor 2: Low biometric quality (20% weight)
    if 'is_low_quality' in df.columns:
        fraud_prob += df['is_low_quality'] * 0.2
        print_success("Factor 2: Low biometric quality (20% weight)")
    elif 'biometric_quality' in df.columns:
        quality_factor = 1 - (df['biometric_quality'] / 100)
        fraud_prob += quality_factor * 0.2
        print_success("Factor 2: Biometric quality (20% weight)")

    # Factor 3: Unusual timing (15% weight)
    if 'is_unusual_time' in df.columns:
        fraud_prob += df['is_unusual_time'] * 0.15
        print_success("Factor 3: Unusual timing (15% weight)")

    # Factor 4: Weekend enrollment (10% weight)
    if 'is_weekend' in df.columns:
        fraud_prob += df['is_weekend'] * 0.10
        print_success("Factor 4: Weekend enrollment (10% weight)")

    # Factor 5: Senior with low quality (15% weight)
    if 'senior_low_quality' in df.columns:
        fraud_prob += df['senior_low_quality'] * 0.15
        print_success("Factor 5: Senior + low quality (15% weight)")

    # Add random noise
    fraud_prob += np.random.normal(0, 0.05, len(df))
    fraud_prob = np.clip(fraud_prob, 0, 1)

    # Convert to binary labels (threshold at 75th percentile)
    threshold = np.percentile(fraud_prob, 75)
    fraud_labels = (fraud_prob > threshold).astype(int)

    # Calculate class distribution
    n_fraud = fraud_labels.sum()
    n_legit = len(fraud_labels) - n_fraud
    fraud_pct = (n_fraud / len(fraud_labels)) * 100

    print(f"\nClass Distribution:")
    print(f"  Legitimate: {n_legit:,} ({100 - fraud_pct:.2f}%)")
    print(f"  Fraud: {n_fraud:,} ({fraud_pct:.2f}%)")
    print(f"  Fraud Threshold: {threshold:.3f}")

    if fraud_pct < 10 or fraud_pct > 40:
        print_warning(f"Imbalanced dataset detected ({fraud_pct:.1f}% fraud)")
    else:
        print_success("Balanced dataset created")

    return fraud_labels, fraud_prob


def prepare_features(df, fraud_labels):
    """Prepare features for classification"""
    print_section("STEP 3: PREPARING FEATURES")

    # Select numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude target-like columns
    exclude_cols = ['combined_anomaly_score', 'iso_forest_score', 'autoencoder_score']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Also exclude ID columns
    id_like = ['enrolment_id', 'operator_id']
    feature_cols = [col for col in feature_cols if col not in id_like]

    print(f"Selected {len(feature_cols)} features for classification")
    print_info(f"Sample features: {', '.join(feature_cols[:5])}...")

    X = df[feature_cols].copy()
    y = fraud_labels

    # Handle any remaining NaN or inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    print_success(f"Feature matrix shape: {X.shape}")
    print_success(f"Target vector shape: {y.shape}")

    return X, y, feature_cols


def split_data(X, y):
    """Split data into train and test sets"""
    print_section("STEP 4: SPLITTING DATA")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training set: {len(X_train):,} samples")
    print(f"  - Fraud: {y_train.sum():,} ({y_train.sum() / len(y_train) * 100:.2f}%)")
    print(f"  - Legitimate: {len(y_train) - y_train.sum():,}")

    print(f"\nTest set: {len(X_test):,} samples")
    print(f"  - Fraud: {y_test.sum():,} ({y_test.sum() / len(y_test) * 100:.2f}%)")
    print(f"  - Legitimate: {len(y_test) - y_test.sum():,}")

    print_success("Data split successfully")

    return X_train, X_test, y_train, y_test


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost Classifier"""
    print_section("STEP 5: TRAINING XGBOOST CLASSIFIER")

    print(f"Model Parameters:")
    print(f"  - N Estimators: {N_ESTIMATORS_XGB}")
    print(f"  - Max Depth: {MAX_DEPTH}")
    print(f"  - Learning Rate: 0.1")

    # Calculate scale_pos_weight for imbalanced data
    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1

    print(f"  - Scale Pos Weight: {scale_pos_weight:.2f} (for class imbalance)")

    # Initialize XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS_XGB,
        max_depth=MAX_DEPTH,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss'
    )

    # Train
    print_info("Training XGBoost...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print_success("XGBoost trained successfully")

    # Predictions
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    y_pred_proba_test = xgb_model.predict_proba(X_test)[:, 1]

    # Training metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_pred_proba_test)

    print(f"\nPerformance Metrics:")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")
    print(f"  Test ROC-AUC: {test_auc:.4f}")

    # Save model
    joblib.dump(xgb_model, XGBOOST_MODEL)
    print_success(f"Model saved: {XGBOOST_MODEL}")

    return xgb_model, y_pred_test, y_pred_proba_test


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest Classifier"""
    print_section("STEP 6: TRAINING RANDOM FOREST CLASSIFIER")

    print(f"Model Parameters:")
    print(f"  - N Estimators: {N_ESTIMATORS_RF}")
    print(f"  - Max Depth: {MAX_DEPTH}")
    print(f"  - Min Samples Split: 5")

    # Calculate class weights for imbalanced data
    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    class_weight = {0: 1.0, 1: n_negative / n_positive if n_positive > 0 else 1.0}

    print(f"  - Class Weight: {class_weight}")

    # Initialize Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS_RF,
        max_depth=MAX_DEPTH,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    # Train
    print_info("Training Random Forest...")
    rf_model.fit(X_train, y_train)

    print_success("Random Forest trained successfully")

    # Predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    y_pred_proba_test = rf_model.predict_proba(X_test)[:, 1]

    # Training metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_pred_proba_test)

    print(f"\nPerformance Metrics:")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")
    print(f"  Test ROC-AUC: {test_auc:.4f}")

    # Save model
    joblib.dump(rf_model, RF_MODEL)
    print_success(f"Model saved: {RF_MODEL}")

    return rf_model, y_pred_test, y_pred_proba_test


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_models(y_test, xgb_pred, xgb_proba, rf_pred, rf_proba):
    """Comprehensive model evaluation"""
    print_section("STEP 7: MODEL EVALUATION")

    # XGBoost Evaluation
    print("\n📊 XGBOOST PERFORMANCE:")
    print("─" * 60)
    print(classification_report(y_test, xgb_pred, target_names=['Legitimate', 'Fraud']))

    # Random Forest Evaluation
    print("\n📊 RANDOM FOREST PERFORMANCE:")
    print("─" * 60)
    print(classification_report(y_test, rf_pred, target_names=['Legitimate', 'Fraud']))

    # Confusion Matrices
    print("\n📊 CONFUSION MATRICES:")
    print("─" * 60)
    xgb_cm = confusion_matrix(y_test, xgb_pred)
    rf_cm = confusion_matrix(y_test, rf_pred)

    print("\nXGBoost:")
    print(f"  TN: {xgb_cm[0, 0]:,}  |  FP: {xgb_cm[0, 1]:,}")
    print(f"  FN: {xgb_cm[1, 0]:,}  |  TP: {xgb_cm[1, 1]:,}")

    print("\nRandom Forest:")
    print(f"  TN: {rf_cm[0, 0]:,}  |  FP: {rf_cm[0, 1]:,}")
    print(f"  FN: {rf_cm[1, 0]:,}  |  TP: {rf_cm[1, 1]:,}")

    # ROC-AUC Scores
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    rf_auc = roc_auc_score(y_test, rf_proba)

    print(f"\n📊 ROC-AUC SCORES:")
    print("─" * 60)
    print(f"  XGBoost: {xgb_auc:.4f}")
    print(f"  Random Forest: {rf_auc:.4f}")

    # Best model
    if xgb_auc > rf_auc:
        print_success(f"\n🏆 Best Model: XGBoost (AUC: {xgb_auc:.4f})")
    else:
        print_success(f"\n🏆 Best Model: Random Forest (AUC: {rf_auc:.4f})")

    return xgb_cm, rf_cm, xgb_auc, rf_auc


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def analyze_feature_importance(xgb_model, rf_model, feature_names):
    """Analyze and compare feature importance"""
    print_section("STEP 8: FEATURE IMPORTANCE ANALYSIS")

    # XGBoost feature importance
    xgb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Random Forest feature importance
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🔝 TOP 10 FEATURES (XGBoost):")
    print("─" * 60)
    for idx, row in xgb_importance.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    print("\n🔝 TOP 10 FEATURES (Random Forest):")
    print("─" * 60)
    for idx, row in rf_importance.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    return xgb_importance, rf_importance


# ============================================================================
# ENSEMBLE PREDICTIONS
# ============================================================================

def create_ensemble_predictions(xgb_proba, rf_proba):
    """Combine XGBoost and Random Forest predictions"""
    print_section("STEP 9: CREATING ENSEMBLE PREDICTIONS")

    print("Combining predictions from both models...")
    print("  Method: Weighted Average (XGB: 0.6, RF: 0.4)")

    # Weighted average (XGBoost gets slightly higher weight)
    ensemble_proba = (xgb_proba * 0.6) + (rf_proba * 0.4)

    # Convert to binary predictions (threshold = 0.5)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    print_success("Ensemble predictions created")

    return ensemble_pred, ensemble_proba


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(y_test, xgb_pred, xgb_proba, rf_pred, rf_proba,
                          xgb_cm, rf_cm, xgb_importance, rf_importance):
    """Create comprehensive fraud detection visualizations"""
    print_section("STEP 10: CREATING VISUALIZATIONS")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Fraud Classification Analysis - XGBoost + Random Forest',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. XGBoost Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    ax1.set_title('XGBoost Confusion Matrix', fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # 2. Random Forest Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    ax2.set_title('Random Forest Confusion Matrix', fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    # 3. ROC Curves
    ax3 = fig.add_subplot(gs[0, 2])
    from sklearn.metrics import roc_curve
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)

    ax3.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC={roc_auc_score(y_test, xgb_proba):.3f})',
             linewidth=2, color='blue')
    ax3.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC={roc_auc_score(y_test, rf_proba):.3f})',
             linewidth=2, color='green')
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess')
    ax3.set_title('ROC Curves', fontweight='bold')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Precision-Recall Curves
    ax4 = fig.add_subplot(gs[1, 0])
    from sklearn.metrics import precision_recall_curve
    xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_proba)
    rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_proba)

    ax4.plot(xgb_recall, xgb_precision, label='XGBoost', linewidth=2, color='blue')
    ax4.plot(rf_recall, rf_precision, label='Random Forest', linewidth=2, color='green')
    ax4.set_title('Precision-Recall Curves', fontweight='bold')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. XGBoost Feature Importance
    ax5 = fig.add_subplot(gs[1, 1])
    top_features_xgb = xgb_importance.head(10)
    ax5.barh(range(len(top_features_xgb)), top_features_xgb['importance'], color='steelblue')
    ax5.set_yticks(range(len(top_features_xgb)))
    ax5.set_yticklabels(top_features_xgb['feature'], fontsize=8)
    ax5.invert_yaxis()
    ax5.set_title('Top 10 Features (XGBoost)', fontweight='bold')
    ax5.set_xlabel('Importance')
    ax5.grid(True, alpha=0.3, axis='x')

    # 6. Random Forest Feature Importance
    ax6 = fig.add_subplot(gs[1, 2])
    top_features_rf = rf_importance.head(10)
    ax6.barh(range(len(top_features_rf)), top_features_rf['importance'], color='seagreen')
    ax6.set_yticks(range(len(top_features_rf)))
    ax6.set_yticklabels(top_features_rf['feature'], fontsize=8)
    ax6.invert_yaxis()
    ax6.set_title('Top 10 Features (Random Forest)', fontweight='bold')
    ax6.set_xlabel('Importance')
    ax6.grid(True, alpha=0.3, axis='x')

    # 7. Prediction Score Distribution (XGBoost)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(xgb_proba[y_test == 0], bins=50, alpha=0.6, label='Legitimate', color='blue')
    ax7.hist(xgb_proba[y_test == 1], bins=50, alpha=0.6, label='Fraud', color='red')
    ax7.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax7.set_title('XGBoost Score Distribution', fontweight='bold')
    ax7.set_xlabel('Fraud Probability')
    ax7.set_ylabel('Frequency')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Prediction Score Distribution (Random Forest)
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(rf_proba[y_test == 0], bins=50, alpha=0.6, label='Legitimate', color='blue')
    ax8.hist(rf_proba[y_test == 1], bins=50, alpha=0.6, label='Fraud', color='red')
    ax8.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax8.set_title('Random Forest Score Distribution', fontweight='bold')
    ax8.set_xlabel('Fraud Probability')
    ax8.set_ylabel('Frequency')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Model Comparison
    ax9 = fig.add_subplot(gs[2, 2])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    xgb_metrics = [
        accuracy_score(y_test, xgb_pred),
        precision_score(y_test, xgb_pred),
        recall_score(y_test, xgb_pred),
        f1_score(y_test, xgb_pred)
    ]
    rf_metrics = [
        accuracy_score(y_test, rf_pred),
        precision_score(y_test, rf_pred),
        recall_score(y_test, rf_pred),
        f1_score(y_test, rf_pred)
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax9.bar(x - width / 2, xgb_metrics, width, label='XGBoost', color='steelblue')
    ax9.bar(x + width / 2, rf_metrics, width, label='Random Forest', color='seagreen')
    ax9.set_title('Model Performance Comparison', fontweight='bold')
    ax9.set_ylabel('Score')
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics, rotation=45, ha='right')
    ax9.legend()
    ax9.set_ylim(0, 1.1)
    ax9.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (xgb_val, rf_val) in enumerate(zip(xgb_metrics, rf_metrics)):
        ax9.text(i - width / 2, xgb_val + 0.02, f'{xgb_val:.3f}', ha='center', fontsize=8)
        ax9.text(i + width / 2, rf_val + 0.02, f'{rf_val:.3f}', ha='center', fontsize=8)

    # Save figure
    output_path = VIZ_DIR / '03_fraud_patterns.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print_success(f"Saved: {output_path}")


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(df, y_pred_full, fraud_proba_full):
    """Save fraud predictions to file"""
    print_section("STEP 11: SAVING RESULTS")

    # Add predictions to dataframe
    results_df = df.copy()
    results_df['fraud_label'] = y_pred_full
    results_df['fraud_probability'] = fraud_proba_full

    # Categorize fraud risk
    def categorize_fraud_risk(prob):
        if prob >= 0.8:
            return 'Very High'
        elif prob >= 0.6:
            return 'High'
        elif prob >= 0.4:
            return 'Medium'
        elif prob >= 0.2:
            return 'Low'
        else:
            return 'Very Low'

    results_df['fraud_risk_level'] = results_df['fraud_probability'].apply(categorize_fraud_risk)

    # Save to CSV
    results_df.to_csv(OUTPUT_FILE, index=False)
    print_success(f"Results saved: {OUTPUT_FILE}")
    print_info(f"Columns added: fraud_label, fraud_probability, fraud_risk_level")

    # Save top 100 fraud risks
    top_100_fraud = results_df.nlargest(100, 'fraud_probability')
    top_100_file = REPORTS_DIR / 'top_100_fraud_risks.csv'
    top_100_fraud.to_csv(top_100_file, index=False)
    print_success(f"Top 100 fraud risks saved: {top_100_file}")

    # Risk distribution
    risk_counts = Counter(results_df['fraud_risk_level'])
    print("\nFraud Risk Distribution:")
    for risk in ['Very High', 'High', 'Medium', 'Low', 'Very Low']:
        if risk in risk_counts:
            count = risk_counts[risk]
            pct = (count / len(results_df)) * 100
            print(f"  {risk:10s}: {count:6,} ({pct:5.2f}%)")

    return results_df


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(xgb_model, rf_model, y_test, xgb_pred, rf_pred,
                    xgb_auc, rf_auc, xgb_importance):
    """Generate comprehensive fraud classification report"""
    print_section("STEP 12: GENERATING REPORT")

    report = []
    report.append("=" * 80)
    report.append("UIDAI DATA HACKATHON 2026 - FRAUD CLASSIFICATION REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Script: 05_fraud_classification.py")
    report.append("\n" + "=" * 80)

    # Model Configuration
    report.append("\n1. MODEL CONFIGURATION")
    report.append("─" * 80)
    report.append(f"  XGBoost:")
    report.append(f"    - N Estimators: {N_ESTIMATORS_XGB}")
    report.append(f"    - Max Depth: {MAX_DEPTH}")
    report.append(f"  Random Forest:")
    report.append(f"    - N Estimators: {N_ESTIMATORS_RF}")
    report.append(f"    - Max Depth: {MAX_DEPTH}")

    # Performance Metrics
    report.append("\n2. PERFORMANCE METRICS")
    report.append("─" * 80)
    report.append(f"\n  XGBoost:")
    report.append(f"    - Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
    report.append(f"    - Precision: {precision_score(y_test, xgb_pred):.4f}")
    report.append(f"    - Recall: {recall_score(y_test, xgb_pred):.4f}")
    report.append(f"    - F1-Score: {f1_score(y_test, xgb_pred):.4f}")
    report.append(f"    - ROC-AUC: {xgb_auc:.4f}")

    report.append(f"\n  Random Forest:")
    report.append(f"    - Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    report.append(f"    - Precision: {precision_score(y_test, rf_pred):.4f}")
    report.append(f"    - Recall: {recall_score(y_test, rf_pred):.4f}")
    report.append(f"    - F1-Score: {f1_score(y_test, rf_pred):.4f}")
    report.append(f"    - ROC-AUC: {rf_auc:.4f}")

    # Top Features
    report.append("\n3. TOP 10 MOST IMPORTANT FEATURES")
    report.append("─" * 80)
    for idx, row in xgb_importance.head(10).iterrows():
        report.append(f"  {idx + 1:2d}. {row['feature']:30s}: {row['importance']:.4f}")

    # Next Steps
    report.append("\n4. NEXT STEPS")
    report.append("─" * 80)
    report.append("  1. Run 06_hybrid_model.py to combine all models")
    report.append("  2. Review top 100 fraud risks manually")
    report.append("  3. Fine-tune thresholds based on business requirements")
    report.append("  4. Deploy models to production")

    report.append("\n" + "=" * 80)
    report.append("END OF FRAUD CLASSIFICATION REPORT")
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
    """Main fraud classification function"""

    print_header("UIDAI DATA HACKATHON 2026 - FRAUD CLASSIFICATION")
    start_time = datetime.now()

    # Load data
    df = load_data()

    # Create fraud labels
    fraud_labels, fraud_proba = create_fraud_labels(df)

    # Prepare features
    X, y, feature_names = prepare_features(df, fraud_labels)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    xgb_model, xgb_pred, xgb_proba = train_xgboost(X_train, y_train, X_test, y_test)
    rf_model, rf_pred, rf_proba = train_random_forest(X_train, y_train, X_test, y_test)

    # Evaluate models
    xgb_cm, rf_cm, xgb_auc, rf_auc = evaluate_models(y_test, xgb_pred, xgb_proba,
                                                     rf_pred, rf_proba)

    # Feature importance
    xgb_importance, rf_importance = analyze_feature_importance(xgb_model, rf_model, feature_names)

    # Ensemble predictions
    ensemble_pred, ensemble_proba = create_ensemble_predictions(xgb_proba, rf_proba)

    # Predict on full dataset
    X_full = X  # Use all data
    xgb_pred_full = xgb_model.predict_proba(X_full)[:, 1]
    rf_pred_full = rf_model.predict_proba(X_full)[:, 1]
    ensemble_pred_full = (xgb_pred_full * 0.6) + (rf_pred_full * 0.4)
    y_pred_full = (ensemble_pred_full >= 0.5).astype(int)

    # Create visualizations
    create_visualizations(y_test, xgb_pred, xgb_proba, rf_pred, rf_proba,
                          xgb_cm, rf_cm, xgb_importance, rf_importance)

    # Save results
    results_df = save_results(df, y_pred_full, ensemble_pred_full)

    # Generate report
    generate_report(xgb_model, rf_model, y_test, xgb_pred, rf_pred,
                    xgb_auc, rf_auc, xgb_importance)

    # Summary
    print_section("EXECUTION SUMMARY")
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.2f} seconds ({duration / 60:.1f} minutes)")
    print(f"\nRecords Classified: {len(results_df):,}")
    print(f"Fraud Cases Detected: {y_pred_full.sum():,} ({y_pred_full.sum() / len(y_pred_full) * 100:.2f}%)")
    print(f"High Risk Cases: {(ensemble_pred_full >= 0.8).sum():,}")

    print_header("✅ FRAUD CLASSIFICATION COMPLETE!")
    print_info("Next step: Run 06_hybrid_model.py to combine all 4 models")


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

