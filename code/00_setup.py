#!/usr/bin/env python3
"""
============================================================================
UIDAI DATA HACKATHON 2026 - ENVIRONMENT SETUP & VALIDATION
============================================================================
File: 00_setup.py
Purpose: Validate Python environment, libraries, and folder structure
Author: Generated with AI Assistance
Date: January 2026
============================================================================
This script:
1. Checks Python version (3.8+ required)
2. Validates all required libraries are installed
3. Checks folder structure exists
4. Creates missing folders automatically
5. Tests basic functionality of key libraries
6. Generates environment report
============================================================================
"""

import sys
import os
from pathlib import Path
import platform
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

REQUIRED_PYTHON_VERSION = (3, 8)
PROJECT_ROOT = Path(__file__).parent.parent  # Parent of 'code' folder

REQUIRED_FOLDERS = [
    'code',
    'data',
    'outputs/data',
    'outputs/models',
    'outputs/visualizations',
    'outputs/reports',
    'outputs/logs',
    'notebooks',
    'presentation',
    'documentation',
    'tests',
    'config'
]

REQUIRED_LIBRARIES = {
    # Library name: (import name, version check)
    'pandas': ('pandas', True),
    'numpy': ('numpy', True),
    'scikit-learn': ('sklearn', True),
    'xgboost': ('xgboost', True),
    'tensorflow': ('tensorflow', True),
    'matplotlib': ('matplotlib', True),
    'seaborn': ('seaborn', True),
    'plotly': ('plotly', True),
    'scipy': ('scipy', True),
    'joblib': ('joblib', False),
    'tqdm': ('tqdm', False),
    'statsmodels': ('statsmodels', True),
    'jupyter': ('jupyter', False),
}


# ============================================================================
# COLOR CODES FOR TERMINAL OUTPUT
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def check_python_version():
    """Check if Python version meets requirements"""
    print_header("CHECKING PYTHON VERSION")

    current_version = sys.version_info[:2]
    version_str = f"{current_version[0]}.{current_version[1]}"

    print(f"Current Python Version: {version_str}")
    print(f"Required Python Version: {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}+")

    if current_version >= REQUIRED_PYTHON_VERSION:
        print_success(f"Python {version_str} is compatible!")
        return True
    else:
        print_error(
            f"Python {version_str} is too old. Please upgrade to Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}+")
        return False


def check_system_info():
    """Display system information"""
    print_header("SYSTEM INFORMATION")

    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Implementation: {platform.python_implementation()}")
    print(f"Python Build: {platform.python_build()}")
    print_success("System information collected")


def check_folder_structure():
    """Check and create required folder structure"""
    print_header("CHECKING FOLDER STRUCTURE")

    missing_folders = []
    created_folders = []

    for folder in REQUIRED_FOLDERS:
        folder_path = PROJECT_ROOT / folder

        if folder_path.exists():
            print_success(f"Found: {folder}/")
        else:
            missing_folders.append(folder)
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                created_folders.append(folder)
                print_warning(f"Created: {folder}/")
            except Exception as e:
                print_error(f"Failed to create {folder}/: {str(e)}")

    if created_folders:
        print_info(f"Created {len(created_folders)} missing folders")

    if not missing_folders:
        print_success("All required folders exist!")

    return len(missing_folders) == 0 or len(created_folders) == len(missing_folders)


def check_libraries():
    """Check if all required libraries are installed"""
    print_header("CHECKING REQUIRED LIBRARIES")

    results = {
        'installed': [],
        'missing': [],
        'errors': []
    }

    for lib_name, (import_name, check_version) in REQUIRED_LIBRARIES.items():
        try:
            # Try to import the library
            module = __import__(import_name)

            # Get version if available
            version = "unknown"
            if check_version and hasattr(module, '__version__'):
                version = module.__version__

            print_success(f"{lib_name:20s} → {version}")
            results['installed'].append((lib_name, version))

        except ImportError as e:
            print_error(f"{lib_name:20s} → NOT INSTALLED")
            results['missing'].append(lib_name)
        except Exception as e:
            print_warning(f"{lib_name:20s} → ERROR: {str(e)}")
            results['errors'].append((lib_name, str(e)))

    return results


def test_basic_functionality():
    """Test basic functionality of key libraries"""
    print_header("TESTING BASIC FUNCTIONALITY")

    tests_passed = 0
    tests_failed = 0

    # Test 1: NumPy array operations
    try:
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0
        print_success("NumPy: Array operations working")
        tests_passed += 1
    except Exception as e:
        print_error(f"NumPy: {str(e)}")
        tests_failed += 1

    # Test 2: Pandas DataFrame operations
    try:
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert df.shape == (3, 2)
        print_success("Pandas: DataFrame operations working")
        tests_passed += 1
    except Exception as e:
        print_error(f"Pandas: {str(e)}")
        tests_failed += 1

    # Test 3: Scikit-learn model creation
    try:
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(random_state=42)
        print_success("Scikit-learn: Model creation working")
        tests_passed += 1
    except Exception as e:
        print_error(f"Scikit-learn: {str(e)}")
        tests_failed += 1

    # Test 4: XGBoost import
    try:
        import xgboost as xgb
        print_success("XGBoost: Import successful")
        tests_passed += 1
    except Exception as e:
        print_error(f"XGBoost: {str(e)}")
        tests_failed += 1

    # Test 5: TensorFlow/Keras import
    try:
        import tensorflow as tf
        from tensorflow import keras
        print_success(f"TensorFlow: Import successful (version {tf.__version__})")
        tests_passed += 1
    except Exception as e:
        print_error(f"TensorFlow: {str(e)}")
        tests_failed += 1

    # Test 6: Matplotlib plotting
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close(fig)
        print_success("Matplotlib: Plotting working")
        tests_passed += 1
    except Exception as e:
        print_error(f"Matplotlib: {str(e)}")
        tests_failed += 1

    # Test 7: Plotly
    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 2, 3])])
        print_success("Plotly: Interactive plotting working")
        tests_passed += 1
    except Exception as e:
        print_error(f"Plotly: {str(e)}")
        tests_failed += 1

    return tests_passed, tests_failed


def generate_report(python_ok, folders_ok, lib_results, tests_passed, tests_failed):
    """Generate environment validation report"""
    print_header("ENVIRONMENT VALIDATION REPORT")

    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project Root: {PROJECT_ROOT}")
    print()

    # Python Version
    print("1. PYTHON VERSION:")
    if python_ok:
        print_success("   Compatible")
    else:
        print_error("   Incompatible - Upgrade required")
    print()

    # Folder Structure
    print("2. FOLDER STRUCTURE:")
    if folders_ok:
        print_success(f"   All {len(REQUIRED_FOLDERS)} folders exist")
    else:
        print_warning("   Some folders were created automatically")
    print()

    # Libraries
    print("3. LIBRARIES:")
    print(f"   Installed: {len(lib_results['installed'])}")
    print(f"   Missing: {len(lib_results['missing'])}")
    print(f"   Errors: {len(lib_results['errors'])}")

    if lib_results['missing']:
        print_error("   Missing libraries:")
        for lib in lib_results['missing']:
            print(f"      - {lib}")
        print_warning("   Install with: pip install -r code/requirements.txt")
    print()

    # Functionality Tests
    print("4. FUNCTIONALITY TESTS:")
    print(f"   Passed: {tests_passed}")
    print(f"   Failed: {tests_failed}")
    total_tests = tests_passed + tests_failed
    if total_tests > 0:
        success_rate = (tests_passed / total_tests) * 100
        print(f"   Success Rate: {success_rate:.1f}%")
    print()

    # Overall Status
    print("5. OVERALL STATUS:")
    all_passed = (python_ok and
                  folders_ok and
                  len(lib_results['missing']) == 0 and
                  tests_failed == 0)

    if all_passed:
        print_success("   ✅ ENVIRONMENT READY FOR DEVELOPMENT!")
        print_success("   You can proceed to the next step.")
        print_info("   Next: Run 01_data_loading.py")
    else:
        print_warning("   ⚠️  ENVIRONMENT HAS ISSUES")
        print_info("   Please fix the issues above before proceeding.")

    return all_passed


def save_report_to_file(python_ok, folders_ok, lib_results, tests_passed, tests_failed):
    """Save validation report to log file"""
    try:
        log_dir = PROJECT_ROOT / 'outputs' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / 'setup_validation.txt'

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("UIDAI HACKATHON 2026 - ENVIRONMENT VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Project Root: {PROJECT_ROOT}\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write(f"Platform: {platform.platform()}\n\n")

            f.write("PYTHON VERSION:\n")
            f.write(f"  Status: {'✅ Compatible' if python_ok else '❌ Incompatible'}\n\n")

            f.write("FOLDER STRUCTURE:\n")
            f.write(f"  Status: {'✅ Complete' if folders_ok else '⚠️  Auto-created'}\n\n")

            f.write("INSTALLED LIBRARIES:\n")
            for lib_name, version in lib_results['installed']:
                f.write(f"  ✅ {lib_name:20s} → {version}\n")

            if lib_results['missing']:
                f.write("\nMISSING LIBRARIES:\n")
                for lib in lib_results['missing']:
                    f.write(f"  ❌ {lib}\n")

            f.write(f"\nFUNCTIONALITY TESTS:\n")
            f.write(f"  Passed: {tests_passed}\n")
            f.write(f"  Failed: {tests_failed}\n")

            all_passed = (python_ok and
                          folders_ok and
                          len(lib_results['missing']) == 0 and
                          tests_failed == 0)

            f.write(f"\nOVERALL STATUS: {'✅ READY' if all_passed else '⚠️  ISSUES FOUND'}\n")

        print_success(f"Report saved to: {log_file}")

    except Exception as e:
        print_warning(f"Could not save report: {str(e)}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main setup validation function"""

    print(f"\n{Colors.BOLD}{Colors.OKCYAN}")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                    UIDAI DATA HACKATHON 2026                               ║")
    print("║              ENVIRONMENT SETUP & VALIDATION SCRIPT                         ║")
    print("║                                                                            ║")
    print("║  This script will validate your development environment and ensure        ║")
    print("║  everything is ready for the hackathon project.                           ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print(Colors.ENDC)

    # Step 1: Check Python version
    python_ok = check_python_version()

    # Step 2: Display system info
    check_system_info()

    # Step 3: Check folder structure
    folders_ok = check_folder_structure()

    # Step 4: Check libraries
    lib_results = check_libraries()

    # Step 5: Test functionality
    tests_passed, tests_failed = test_basic_functionality()

    # Step 6: Generate report
    all_passed = generate_report(python_ok, folders_ok, lib_results,
                                 tests_passed, tests_failed)

    # Step 7: Save report to file
    save_report_to_file(python_ok, folders_ok, lib_results,
                        tests_passed, tests_failed)

    # Exit with appropriate status code
    if all_passed:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 SUCCESS! Your environment is ready!{Colors.ENDC}\n")
        sys.exit(0)
    else:
        print(
            f"\n{Colors.WARNING}{Colors.BOLD}⚠️  Please fix the issues above and run this script again.{Colors.ENDC}\n")
        sys.exit(1)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Script interrupted by user.{Colors.ENDC}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.FAIL}Unexpected error: {str(e)}{Colors.ENDC}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)

