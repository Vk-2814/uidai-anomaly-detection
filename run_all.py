#!/usr/bin/env python3
"""
============================================================================
UIDAI DATA HACKATHON 2026 - MASTER EXECUTION SCRIPT
============================================================================
File: run_all.py
Purpose: Execute entire fraud detection pipeline with one command
Author: Generated with AI Assistance
Date: January 2026
============================================================================
This script:
1. Validates environment setup
2. Runs all 9 pipeline scripts in sequence
3. Handles errors gracefully
4. Provides progress tracking
5. Generates execution report
6. Estimates completion time
============================================================================
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
CODE_DIR = PROJECT_ROOT / "code"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Create logs directory
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline scripts in execution order
PIPELINE_SCRIPTS = [
    {
        "name": "Environment Setup Validation",
        "script": "00_setup.py",
        "estimated_time": 10,  # seconds
        "critical": True,
        "description": "Validate Python environment and dependencies"
    },
    {
        "name": "Data Loading",
        "script": "01_data_loading.py",
        "estimated_time": 15,
        "critical": True,
        "description": "Load and validate Aadhaar enrolment data"
    },
    {
        "name": "Exploratory Data Analysis",
        "script": "02_eda.py",
        "estimated_time": 30,
        "critical": False,
        "description": "Generate statistical summaries and visualizations"
    },
    {
        "name": "Data Preprocessing & Feature Engineering",
        "script": "03_preprocessing.py",
        "estimated_time": 25,
        "critical": True,
        "description": "Clean data and engineer 35+ features"
    },
    {
        "name": "Anomaly Detection (Unsupervised)",
        "script": "04_anomaly_detection.py",
        "estimated_time": 300,  # 5 minutes
        "critical": True,
        "description": "Train Isolation Forest + Autoencoder models"
    },
    {
        "name": "Fraud Classification (Supervised)",
        "script": "05_fraud_classification.py",
        "estimated_time": 180,  # 3 minutes
        "critical": True,
        "description": "Train XGBoost + Random Forest classifiers"
    },
    {
        "name": "Hybrid Model Ensemble",
        "script": "06_hybrid_model.py",
        "estimated_time": 20,
        "critical": True,
        "description": "Combine all models into unified risk score"
    },
    {
        "name": "Advanced Visualizations",
        "script": "07_visualization.py",
        "estimated_time": 45,
        "critical": False,
        "description": "Create dashboards and 15+ visualizations"
    },
    {
        "name": "Report Generation",
        "script": "08_report_generation.py",
        "estimated_time": 15,
        "critical": False,
        "description": "Generate executive and technical reports"
    }
]


# Color codes for terminal output (ANSI)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_banner():
    """Print welcome banner"""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║           UIDAI DATA HACKATHON 2026 - MASTER PIPELINE EXECUTOR           ║
║                 Fraud & Anomaly Detection System v1.0                     ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
{Colors.ENDC}
"""
    print(banner)


def print_header(text):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}{Colors.ENDC}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.ENDC}")


def format_time(seconds):
    """Format seconds to human-readable time"""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes} min {secs} sec"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours} hr {minutes} min"


def calculate_total_time():
    """Calculate estimated total execution time"""
    total = sum(script["estimated_time"] for script in PIPELINE_SCRIPTS)
    return total


def check_environment():
    """Check if environment is ready"""
    print_header("ENVIRONMENT CHECK")

    # Check Python version
    py_version = sys.version_info
    print_info(f"Python Version: {py_version.major}.{py_version.minor}.{py_version.micro}")

    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print_error("Python 3.8+ required!")
        return False

    print_success("Python version compatible")

    # Check if code directory exists
    if not CODE_DIR.exists():
        print_error(f"Code directory not found: {CODE_DIR}")
        return False

    print_success(f"Code directory found: {CODE_DIR}")

    # Check if all scripts exist
    missing_scripts = []
    for script_info in PIPELINE_SCRIPTS:
        script_path = CODE_DIR / script_info["script"]
        if not script_path.exists():
            missing_scripts.append(script_info["script"])

    if missing_scripts:
        print_error(f"Missing scripts: {', '.join(missing_scripts)}")
        return False

    print_success(f"All {len(PIPELINE_SCRIPTS)} scripts found")

    # Check for required packages (basic check)
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'sklearn', 'xgboost', 'tensorflow'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print_warning(f"Some packages may be missing: {', '.join(missing_packages)}")
        print_info("Run: pip install -r code/requirements.txt")
        response = input(f"\n{Colors.YELLOW}Continue anyway? (y/n): {Colors.ENDC}").lower()
        if response != 'y':
            return False
    else:
        print_success("All required packages installed")

    return True


def run_script(script_info, step_num, total_steps):
    """Execute a single script with progress tracking"""
    script_path = CODE_DIR / script_info["script"]

    # Print step header
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'─' * 80}")
    print(f"STEP {step_num}/{total_steps}: {script_info['name']}")
    print(f"{'─' * 80}{Colors.ENDC}")
    print(f"{Colors.CYAN}Script: {script_info['script']}")
    print(f"Description: {script_info['description']}")
    print(f"Estimated Time: {format_time(script_info['estimated_time'])}{Colors.ENDC}")
    print()

    # Start timer
    start_time = time.time()

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,  # Show output in real-time
            text=True,
            cwd=PROJECT_ROOT
        )

        # Calculate execution time
        execution_time = time.time() - start_time

        # Check result
        if result.returncode == 0:
            print_success(f"Completed in {format_time(execution_time)}")
            return True, execution_time
        else:
            print_error(f"Script failed with exit code {result.returncode}")
            return False, execution_time

    except Exception as e:
        execution_time = time.time() - start_time
        print_error(f"Error executing script: {str(e)}")
        return False, execution_time


def generate_execution_report(results, total_time):
    """Generate execution summary report"""
    print_header("EXECUTION SUMMARY")

    # Calculate statistics
    total_scripts = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total_scripts - successful

    # Print summary
    print(f"{Colors.BOLD}Pipeline Execution Complete!{Colors.ENDC}\n")
    print(f"Total Scripts:      {total_scripts}")
    print(f"{Colors.GREEN}Successful:         {successful} ✓{Colors.ENDC}")
    if failed > 0:
        print(f"{Colors.RED}Failed:             {failed} ✗{Colors.ENDC}")
    else:
        print(f"Failed:             {failed}")
    print(f"\nTotal Execution Time: {format_time(total_time)}")

    # Print detailed results
    print(f"\n{Colors.BOLD}Detailed Results:{Colors.ENDC}\n")
    print(f"{'Step':<6} {'Status':<10} {'Time':<15} {'Script':<35}")
    print("─" * 80)

    for i, result in enumerate(results, 1):
        status = f"{Colors.GREEN}SUCCESS{Colors.ENDC}" if result["success"] else f"{Colors.RED}FAILED{Colors.ENDC}"
        time_str = format_time(result["time"])
        print(f"{i:<6} {status:<20} {time_str:<15} {result['script']:<35}")

    # Save to log file
    log_file = LOGS_DIR / f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("UIDAI FRAUD DETECTION PIPELINE - EXECUTION LOG\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Scripts: {total_scripts}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Total Time: {format_time(total_time)}\n\n")
        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(results, 1):
            status = "SUCCESS" if result["success"] else "FAILED"
            f.write(f"Step {i}: {result['name']}\n")
            f.write(f"  Script: {result['script']}\n")
            f.write(f"  Status: {status}\n")
            f.write(f"  Time: {format_time(result['time'])}\n\n")

    print(f"\n{Colors.CYAN}Log file saved: {log_file}{Colors.ENDC}")

    # Print next steps
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL SCRIPTS EXECUTED SUCCESSFULLY! 🎉{Colors.ENDC}")
        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print("  1. Review outputs in: outputs/ folder")
        print("  2. Open dashboard: outputs/visualizations/dashboard.html")
        print("  3. Read reports: outputs/reports/EXECUTIVE_SUMMARY.txt")
        print("  4. Check models: outputs/models/")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  PIPELINE COMPLETED WITH ERRORS{Colors.ENDC}")
        print(f"\nFailed scripts:")
        for result in results:
            if not result["success"]:
                print(f"  • {result['script']}")
        print(f"\nCheck log file for details: {log_file}")


def confirm_execution():
    """Ask user to confirm execution"""
    total_time = calculate_total_time()

    print_info(f"This will execute {len(PIPELINE_SCRIPTS)} scripts")
    print_info(f"Estimated total time: {format_time(total_time)}")
    print_info(f"Scripts will run automatically in sequence")
    print()

    # Show script list
    print(f"{Colors.BOLD}Scripts to be executed:{Colors.ENDC}\n")
    for i, script in enumerate(PIPELINE_SCRIPTS, 1):
        critical = "🔴 CRITICAL" if script["critical"] else "🟢 OPTIONAL"
        print(f"  {i}. {script['name']:<40} {critical}")

    print()
    response = input(f"{Colors.YELLOW}{Colors.BOLD}Start execution? (y/n): {Colors.ENDC}").lower()

    return response == 'y'


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    # Print banner
    print_banner()

    # Check environment
    if not check_environment():
        print_error("Environment check failed!")
        print_info("Please fix the issues and try again.")
        sys.exit(1)

    print_success("Environment check passed!")

    # Confirm execution
    if not confirm_execution():
        print_warning("Execution cancelled by user.")
        sys.exit(0)

    # Start execution
    print_header("STARTING PIPELINE EXECUTION")
    print_info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    pipeline_start = time.time()
    results = []

    # Execute each script
    for i, script_info in enumerate(PIPELINE_SCRIPTS, 1):
        success, exec_time = run_script(script_info, i, len(PIPELINE_SCRIPTS))

        results.append({
            "name": script_info["name"],
            "script": script_info["script"],
            "success": success,
            "time": exec_time
        })

        # If critical script fails, ask user whether to continue
        if not success and script_info["critical"]:
            print_error(f"Critical script failed: {script_info['name']}")
            response = input(f"\n{Colors.YELLOW}Continue with remaining scripts? (y/n): {Colors.ENDC}").lower()
            if response != 'y':
                print_warning("Pipeline execution stopped by user.")
                break

    # Calculate total time
    total_execution_time = time.time() - pipeline_start

    # Generate report
    generate_execution_report(results, total_execution_time)

    # Exit with appropriate code
    all_success = all(r["success"] for r in results)
    sys.exit(0 if all_success else 1)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Execution interrupted by user (Ctrl+C){Colors.ENDC}")
        print_info("Partial results may be available in outputs/ folder")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n{Colors.RED}❌ Unexpected error: {str(e)}{Colors.ENDC}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
