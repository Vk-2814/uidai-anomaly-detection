#!/usr/bin/env python3
"""
============================================================================
UIDAI DATA HACKATHON 2026 - CLEANUP SCRIPT
============================================================================
File: cleanup.py
Purpose: Delete all generated files from previous runs
Author: Generated with AI Assistance
Date: January 2026
============================================================================
Deletes:
- All files in outputs/ folder (keeps folder structure)
- Preserves: data/ folder (your input data)
- Preserves: code/ folder (your scripts)
============================================================================
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Folders to clean (will delete all files inside)
FOLDERS_TO_CLEAN = [
    OUTPUTS_DIR / "data",
    OUTPUTS_DIR / "models",
    OUTPUTS_DIR / "visualizations",
    OUTPUTS_DIR / "reports",
    OUTPUTS_DIR / "logs"
]

# Files to keep (optional - won't delete these)
FILES_TO_KEEP = [
    ".gitkeep",
    "README.md"
]


# Color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


# ============================================================================
# FUNCTIONS
# ============================================================================

def print_header(text):
    """Print header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}{Colors.ENDC}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.ENDC}")


def get_folder_size(folder):
    """Calculate total size of folder in MB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception:
        pass
    return total_size / (1024 * 1024)  # Convert to MB


def count_files(folder):
    """Count total files in folder"""
    count = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder):
            count += len(filenames)
    except Exception:
        pass
    return count


def scan_outputs():
    """Scan outputs folder and show what will be deleted"""
    print_header("SCANNING OUTPUT FOLDER")

    if not OUTPUTS_DIR.exists():
        print_warning("outputs/ folder not found. Nothing to clean!")
        return False, 0, 0

    total_files = 0
    total_size = 0

    print(f"{'Folder':<30} {'Files':<10} {'Size (MB)':<12}")
    print("─" * 70)

    for folder in FOLDERS_TO_CLEAN:
        if folder.exists():
            file_count = count_files(folder)
            folder_size = get_folder_size(folder)
            total_files += file_count
            total_size += folder_size

            folder_name = folder.relative_to(PROJECT_ROOT)
            print(f"{str(folder_name):<30} {file_count:<10} {folder_size:>10.2f} MB")

    print("─" * 70)
    print(f"{'TOTAL':<30} {total_files:<10} {total_size:>10.2f} MB")
    print()

    if total_files == 0:
        print_info("No files found to delete. Folder is already clean!")
        return False, 0, 0

    return True, total_files, total_size


def clean_folder(folder):
    """Clean a single folder (delete all files, keep folder structure)"""
    if not folder.exists():
        return 0

    deleted_count = 0

    try:
        for item in folder.iterdir():
            # Skip files in keep list
            if item.name in FILES_TO_KEEP:
                continue

            try:
                if item.is_file():
                    item.unlink()
                    deleted_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_count += 1
            except Exception as e:
                print_warning(f"Could not delete {item.name}: {e}")

    except Exception as e:
        print_warning(f"Error accessing {folder}: {e}")

    return deleted_count


def cleanup_all():
    """Clean all output folders"""
    print_header("CLEANING OUTPUT FOLDERS")

    total_deleted = 0

    for folder in FOLDERS_TO_CLEAN:
        if folder.exists():
            folder_name = folder.relative_to(PROJECT_ROOT)
            print(f"Cleaning: {folder_name}...", end=" ")

            deleted = clean_folder(folder)
            total_deleted += deleted

            if deleted > 0:
                print(f"{Colors.GREEN}✓ ({deleted} items){Colors.ENDC}")
            else:
                print(f"{Colors.CYAN}(empty){Colors.ENDC}")

    print()
    print_success(f"Deleted {total_deleted} items")
    return total_deleted


def create_backup():
    """Create backup of outputs folder before cleaning"""
    print_header("CREATING BACKUP")

    if not OUTPUTS_DIR.exists():
        print_info("No outputs folder to backup")
        return None

    # Check if folder is empty
    if count_files(OUTPUTS_DIR) == 0:
        print_info("Outputs folder is empty. Skipping backup.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"outputs_backup_{timestamp}"
    backup_path = PROJECT_ROOT / backup_name

    try:
        print(f"Creating backup: {backup_name}...", end=" ")
        shutil.copytree(OUTPUTS_DIR, backup_path)
        backup_size = get_folder_size(backup_path)
        print(f"{Colors.GREEN}✓ ({backup_size:.2f} MB){Colors.ENDC}")
        return backup_path
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.ENDC}")
        print_warning(f"Backup failed: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main cleanup function"""

    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                                                                  ║")
    print("║         UIDAI HACKATHON 2026 - OUTPUT CLEANUP SCRIPT            ║")
    print("║                                                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")

    # Scan what will be deleted
    has_files, total_files, total_size = scan_outputs()

    if not has_files:
        return

    # Ask for confirmation
    print(
        f"{Colors.YELLOW}{Colors.BOLD}⚠️  WARNING: This will delete {total_files} files ({total_size:.2f} MB){Colors.ENDC}")
    print()
    print("What to delete:")
    print("  ✓ All CSV files in outputs/data/")
    print("  ✓ All model files in outputs/models/")
    print("  ✓ All PNG/HTML files in outputs/visualizations/")
    print("  ✓ All reports in outputs/reports/")
    print("  ✓ All logs in outputs/logs/")
    print()
    print("What to KEEP:")
    print("  ✓ Your input data in data/ folder")
    print("  ✓ All Python scripts in code/ folder")
    print("  ✓ README.md and documentation")
    print()

    response = input(f"{Colors.YELLOW}{Colors.BOLD}Do you want to create a backup first? (y/n): {Colors.ENDC}").lower()

    backup_path = None
    if response == 'y':
        backup_path = create_backup()

    print()
    response = input(f"{Colors.YELLOW}{Colors.BOLD}Proceed with cleanup? (y/n): {Colors.ENDC}").lower()

    if response != 'y':
        print_warning("Cleanup cancelled by user.")
        return

    # Clean all folders
    deleted = cleanup_all()

    # Summary
    print_header("CLEANUP COMPLETE")
    print_success(f"Successfully deleted {deleted} items")

    if backup_path:
        print_info(f"Backup saved at: {backup_path}")
        print_info("You can delete the backup folder manually if not needed")

    print()
    print(f"{Colors.GREEN}{Colors.BOLD}✓ Ready for fresh run!{Colors.ENDC}")
    print(f"{Colors.CYAN}Run: python run_all.py{Colors.ENDC}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Cleanup cancelled by user (Ctrl+C){Colors.ENDC}")
    except Exception as e:
        print(f"\n\n{Colors.RED}❌ Error: {str(e)}{Colors.ENDC}")
        import traceback

        traceback.print_exc()
