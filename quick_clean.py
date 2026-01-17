#!/usr/bin/env python3
"""
Quick cleanup - deletes all outputs without asking
Use with caution!
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def quick_clean():
    """Delete and recreate outputs folder"""
    if OUTPUTS_DIR.exists():
        shutil.rmtree(OUTPUTS_DIR)
        print(f"✅ Deleted outputs/")

    # Recreate folder structure
    folders = [
        "outputs/data",
        "outputs/models",
        "outputs/visualizations",
        "outputs/reports",
        "outputs/logs"
    ]

    for folder in folders:
        (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)

    print("✅ Recreated clean folder structure")
    print("Ready for fresh run!")


if __name__ == "__main__":
    quick_clean()
