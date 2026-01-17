#!/usr/bin/env python3
"""
HACKATHON DATA MERGER
=====================
Combines multiple UIDAI hackathon datasets into one unified dataset
Handles biometric, demographic, and enrollment data

Usage:
    python merge_hackathon_data.py

This will:
1. Read all CSV files from data/ folder
2. Merge biometric, demographic, and enrollment data
3. Create unified dataset for anomaly detection
4. Save as 'merged_dataset.csv' in data/ folder
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime

class HackathonDataMerger:
    """Merge and prepare UIDAI hackathon datasets"""

    def __init__(self, data_folder='data'):
        self.data_folder = Path(data_folder)
        self.biometric_files = []
        self.demographic_files = []
        self.enrolment_files = []
        self.merged_df = None

    def discover_files(self):
        """Find all CSV files and categorize them"""
        print("🔍 Discovering files...")

        all_files = list(self.data_folder.glob('*.csv'))

        for file in all_files:
            filename = file.name.lower()

            if 'biometric' in filename:
                self.biometric_files.append(file)
            elif 'demographic' in filename:
                self.demographic_files.append(file)
            elif 'enrolment' in filename or 'enrollment' in filename:
                self.enrolment_files.append(file)

        print(f"   ✓ Found {len(self.biometric_files)} biometric files")
        print(f"   ✓ Found {len(self.demographic_files)} demographic files")
        print(f"   ✓ Found {len(self.enrolment_files)} enrollment files")
        print(f"   ✓ Total: {len(all_files)} files")

        return self

    def load_biometric_data(self):
        """Load and combine all biometric files"""
        if not self.biometric_files:
            print("⚠️ No biometric files found")
            return pd.DataFrame()

        print(f"\n📊 Loading {len(self.biometric_files)} biometric files...")

        dfs = []
        for file in self.biometric_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"   ✓ Loaded {file.name}: {len(df):,} records")
            except Exception as e:
                print(f"   ❌ Error loading {file.name}: {e}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            print(f"   ✅ Total biometric records: {len(combined):,}")
            return combined

        return pd.DataFrame()

    def load_demographic_data(self):
        """Load and combine all demographic files"""
        if not self.demographic_files:
            print("⚠️ No demographic files found")
            return pd.DataFrame()

        print(f"\n📊 Loading {len(self.demographic_files)} demographic files...")

        dfs = []
        for file in self.demographic_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"   ✓ Loaded {file.name}: {len(df):,} records")
            except Exception as e:
                print(f"   ❌ Error loading {file.name}: {e}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            print(f"   ✅ Total demographic records: {len(combined):,}")
            return combined

        return pd.DataFrame()

    def load_enrolment_data(self):
        """Load and combine all enrollment files"""
        if not self.enrolment_files:
            print("⚠️ No enrollment files found")
            return pd.DataFrame()

        print(f"\n📊 Loading {len(self.enrolment_files)} enrollment files...")

        dfs = []
        for file in self.enrolment_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"   ✓ Loaded {file.name}: {len(df):,} records")
            except Exception as e:
                print(f"   ❌ Error loading {file.name}: {e}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            print(f"   ✅ Total enrollment records: {len(combined):,}")
            return combined

        return pd.DataFrame()

    def merge_datasets(self, bio_df, demo_df, enrol_df):
        """Intelligently merge all datasets"""
        print("\n🔄 Merging datasets...")

        # Start with the largest dataset
        datasets = [
            ('biometric', bio_df),
            ('demographic', demo_df),
            ('enrollment', enrol_df)
        ]

        # Filter out empty dataframes
        datasets = [(name, df) for name, df in datasets if not df.empty]

        if not datasets:
            print("❌ No data to merge!")
            return pd.DataFrame()

        # Sort by size (largest first)
        datasets.sort(key=lambda x: len(x[1]), reverse=True)

        print(f"   Starting with {datasets[0][0]} data: {len(datasets[0][1]):,} records")
        merged = datasets[0][1].copy()

        # Merge other datasets
        for i in range(1, len(datasets)):
            name, df = datasets[i]
            print(f"   Merging {name} data: {len(df):,} records")

            # Find common columns for merging
            common_cols = list(set(merged.columns) & set(df.columns))

            if common_cols:
                # Use common columns as key
                merge_key = common_cols[0] if len(common_cols) == 1 else common_cols[:2]
                print(f"      Using merge key: {merge_key}")

                merged = merged.merge(df, on=merge_key, how='outer', suffixes=('', f'_{name}'))
            else:
                # No common columns, concatenate side by side
                print(f"      No common columns, adding as new features")

                # Ensure same number of rows (truncate or pad)
                if len(df) < len(merged):
                    df = pd.concat([df, pd.DataFrame(index=range(len(merged) - len(df)))], ignore_index=True)
                elif len(df) > len(merged):
                    df = df.iloc[:len(merged)]

                # Add suffix to all columns
                df.columns = [f"{col}_{name}" for col in df.columns]
                merged = pd.concat([merged, df], axis=1)

        print(f"   ✅ Merged dataset: {len(merged):,} records, {len(merged.columns)} columns")

        return merged

    def prepare_for_ml(self, df):
        """Prepare merged data for ML models"""
        print("\n⚙️ Preparing data for ML models...")

        # Ensure required columns exist or create them
        required_base_cols = ['date', 'state', 'district', 'pincode', 'bio_age_5_17', 'bio_age_17_']

        # Check what we have
        existing_cols = df.columns.tolist()
        print(f"   Current columns: {len(existing_cols)}")

        # Smart column mapping
        for req_col in required_base_cols:
            if req_col not in df.columns:
                # Try to find similar columns
                similar = [col for col in existing_cols if req_col.lower() in col.lower()]

                if similar:
                    print(f"   📌 Mapping {similar[0]} → {req_col}")
                    df[req_col] = df[similar[0]]
                else:
                    # Create dummy column based on type
                    if req_col == 'date':
                        df[req_col] = datetime.now().strftime('%d-%m-%Y')
                    elif req_col in ['state', 'district']:
                        df[req_col] = 'Unknown'
                    else:
                        df[req_col] = 0
                    print(f"   ⚠️ Created placeholder for {req_col}")

        # Remove duplicates
        original_len = len(df)
        df.drop_duplicates(inplace=True)
        if len(df) < original_len:
            print(f"   ✓ Removed {original_len - len(df):,} duplicates")

        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        text_cols = df.select_dtypes(include=['object']).columns
        df[text_cols] = df[text_cols].fillna('Unknown')

        print(f"   ✅ Dataset ready: {len(df):,} records, {len(df.columns)} columns")

        return df

    def save_merged_data(self, df, filename='merged_dataset.csv'):
        """Save merged dataset"""
        output_path = self.data_folder / filename

        print(f"\n💾 Saving merged dataset to {output_path}...")

        try:
            df.to_csv(output_path, index=False)
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   ✅ Saved successfully!")
            print(f"   📊 File size: {file_size:.2f} MB")
            print(f"   📁 Location: {output_path.absolute()}")

            return str(output_path)
        except Exception as e:
            print(f"   ❌ Error saving: {e}")
            return None

    def run(self):
        """Execute complete merging pipeline"""
        print("="*80)
        print("🚀 HACKATHON DATA MERGER - STARTING")
        print("="*80)

        # Discover files
        self.discover_files()

        # Load all datasets
        bio_df = self.load_biometric_data()
        demo_df = self.load_demographic_data()
        enrol_df = self.load_enrolment_data()

        # Merge
        merged = self.merge_datasets(bio_df, demo_df, enrol_df)

        if merged.empty:
            print("\n❌ No data to process!")
            return None

        # Prepare for ML
        prepared = self.prepare_for_ml(merged)

        # Save
        output_file = self.save_merged_data(prepared)

        print("\n" + "="*80)
        print("✅ MERGING COMPLETE!")
        print("="*80)
        print(f"\n📊 Summary:")
        print(f"   • Input files: {len(self.biometric_files) + len(self.demographic_files) + len(self.enrolment_files)}")
        print(f"   • Output records: {len(prepared):,}")
        print(f"   • Output columns: {len(prepared.columns)}")
        print(f"   • Output file: {output_file}")
        print(f"\n🎯 Next Step: Use '{output_file}' in the Streamlit GUI!")

        self.merged_df = prepared
        return prepared

def main():
    """Main execution"""
    merger = HackathonDataMerger(data_folder='data')
    result = merger.run()

    if result is not None:
        print("\n" + "="*80)
        print("🎉 SUCCESS! Your data is ready!")
        print("="*80)
        print("\nNow run the GUI:")
        print("   streamlit run app.py")
        print("\nThen upload: data/merged_dataset.csv")

if __name__ == "__main__":
    main()
