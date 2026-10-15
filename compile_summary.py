import glob
import os
import pandas as pd

def compile_reports():
    files = sorted(glob.glob("benchmark_batch_*.csv"))
    if not files:
        print("No benchmark CSV files found. Please run ./run_sweep.sh first.")
        return

    consolidated_rows = []

    for file_path in files:
        # Extract batch size from filename (e.g., benchmark_batch_16.csv -> 16)
        batch_size = file_path.split("_")[-1].replace(".csv", "")
        
        try:
            df = pd.read_csv(file_path)
            # Clean column names by stripping whitespace
            df.columns = df.columns.str.strip()
            
            # Tag rows with the corresponding batch size
            df.insert(0, "Batch_Size", batch_size)
            consolidated_rows.append(df)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if consolidated_rows:
        # Concatenate all dataframes into one master dataframe
        master_df = pd.concat(consolidated_rows, ignore_index=True)
        
        output_file = "consolidated_benchmark_report.csv"
        master_df.to_csv(output_file, index=False)
        
        print(f"Consolidated report successfully saved to {output_file}")
        print(f"\nTotal rows compiled: {len(master_df)}")
        print("\nPreview of consolidated data:")
        print(master_df.head())

if __name__ == "__main__":
    compile_reports()
