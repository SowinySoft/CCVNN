#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_BASE = "ccvnn_results.csv"
CSV_V20 = "ccvnn_v20_results.csv"
SUMMARY_CSV = "combined_latency_summary_capped.csv"
PLOT_IMAGE = "percentile_latency_breakdown.png"

def find_col(df, target_keywords):
    for col in df.columns:
        col_clean = col.lower().strip()
        if all(kw in col_clean for kw in target_keywords):
            return col
    return None

def normalize_to_ms(series, col_name):
    # Convert microseconds (usec or values > 1000) to milliseconds
    if "usec" in col_name.lower() or "us" in col_name.lower() or series.mean() > 500:
        return series / 1000.0
    return series

def load_and_parse(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing required CSV input file: {csv_path}")

    df = pd.read_csv(csv_path)

    concurrency_col = find_col(df, ["concurrency"]) or df.columns[0]
    p50_col = find_col(df, ["p50"]) or find_col(df, ["50"])
    p90_col = find_col(df, ["p90"]) or find_col(df, ["90"])
    p99_col = find_col(df, ["p99"]) or find_col(df, ["99"])
    server_col = find_col(df, ["server", "latency"]) or find_col(df, ["avg", "latency"]) or p99_col

    if not all([p50_col, p90_col, p99_col]):
        raise ValueError(f"Could not locate p50, p90, and p99 columns in {csv_path}. Columns found: {list(df.columns)}")

    parsed = pd.DataFrame({
        "Concurrency": df[concurrency_col],
        "p50_ms": normalize_to_ms(df[p50_col], p50_col),
        "p90_ms": normalize_to_ms(df[p90_col], p90_col),
        "p99_ms": normalize_to_ms(df[p99_col], p99_col),
        "Server_Total_ms": normalize_to_ms(df[server_col], server_col)
    })
    return parsed

def main():
    print(f"[INFO] Loading raw benchmark data from {CSV_BASE} and {CSV_V20}...")
    df_base = load_and_parse(CSV_BASE)
    df_v20 = load_and_parse(CSV_V20)

    # Merge summaries
    merged = pd.merge(
        df_base, df_v20,
        on="Concurrency",
        suffixes=("_Baseline", "_V20")
    )

    summary_df = pd.DataFrame({
        "Concurrency": merged["Concurrency"],
        "Baseline_p50_ms": merged["p50_ms_Baseline"],
        "Baseline_p90_ms": merged["p90_ms_Baseline"],
        "Baseline_p99_ms": merged["p99_ms_Baseline"],
        "Baseline_Total_Server_ms": merged["Server_Total_ms_Baseline"],
        "V20_p50_ms": merged["p50_ms_V20"],
        "V20_p90_ms": merged["p90_ms_V20"],
        "V20_p99_ms": merged["p99_ms_V20"],
        "V20_Total_Server_ms": merged["Server_Total_ms_V20"],
    })

    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"[SUCCESS] Saved summary metrics to '{SUMMARY_CSV}'.")
    
    # Render Stacked Bar Chart (p50, p90 differential, p99 differential)
    concurrencies = summary_df["Concurrency"].astype(str).tolist()
    x = np.arange(len(concurrencies))
    width = 0.35

    # Compute differentials for stacked visual representation
    b_p50 = summary_df["Baseline_p50_ms"]
    b_p90_diff = summary_df["Baseline_p90_ms"] - b_p50
    b_p99_diff = summary_df["Baseline_p99_ms"] - summary_df["Baseline_p90_ms"]

    v_p50 = summary_df["V20_p50_ms"]
    v_p90_diff = summary_df["V20_p90_ms"] - v_p50
    v_p99_diff = summary_df["V20_p99_ms"] - summary_df["V20_p90_ms"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline Stacked Bars
    ax.bar(x - width/2, b_p50, width, label="Baseline p50", color="#1f77b4")
    ax.bar(x - width/2, b_p90_diff, width, bottom=b_p50, label="Baseline p90 (Delta)", color="#6baed6")
    ax.bar(x - width/2, b_p99_diff, width, bottom=summary_df["Baseline_p90_ms"], label="Baseline p99 (Delta)", color="#c6dbef")

    # V20 Stacked Bars
    ax.bar(x + width/2, v_p50, width, label="V20 p50", color="#2ca02c")
    ax.bar(x + width/2, v_p90_diff, width, bottom=v_p50, label="V20 p90 (Delta)", color="#74c476")
    ax.bar(x + width/2, v_p99_diff, width, bottom=summary_df["V20_p90_ms"], label="V20 p99 (Delta)", color="#c7e9c0")

    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Request Concurrency")
    ax.set_title("Percentile Latency Comparison (p50 / p90 / p99): Baseline vs V20")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Concurrency {c}" for c in concurrencies])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOT_IMAGE, dpi=300)
    print(f"[SUCCESS] Benchmark visualization plot saved to '{PLOT_IMAGE}'.")

if __name__ == "__main__":
    main()