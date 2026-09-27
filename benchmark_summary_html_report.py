import os
import json
import pandas as pd

def generate_benchmark_html(csv_file="sla_benchmark_results.csv", output_html="benchmark_report.html"):
    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found. Run perf_analyzer first.")
        return

    # Load and clean dataset
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    # Detect dynamic columns
    concurrency_col = [col for col in df.columns if "concurrency" in col.lower()][0]
    fps_col = [col for col in df.columns if any(k in col.lower() for k in ["throughput", "infer", "fps"])][0]
    p99_col = [col for col in col_list if "p99" in col.lower()] if (col_list := list(df.columns)) else []
    p99_col = p99_col[0] if p99_col else [col for col in df.columns if "latency" in col.lower()][0]

    # Convert latency from usec to ms if necessary
    if "usec" in p99_col.lower() or df[p99_col].mean() > 500:
        df["p99_ms"] = df[p99_col] / 1000.0
    else:
        df["p99_ms"] = df[p99_col]

    df = df.sort_values(by=concurrency_col)

    # Compute key performance metrics
    max_fps = float(df[fps_col].max())
    min_p99 = float(df["p99_ms"].min())
    max_p99 = float(df["p99_ms"].max())
    optimal_row = df.loc[df[fps_col].idxmax()]
    optimal_concurrency = int(optimal_row[concurrency_col])
    sla_limit = 50.0

    # Determine SLA compliance
    df["sla_pass"] = df["p99_ms"] <= sla_limit

    # Prepare data arrays for Chart.js
    labels_js = json.dumps(df[concurrency_col].astype(int).tolist())
    fps_js = json.dumps(df[fps_col].round(2).tolist())
    p99_js = json.dumps(df["p99_ms"].round(2).tolist())

    # Build HTML table rows
    table_rows = ""
    for _, row in df.iterrows():
        c_val = int(row[concurrency_col])
        fps_val = round(row[fps_col], 2)
        p99_val = round(row["p99_ms"], 2)
        status_badge = (
            '<span class="badge pass">PASS</span>' if row["sla_pass"] else '<span class="badge fail">FAIL</span>'
        )
        table_rows += f"""
        <tr>
            <td>{c_val}</td>
            <td>{fps_val:,}</td>
            <td>{p99_val:.2f} ms</td>
            <td>{status_badge}</td>
        </tr>
        """

    # Self-contained HTML report template with CDN Chart.js
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Triton Benchmark Report — vector_vision_general_v1</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-red: #f87171;
            --accent-green: #4ade80;
            --border-color: #334155;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 24px;
            color: var(--accent-blue);
        }}
        .header p {{
            margin: 0;
            color: var(--text-muted);
            font-size: 14px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .kpi-title {{
            font-size: 12px;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: bold;
        }}
        .chart-card {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            margin-bottom: 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
            font-size: 14px;
        }}
        th {{
            background-color: #0f172a;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 12px;
        }}
        .badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }}
        .badge.pass {{
            background-color: rgba(74, 222, 128, 0.15);
            color: var(--accent-green);
        }}
        .badge.fail {{
            background-color: rgba(248, 113, 113, 0.15);
            color: var(--accent-red);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Triton Inference Benchmark Report</h1>
            <p>Model: <strong>vector_vision_general_v1</strong> | Endpoint: <strong>http://localhost:8005</strong></p>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Peak Throughput</div>
                <div class="kpi-value" style="color: var(--accent-blue);">{max_fps:,.1f} <span style="font-size: 14px;">FPS</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Optimal Concurrency</div>
                <div class="kpi-value">c = {optimal_concurrency}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Min P99 Latency</div>
                <div class="kpi-value">{min_p99:.2f} <span style="font-size: 14px;">ms</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Max P99 Latency</div>
                <div class="kpi-value" style="color: {'var(--accent-red)' if max_p99 > sla_limit else 'var(--accent-green)'};">{max_p99:.2f} <span style="font-size: 14px;">ms</span></div>
            </div>
        </div>

        <div class="chart-card">
            <canvas id="benchmarkChart" height="110"></canvas>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Concurrency Level (c)</th>
                    <th>Throughput (FPS)</th>
                    <th>P99 Latency (ms)</th>
                    <th>SLA Status (&le; {sla_limit:.0f}ms)</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>

    <script>
        const ctx = document.getElementById('benchmarkChart').getContext('2d');
        const labels = {labels_js};
        const fpsData = {fps_js};
        const p99Data = {p99_js};

        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'Throughput (FPS)',
                        data: fpsData,
                        borderColor: '#38bdf8',
                        backgroundColor: '#38bdf8',
                        yAxisID: 'y_fps',
                        tension: 0.2,
                        pointRadius: 5
                    }},
                    {{
                        label: 'P99 Latency (ms)',
                        data: p99Data,
                        borderColor: '#f87171',
                        backgroundColor: '#f87171',
                        borderDash: [5, 5],
                        yAxisID: 'y_latency',
                        tension: 0.2,
                        pointRadius: 5
                    }}
                ]
            }},
            options: {{
                responsive: true,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Concurrency Level (c)', color: '#94a3b8' }},
                        grid: {{ color: '#334155' }},
                        ticks: {{ color: '#f8fafc' }}
                    }},
                    y_fps: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{ display: true, text: 'Throughput (Inferences / sec)', color: '#38bdf8' }},
                        grid: {{ color: '#334155' }},
                        ticks: {{ color: '#38bdf8' }}
                    }},
                    y_latency: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{ display: true, text: 'P99 Latency (ms)', color: '#f87171' }},
                        grid: {{ drawOnChartArea: false }},
                        ticks: {{ color: '#f87171' }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        labels: {{ color: '#f8fafc' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML Benchmark report generated successfully: {output_html}")

if __name__ == "__main__":
    generate_benchmark_html()