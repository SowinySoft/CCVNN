import json
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

json_path = os.getenv("BENCHMARK_RESULTS_PATH", "benchmark_results.json")

if not os.path.exists(json_path):
    print(
        f"[ERROR] '{json_path}' not found. Run benchmark or stream test first to generate telemetry data."
    )
    sys.exit(1)

with open(json_path, "r") as f:
    data = json.load(f)

totals = data.get("frame_totals_us", [])
totals_ms = [t / 1000.0 for t in totals] if totals else []
summary = data.get("summary_ms", {})

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Stage Latency Percentiles (ms)",
        "Frame-by-Frame Glass-to-Glass Telemetry",
    ),
)

stages = list(summary.keys())
p50 = [summary[s].get("p50", 0.0) for s in stages]
p99 = [summary[s].get("p99", 0.0) for s in stages]

fig.add_trace(
    go.Bar(
        name="p50 Latency",
        x=stages,
        y=p50,
        marker_color="#00CC96",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        name="p99 Latency",
        x=stages,
        y=p99,
        marker_color="#EF553B",
    ),
    row=1,
    col=1,
)

if totals_ms:
    fig.add_trace(
        go.Scatter(
            y=totals_ms,
            mode="lines+markers",
            name="Glass-to-Glass (ms)",
            line=dict(color="#636EFA", width=2),
        ),
        row=1,
        col=2,
    )
else:
    # Fallback scatter point when raw frame arrays are omitted in summary mode
    fig.add_trace(
        go.Scatter(
            x=["p50", "p99"],
            y=[summary.get("total_glass_to_glass", {}).get("p50", 0.0),
               summary.get("total_glass_to_glass", {}).get("p99", 0.0)],
            mode="markers+text",
            name="Summary SLA Marks",
            marker=dict(size=12, color="#FFA15A"),
        ),
        row=1,
        col=2,
    )

fig.update_layout(
    title_text="⚡ CCVNN V14 Edge Inspection Engine - Performance Dashboard",
    template="plotly_dark",
    barmode="group",
    margin=dict(l=40, r=40, t=60, b=40),
)

os.makedirs("public", exist_ok=True)
fig.write_html("public/index.html")
print("[✓] Dashboard successfully generated at public/index.html")