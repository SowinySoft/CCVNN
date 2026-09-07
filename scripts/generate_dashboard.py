import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

json_path = 'benchmark_results.json'

if not os.path.exists(json_path):
    raise FileNotFoundError(f"'{json_path}' not found. Run the C++ profiler binary first to generate telemetry data.")

with open(json_path) as f:
    data = json.load(f)

totals = data.get('frame_totals_us', [])
totals_ms = [t / 1000.0 for t in totals]
summary = data.get('summary_ms', {})

fig = make_subplots(rows=1, cols=2, subplot_titles=("Stage Latency Percentiles (ms)", "Frame-by-Frame Telemetry"))

stages = list(summary.keys())
p50 = [summary[s]['p50'] for s in stages]
p99 = [summary[s]['p99'] for s in stages]

fig.add_trace(go.Bar(name='p50 Latency', x=stages, y=p50), row=1, col=1)
fig.add_trace(go.Bar(name='p99 Latency', x=stages, y=p99), row=1, col=1)
fig.add_trace(go.Scatter(y=totals_ms, mode='lines+markers', name='Glass-to-Glass Latency (ms)'), row=1, col=2)

fig.update_layout(title_text="🚀 CCVNN Edge Inspection Engine - Interactive Performance Dashboard", template="plotly_dark")

os.makedirs("public", exist_ok=True)
fig.write_html("public/index.html")
print("[✓] Dashboard successfully generated at public/index.html")
