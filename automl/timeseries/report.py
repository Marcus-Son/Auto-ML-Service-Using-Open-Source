# automl/timeseries/report.py

import os

def save_report_html(
    df, target, time_col, task, eda_logs, preprocessing_logs,
    leaderboard_df, best_model_name, best_params,
    test_metrics, error_samples, xai_summary,
    file_path="AutoML_Timeseries_Report.html"
):
    html_content = f"""
    <html><head><meta charset='utf-8'>
    <style>
    body {{font-family:'Pretendard','Arial',sans-serif;background:#f6f8fa;}}
    .report-section {{background:#F8FAFC; border-radius:18px; box-shadow:0 2px 8px #E0E7EF; padding:30px 32px 24px 32px; margin-bottom:32px;}}
    .report-title {{font-size:2.3rem;font-weight:700;letter-spacing:-0.02em; color:#305784; margin-bottom:1.3rem;}}
    .report-sub {{font-size:1.3rem;font-weight:600;color:#1B334B;}}
    .metric-card {{display:inline-block; background:#fff; margin: 0 16px 16px 0; border-radius:14px; padding:16px 22px; box-shadow:0 2px 7px #E7EEF6;}}
    .metric-title {{font-size:1rem; color:#666;}}
    .metric-value {{font-size:1.2rem; font-weight:700; color:#3854AF;}}
    </style>
    </head><body>
    <div class="report-title">🔮 AutoML Time Series Report</div>
    <div class="report-section">
    <span class="report-sub">📂 Project Overview</span>
    <ul>
        <li><b>Task:</b> {task}</li>
        <li><b>Target:</b> {target}</li>
        <li><b>Time Column:</b> {time_col}</li>
        <li><b>Rows × Cols:</b> {df.shape[0]:,} × {df.shape[1]}</li>
    </ul>
    </div>
    <div class="report-section">
    <span class="report-sub">🔍 EDA & Data Summary</span>
    <ul>
    {''.join([f'<li>{l}</li>' for l in eda_logs])}
    </ul>
    </div>
    <div class="report-section">
    <span class="report-sub">🛠️ Preprocessing Steps</span>
    <ul>
    {''.join([f'<li>{l}</li>' for l in preprocessing_logs])}
    </ul>
    </div>
    <div class="report-section">
    <span class="report-sub">🤖 Model Search & Leaderboard</span>
    {leaderboard_df.to_html(index=False)}
    <div class='metric-card'><span class='metric-title'>Best Model</span><br><span class='metric-value'>{best_model_name}</span></div>
    <div class='metric-card'><span class='metric-title'>Best Params</span><br><span class='metric-value'>{best_params}</span></div>
    </div>
    <div class="report-section">
    <span class="report-sub">📊 Test Set Metrics</span><br>
    {''.join([f"<div class='metric-card'><span class='metric-title'>{k}</span><br><span class='metric-value'>{v:.4f}</span></div>" for k, v in test_metrics.items()])}
    </div>
    <div class="report-section">
    <span class="report-sub">🚨 Top 5 Largest Errors (시점)</span>
    {error_samples.to_html(index=False)}
    </div>
    <div class="report-section">
    <span class="report-sub">🌈 XAI Interpretation Highlights</span>
    <ul>
    {''.join([f'<li>{l}</li>' for l in xai_summary])}
    </ul>
    </div>
    </body></html>
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return os.path.abspath(file_path)