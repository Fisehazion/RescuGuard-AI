"""
RescuGuard AI - Model Analytics & Graphical Telemetry Generator
---------------------------------------------------------------
Evaluates the trained machine learning pipeline against real records sampled from
data/raw/US_Accidents_March23.csv, computes loss convergence dynamics (multi-class
log-loss across estimator depths), holdout confusion matrices, and feature importances,
and renders publication-grade vector visualizations (SVG) for GitHub and the Web UI.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, log_loss, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rescu_guard.analytics")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "US_Accidents_March23.csv"
MODEL_PATH = BASE_DIR / "data" / "processed" / "model_pipeline.joblib"
DOCS_IMG_DIR = BASE_DIR / "docs" / "images"
FRONTEND_IMG_DIR = BASE_DIR / "frontend" / "assets" / "images"

DOCS_IMG_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_test_data(sample_size: int = 50000):
    logger.info(f"Loading trained model artifact from {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)
    
    numeric_features = ['Start_Lat', 'Start_Lng', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)']
    boolean_features = ['Bump', 'Crossing', 'Traffic_Signal']
    categorical_features = ['Weather_Condition', 'Sunrise_Sunset']
    target = 'Severity'
    required_cols = numeric_features + boolean_features + categorical_features + [target]
    
    logger.info(f"Sampling {sample_size:,} rows from {RAW_DATA_PATH} for evaluation...")
    df = pd.read_csv(RAW_DATA_PATH, usecols=required_cols, nrows=sample_size * 2, engine='c')
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)
    for col in boolean_features:
        df[col] = df[col].astype(bool).astype(int)
        
    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    
    X = df[numeric_features + boolean_features + categorical_features]
    y = df[target]
    
    return pipeline, X, y, df


def generate_svg_loss_curve(estimators, train_losses, val_losses, output_path: Path):
    """Generates an enterprise-styled, publication-ready SVG loss convergence curve."""
    width, height = 900, 480
    pad_left, pad_right, pad_top, pad_bottom = 80, 50, 70, 70
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    max_x = max(estimators)
    min_x = min(estimators)
    all_y = train_losses + val_losses
    min_y = max(0.0, min(all_y) * 0.9)
    max_y = max(all_y) * 1.05

    def scale_x(x):
        return pad_left + (x - min_x) / (max_x - min_x) * plot_w

    def scale_y(y):
        return pad_top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    # Gridlines
    grid_lines = []
    for step in range(5):
        val = min_y + step * (max_y - min_y) / 4
        y_pos = scale_y(val)
        grid_lines.append(f'<line x1="{pad_left}" y1="{y_pos:.1f}" x2="{width - pad_right}" y2="{y_pos:.1f}" stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>')
        grid_lines.append(f'<text x="{pad_left - 12}" y="{y_pos + 4:.1f}" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="12" text-anchor="end">{val:.3f}</text>')

    for est in [1, 20, 40, 60, 80, 100]:
        if est <= max_x:
            x_pos = scale_x(est)
            grid_lines.append(f'<line x1="{x_pos:.1f}" y1="{pad_top}" x2="{x_pos:.1f}" y2="{pad_top + plot_h}" stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>')
            grid_lines.append(f'<text x="{x_pos:.1f}" y="{pad_top + plot_h + 22}" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="12" text-anchor="middle">{est}</text>')

    # Path points
    train_pts = " ".join([f"{scale_x(x):.1f},{scale_y(y):.1f}" for x, y in zip(estimators, train_losses)])
    val_pts = " ".join([f"{scale_x(x):.1f},{scale_y(y):.1f}" for x, y in zip(estimators, val_losses)])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    <defs>
        <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#0f172a"/>
            <stop offset="100%" stop-color="#020617"/>
        </linearGradient>
    </defs>
    <rect width="{width}" height="{height}" fill="url(#bgGrad)" rx="12"/>
    <rect x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}" fill="#1e293b" fill-opacity="0.4" rx="6" stroke="#334155" stroke-width="1.5"/>
    
    <!-- Title and Subtitle -->
    <text x="{width / 2}" y="36" fill="#f8fafc" font-family="system-ui, -apple-system, sans-serif" font-size="20" font-weight="700" text-anchor="middle">Multi-Class Cross-Entropy Loss Convergence (Log-Loss vs Estimators)</text>
    <text x="{width / 2}" y="56" fill="#94a3b8" font-family="system-ui, -apple-system, sans-serif" font-size="13" text-anchor="middle">Evaluated on US Accidents Dataset (data/raw/US_Accidents_March23.csv) — Balanced Ensemble</text>

    <!-- Grid -->
    {''.join(grid_lines)}

    <!-- Axes Labels -->
    <text x="{pad_left + plot_w / 2}" y="{height - 18}" fill="#cbd5e1" font-family="system-ui, sans-serif" font-size="13" font-weight="600" text-anchor="middle">Ensemble Complexity (Number of Decision Trees / Estimators)</text>
    <text x="26" y="{pad_top + plot_h / 2}" fill="#cbd5e1" font-family="system-ui, sans-serif" font-size="13" font-weight="600" text-anchor="middle" transform="rotate(-90 26 {pad_top + plot_h / 2})">Multi-Class Cross-Entropy (Log-Loss)</text>

    <!-- Lines -->
    <polyline fill="none" stroke="#38bdf8" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{train_pts}"/>
    <polyline fill="none" stroke="#f43f5e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{val_pts}"/>

    <!-- End value callouts -->
    <circle cx="{scale_x(estimators[-1]):.1f}" cy="{scale_y(train_losses[-1]):.1f}" r="5" fill="#38bdf8"/>
    <circle cx="{scale_x(estimators[-1]):.1f}" cy="{scale_y(val_losses[-1]):.1f}" r="5" fill="#f43f5e"/>

    <!-- Legend -->
    <g transform="translate({width - pad_right - 270}, {pad_top + 16})">
        <rect width="250" height="52" fill="#0f172a" fill-opacity="0.85" rx="6" stroke="#475569" stroke-width="1"/>
        <line x1="16" y1="18" x2="44" y2="18" stroke="#38bdf8" stroke-width="3"/>
        <circle cx="30" cy="18" r="4" fill="#38bdf8"/>
        <text x="52" y="22" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="12">Training Loss: {train_losses[-1]:.3f}</text>

        <line x1="16" y1="36" x2="44" y2="36" stroke="#f43f5e" stroke-width="3"/>
        <circle cx="30" cy="36" r="4" fill="#f43f5e"/>
        <text x="52" y="40" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="12">Validation Log-Loss: {val_losses[-1]:.3f}</text>
    </g>
</svg>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    logger.info(f"Saved loss curve to {output_path}")


def generate_svg_confusion_matrix(cm: np.ndarray, labels, output_path: Path):
    """Generates a normalized confusion matrix heatmap SVG."""
    width, height = 750, 620
    pad_left, pad_top = 160, 110
    cell_size = 110
    n = len(labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cells = []
    for i in range(n):
        for j in range(n):
            val_pct = cm_norm[i, j]
            val_cnt = cm[i, j]
            x = pad_left + j * cell_size
            y = pad_top + i * cell_size
            # Color intensity
            alpha = max(0.12, val_pct)
            bg_color = f"rgba(14, 165, 233, {alpha:.2f})" if i == j else f"rgba(244, 63, 94, {alpha * 0.75:.2f})"
            txt_color = "#f8fafc" if alpha > 0.45 else "#94a3b8"

            cells.append(f"""
            <rect x="{x}" y="{y}" width="{cell_size - 4}" height="{cell_size - 4}" rx="6" fill="{bg_color}" stroke="#334155" stroke-width="1.5"/>
            <text x="{x + cell_size/2 - 2}" y="{y + cell_size/2 - 6}" fill="{txt_color}" font-family="system-ui, sans-serif" font-size="18" font-weight="700" text-anchor="middle">{val_pct * 100:.1f}%</text>
            <text x="{x + cell_size/2 - 2}" y="{y + cell_size/2 + 16}" fill="{txt_color}" fill-opacity="0.8" font-family="system-ui, sans-serif" font-size="12" text-anchor="middle">n={val_cnt:,}</text>
            """)

    # Axis labels
    axis_elements = []
    for idx, lbl in enumerate(labels):
        # Y labels (True)
        axis_elements.append(f'<text x="{pad_left - 16}" y="{pad_top + idx * cell_size + cell_size/2 + 4}" fill="#e2e8f0" font-family="system-ui, sans-serif" font-size="13" font-weight="600" text-anchor="end">{lbl}</text>')
        # X labels (Predicted)
        axis_elements.append(f'<text x="{pad_left + idx * cell_size + cell_size/2}" y="{pad_top - 14}" fill="#e2e8f0" font-family="system-ui, sans-serif" font-size="13" font-weight="600" text-anchor="middle">{lbl}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    <defs>
        <linearGradient id="bgGradCm" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#0f172a"/>
            <stop offset="100%" stop-color="#020617"/>
        </linearGradient>
    </defs>
    <rect width="{width}" height="{height}" fill="url(#bgGradCm)" rx="12"/>

    <!-- Header -->
    <text x="{width / 2}" y="36" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="20" font-weight="700" text-anchor="middle">Normalized Holdout Confusion Matrix (US Accidents Real Holdout)</text>
    <text x="{width / 2}" y="58" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="13" text-anchor="middle">Predicted vs Ground Truth Severity Tiers (Balanced Random Forest)</text>

    <!-- Axis Group Titles -->
    <text x="{pad_left + (n * cell_size) / 2}" y="88" fill="#38bdf8" font-family="system-ui, sans-serif" font-size="14" font-weight="700" text-anchor="middle">PREDICTED SEVERITY TIER</text>
    <text x="36" y="{pad_top + (n * cell_size) / 2}" fill="#38bdf8" font-family="system-ui, sans-serif" font-size="14" font-weight="700" text-anchor="middle" transform="rotate(-90 36 {pad_top + (n * cell_size) / 2})">GROUND TRUTH SEVERITY</text>

    <!-- Cells & Labels -->
    {''.join(cells)}
    {''.join(axis_elements)}

    <!-- Footnote -->
    <text x="{width / 2}" y="{height - 24}" fill="#64748b" font-family="system-ui, sans-serif" font-size="12" text-anchor="middle">Diagonal cells represent true positive recall across severity tiers 1 to 4.</text>
</svg>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    logger.info(f"Saved confusion matrix to {output_path}")


def generate_svg_feature_importance(importances_dict, output_path: Path):
    """Generates a horizontal bar chart of feature importances."""
    width, height = 850, 480
    pad_left, pad_top, pad_right = 210, 80, 60
    bar_h = 28
    gap = 14

    sorted_feats = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)[:9]
    max_val = max(v for _, v in sorted_feats)

    bars = []
    for idx, (feat, val) in enumerate(sorted_feats):
        y = pad_top + idx * (bar_h + gap)
        w = (val / max_val) * (width - pad_left - pad_right - 60)
        pct = val * 100

        bars.append(f"""
        <text x="{pad_left - 14}" y="{y + bar_h / 2 + 5}" fill="#e2e8f0" font-family="system-ui, sans-serif" font-size="13" font-weight="500" text-anchor="end">{feat}</text>
        <rect x="{pad_left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" fill="#3b82f6" fill-opacity="0.85" stroke="#60a5fa" stroke-width="1"/>
        <text x="{pad_left + w + 10:.1f}" y="{y + bar_h / 2 + 5}" fill="#93c5fd" font-family="system-ui, sans-serif" font-size="12" font-weight="600">{pct:.1f}%</text>
        """)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    <defs>
        <linearGradient id="bgGradFi" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#0f172a"/>
            <stop offset="100%" stop-color="#020617"/>
        </linearGradient>
    </defs>
    <rect width="{width}" height="{height}" fill="url(#bgGradFi)" rx="12"/>

    <text x="{width / 2}" y="36" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="20" font-weight="700" text-anchor="middle">Feature Attribution &amp; Gini Impurity Importance</text>
    <text x="{width / 2}" y="56" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="13" text-anchor="middle">Relative Impact on Roadway Incident Severity Classification</text>

    {''.join(bars)}
</svg>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    logger.info(f"Saved feature importance to {output_path}")


def generate_svg_class_distribution(dist_dict, output_path: Path):
    """Generates an accident severity distribution chart."""
    width, height = 800, 420
    pad_left, pad_top, pad_bottom = 80, 80, 60
    chart_w = width - pad_left - 60
    chart_h = height - pad_top - pad_bottom

    total = sum(dist_dict.values())
    max_val = max(dist_dict.values())

    bars = []
    n = len(dist_dict)
    bar_width = chart_w / (n * 1.8)
    colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]

    for idx, (label, count) in enumerate(dist_dict.items()):
        x = pad_left + idx * (chart_w / n) + 30
        pct = (count / total) * 100
        bar_h = (count / max_val) * (chart_h - 40)
        y = pad_top + chart_h - bar_h
        col = colors[idx % len(colors)]

        bars.append(f"""
        <rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_h:.1f}" fill="{col}" rx="6" fill-opacity="0.85" stroke="#ffffff" stroke-opacity="0.2"/>
        <text x="{x + bar_width/2}" y="{y - 10:.1f}" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="14" font-weight="700" text-anchor="middle">{pct:.1f}%</text>
        <text x="{x + bar_width/2}" y="{y - 28:.1f}" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="11" text-anchor="middle">({count:,})</text>
        <text x="{x + bar_width/2}" y="{pad_top + chart_h + 24}" fill="#e2e8f0" font-family="system-ui, sans-serif" font-size="13" font-weight="600" text-anchor="middle">{label}</text>
        """)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    <defs>
        <linearGradient id="bgGradDist" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#0f172a"/>
            <stop offset="100%" stop-color="#020617"/>
        </linearGradient>
    </defs>
    <rect width="{width}" height="{height}" fill="url(#bgGradDist)" rx="12"/>

    <text x="{width / 2}" y="36" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="20" font-weight="700" text-anchor="middle">Severity Class Distribution (US Accidents Benchmark)</text>
    <text x="{width / 2}" y="56" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="13" text-anchor="middle">Ground Truth Incident Distribution Across Severity Tiers (Sampled Records)</text>

    <!-- Baseline axis -->
    <line x1="{pad_left}" y1="{pad_top + chart_h}" x2="{pad_left + chart_w}" y2="{pad_top + chart_h}" stroke="#475569" stroke-width="1.5"/>

    {''.join(bars)}
</svg>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    logger.info(f"Saved class distribution to {output_path}")


def main():
    pipeline, X, y, df = load_model_and_test_data(sample_size=30000)

    # Class distribution
    dist_counts = df['Severity'].value_counts().sort_index().to_dict()
    dist_dict = {f"Severity {k}": v for k, v in dist_counts.items()}

    # Train / test split for holdout evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Predict with trained pipeline
    logger.info("Computing predictions on holdout evaluation split...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    
    classes = pipeline.classes_
    labels = [f"Severity {c}" for c in classes]

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)

    # Loss curve across estimators
    # In Random Forest with 100 trees, simulate step-wise cross-entropy loss convergence
    logger.info("Computing multi-class log-loss convergence across estimators...")
    rf_clf = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    X_test_trans = preprocessor.transform(X_test)
    X_train_trans = preprocessor.transform(X_train[:5000])

    estimators_steps = [1, 5, 10, 20, 35, 50, 75, 100]
    val_losses = []
    train_losses = []

    # Map classes to indices
    y_test_arr = np.array(y_test)
    y_train_sample = np.array(y_train[:5000])

    # Accumulate tree probabilities
    all_trees = rf_clf.estimators_
    total_trees = len(all_trees)

    for est_count in estimators_steps:
        sub_trees = all_trees[:min(est_count, total_trees)]
        # Average probability across sub_trees
        test_probs = np.mean([t.predict_proba(X_test_trans) for t in sub_trees], axis=0)
        train_probs = np.mean([t.predict_proba(X_train_trans) for t in sub_trees], axis=0)

        # Clip for numerical stability
        test_probs = np.clip(test_probs, 1e-6, 1 - 1e-6)
        train_probs = np.clip(train_probs, 1e-6, 1 - 1e-6)
        test_probs /= test_probs.sum(axis=1, keepdims=True)
        train_probs /= train_probs.sum(axis=1, keepdims=True)

        v_loss = log_loss(y_test, test_probs, labels=classes)
        t_loss = log_loss(y_train_sample, train_probs, labels=classes)

        val_losses.append(round(float(v_loss), 4))
        train_losses.append(round(float(t_loss), 4))

    # Feature importances
    num_cols = ['Start_Lat', 'Start_Lng', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Bump', 'Crossing', 'Traffic_Signal']
    try:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(['Weather_Condition', 'Sunrise_Sunset']).tolist()
    except Exception:
        cat_feature_names = ['Weather_Condition_Enc', 'Sunrise_Sunset_Enc']

    all_feat_names = num_cols + cat_feature_names
    raw_fi = rf_clf.feature_importances_

    # Aggregate back to high-level features
    fi_dict = {
        "Atmospheric Visibility": float(raw_fi[4]),
        "Spatial Coordinates (Lat/Lng)": float(raw_fi[0] + raw_fi[1]) / 2,
        "Ambient Temperature": float(raw_fi[2]),
        "Relative Humidity": float(raw_fi[3]),
        "Weather Condition": float(sum(raw_fi[len(num_cols):-2])),
        "Traffic Signal": float(raw_fi[7]),
        "Pedestrian Crossing": float(raw_fi[6]),
        "Solar Period (Day/Night)": float(sum(raw_fi[-2:])),
        "Speed Bump / Deflection": float(raw_fi[5])
    }
    # Normalize
    fi_total = sum(fi_dict.values())
    fi_dict = {k: round(v / fi_total, 4) for k, v in fi_dict.items()}

    # Save summary metrics JSON
    summary_metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "final_train_log_loss": train_losses[-1],
        "final_val_log_loss": val_losses[-1],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importances": fi_dict,
        "class_distribution": dist_dict
    }

    metrics_out = BASE_DIR / "data" / "processed" / "evaluation_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_out}")

    # Generate SVGs into docs/images and frontend/assets/images
    for target_dir in [DOCS_IMG_DIR, FRONTEND_IMG_DIR]:
        generate_svg_loss_curve(estimators_steps, train_losses, val_losses, target_dir / "loss_convergence_curve.svg")
        generate_svg_confusion_matrix(cm, labels, target_dir / "confusion_matrix.svg")
        generate_svg_feature_importance(fi_dict, target_dir / "feature_importance.svg")
        generate_svg_class_distribution(dist_dict, target_dir / "class_distribution.svg")

    logger.info("All model visualization graphs successfully rendered!")


if __name__ == "__main__":
    main()
