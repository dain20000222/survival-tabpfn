import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
METRICS_PATH = "metrics.csv"
OUTPUT_DIR = "plots_by_metric"

# models as stored in metrics.csv  -> pretty labels
MODEL_LABELS = {
    "deephit": "Dynamic DeepHit",
    "tabpfn": "Dynamic TabPFN",
    "landmark_cox": "Landmark Cox",
}
MODELS = list(MODEL_LABELS.keys())          # order in plots
METRICS = ["brier", "auc", "cindex"]        # metrics to plot


# ---------------------------------------------------------
# Load metrics
# ---------------------------------------------------------
df = pd.read_csv(METRICS_PATH)

# only our models
df = df[df["model"].isin(MODELS)].copy()

# drop NaN values
df = df[~df["value"].isna()].copy()


# ---------------------------------------------------------
# Find datasets common to all models AND all metrics
# ---------------------------------------------------------
def datasets_for_metric(metric_name: str):
    sub = df[df["metric"] == metric_name]
    # intersection across models
    return set.intersection(
        *[set(sub[sub["model"] == m]["dataset"].unique()) for m in MODELS]
    )

common_datasets_per_metric = [datasets_for_metric(m) for m in METRICS]
common_datasets = sorted(set.intersection(*common_datasets_per_metric))

print("Common datasets across all models & metrics:")
for ds in common_datasets:
    print("  -", ds)

if not common_datasets:
    print("No common datasets with all metrics for all models. Nothing to plot.")
    raise SystemExit

# restrict df to common datasets only
df = df[df["dataset"].isin(common_datasets)].copy()

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Helper: grouped bar plot for Brier / AUC for one dataset
# ---------------------------------------------------------
def plot_brier_or_auc(df_metric, metric_name, dname, out_dir):
    """
    df_metric: rows for a single dataset & metric ('brier' or 'auc')
    """
    df_metric = df_metric.copy()
    df_metric["interval"] = list(zip(df_metric["tau_start"], df_metric["tau_end"]))
    intervals = sorted(df_metric["interval"].unique())  # (start, end)

    x = np.arange(len(intervals))
    bar_width = 0.20
    n_models = len(MODELS)
    offset_start = -bar_width * (n_models - 1) / 2.0

    plt.figure(figsize=(10, 6))

    for i, model in enumerate(MODELS):
        df_m = df_metric[df_metric["model"] == model].copy()
        if df_m.empty:
            continue

        df_m = df_m.set_index("interval").reindex(intervals)
        y_vals = df_m["value"].to_numpy(dtype=float)

        x_pos = x + offset_start + i * bar_width
        bars = plt.bar(x_pos, y_vals, width=bar_width, label=MODEL_LABELS[model])

        # Add labels
        for bar, val in zip(bars, y_vals):
            if np.isnan(val):
                continue
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # x tick labels: BS(t_k | t_{k-1}) or AUC(t_k | t_{k-1})
    if metric_name == "brier":
        x_labels = [f"BS({e:g}|{s:g})" for (s, e) in intervals]
        ylabel = "Brier Score"
        title_metric = "Brier Score"
        file_prefix = "brier"
    else:  # auc
        x_labels = [f"AUC({e:g}|{s:g})" for (s, e) in intervals]
        ylabel = "AUC"
        title_metric = "AUC"
        file_prefix = "auc"

    plt.xticks(x, x_labels, rotation=0)
    plt.ylim(0.0, 1.0)
    plt.ylabel(ylabel)
    plt.xlabel(f"{title_metric} at successive time points")
    plt.title(f"{title_metric} by Interval - {dname}")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{file_prefix}_{dname}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {metric_name} plot for dataset '{dname}' to {out_path}")


# ---------------------------------------------------------
# Helper: grouped bar plot for c-index for one dataset
# ---------------------------------------------------------
def plot_cindex(df_metric, dname, out_dir):
    """
    df_metric: rows for a single dataset & metric 'cindex'
    """
    df_metric = df_metric.rename(columns={"tau_start": "tau"}).copy()
    taus = sorted(df_metric["tau"].unique())

    x = np.arange(len(taus))
    bar_width = 0.20
    n_models = len(MODELS)
    offset_start = -bar_width * (n_models - 1) / 2.0

    plt.figure(figsize=(10, 6))

    for i, model in enumerate(MODELS):
        df_m = df_metric[df_metric["model"] == model].copy()
        if df_m.empty:
            continue

        df_m = df_m.set_index("tau").reindex(taus)
        y_vals = df_m["value"].to_numpy(dtype=float)

        x_pos = x + offset_start + i * bar_width
        bars = plt.bar(x_pos, y_vals, width=bar_width, label=MODEL_LABELS[model])

        for bar, val in zip(bars, y_vals):
            if np.isnan(val):
                continue
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    x_labels = [f"C({t:g})" for t in taus]
    plt.xticks(x, x_labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("C-index")
    plt.xlabel("Landmark Time")
    plt.title(f"C-index by Time Point - {dname}")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"cindex_{dname}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved cindex plot for dataset '{dname}' to {out_path}")


# ---------------------------------------------------------
# Dataset-wise plots (bars)
# ---------------------------------------------------------
for dname in common_datasets:
    df_ds = df[df["dataset"] == dname].copy()

    # Brier
    df_brier = df_ds[df_ds["metric"] == "brier"].copy()
    plot_brier_or_auc(df_brier, "brier", dname, OUTPUT_DIR)

    # AUC
    df_auc = df_ds[df_ds["metric"] == "auc"].copy()
    plot_brier_or_auc(df_auc, "auc", dname, OUTPUT_DIR)

    # C-index
    df_cidx = df_ds[df_ds["metric"] == "cindex"].copy()
    plot_cindex(df_cidx, dname, OUTPUT_DIR)

# ---------------------------------------------------------
# Boxplots: distribution of each metric by model (blue fill only)
# ---------------------------------------------------------
for metric in METRICS:
    df_metric = df[(df["metric"] == metric)].copy()
    if df_metric.empty:
        continue

    # build list of values per model
    data_by_model = []
    labels = []
    for m in MODELS:
        vals = df_metric[df_metric["model"] == m]["value"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        data_by_model.append(vals)
        labels.append(MODEL_LABELS[m])

    if not data_by_model:
        continue

    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(
        data_by_model,
        labels=labels,
        patch_artist=True,    # allows coloring the inside
        showfliers=True
    )

    # --- Blue fill only, black outlines ---
    for box in bp['boxes']:
        box.set(facecolor='#1f77b4', edgecolor='black', linewidth=1.3)

    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.3)

    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.3)

    for median in bp['medians']:
        median.set(color='black', linewidth=1.6)

    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='#1f77b4',
                  markeredgecolor='black', alpha=0.6, markersize=5)

    plt.xticks(rotation=25, ha="right")

    # titles
    if metric == "brier":
        ylabel = "Brier Score"
        title = "Brier Score Distribution by Model"
        fname = "brier_boxplot.png"
    elif metric == "auc":
        ylabel = "AUC"
        title = "AUC Distribution by Model"
        fname = "auc_boxplot.png"
    else:  # cindex
        ylabel = "C-index"
        title = "C-index Distribution by Model"
        fname = "cindex_boxplot.png"

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {metric} boxplot to {out_path}")
