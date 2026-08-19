import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

METRICS = [
    'dice', 'mcc', 'cldice', 'fragmentation_ratio',
    'skeleton_component_connectivity', 'largest_gt_recall',
    'bifurcation_f1', 'n_components_pred',
]


def plot_metrics_vs_min_size(df, output_dir, metrics=None, filename="metrics_vs_min_size.png"):
    metrics = metrics or [m for m in METRICS if m in df.columns]
    n = len(metrics)
    ncols = 3
    nrows = -(-n // ncols)  # ceil

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    operators = sorted(df['operator'].unique())
    min_sizes = sorted(df['min_size'].unique())

    for idx, metric in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        for op in operators:
            sub = df[df.operator == op]
            means = [sub[sub.min_size == ms][metric].mean() for ms in min_sizes]
            stds = [sub[sub.min_size == ms][metric].std() for ms in min_sizes]
            means, stds = np.array(means), np.array(stds)
            ax.plot(min_sizes, means, marker='o', label=op, linewidth=1.5)
            ax.fill_between(min_sizes, means - stds, means + stds, alpha=0.12)
        ax.set_title(metric)
        ax.set_xlabel("min_size")
        ax.grid(alpha=0.3)

    # une seule légende partagée
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(operators),
               bbox_to_anchor=(0.5, -0.02))

    # désactiver les axes vides
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = Path(output_dir) / filename
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure exportée : {out_path}")
    return out_path