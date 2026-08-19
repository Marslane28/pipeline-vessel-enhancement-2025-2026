"""

Dernier maillon de la chaîne de benchmark GT analytique :
  Comparaison_hessian.py → erreur sur la Hessienne brute
  Comparaison_valeurs_propres.py → erreur sur les valeurs propres
  Comparaison_vesselness.py → corrélation/RMSE/F1 sur la vesselness (Enhancer)
  Comparaison_segmentation.py → erreur de SEGMENTATION vs GT (ce script)

Réutilise EXACTEMENT la vesselness calculée par Comparaison_vesselness.py
(compute_vesselness, mêmes SCALES/BLACK_RIDGES/FILTERS_TO_TEST) pour ne pas
dupliquer/dériver la logique Enhancer, puis :
  1. Segmenter().thresholding(...) seuil simple optimisé par F1 vs GT, exactement le chemin
     Processor.process_data() → segmentation_function(...).
  2. detailed_metrics(...) (core.experiments.metrics) sur les données
     orientées par polarité (même orientation que celle utilisée par
     Segmenter pour choisir le seuil) Dice, MCC, sensitivity, specificity,
     precision, accuracy, ROC-AUC, PR-AUC, clDice, métriques de composantes
     connexes (fragmentation, recall du tronc GT, overlap composante
     principale) et détection de bifurcations (BDR, precision, F1).

GT : mask = vol > 0.15*vol.max() (AnalyticalVessel), identique aux étapes
précédentes de la chaîne.

skip_bifurcation=False par défaut : les volumes (64³) restent petits, la
squelettisation/détection de bifurcations est donc peu coûteuse ici à
désactiver via SKIP_BIFURCATION si le coût devient un problème sur un
balayage plus large (plus de méthodes, plus d'échelles).
"""

import time
import warnings
from pathlib import Path
from datetime import datetime
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Chemin ajouté au PYTHONPATH: {project_root}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Maintenant les imports fonctionnent
# Import qualifié ( dans Comparaison_valeurs_propres.py) : la
# segmentation réutilise la vesselness de CVN, qui elle-même réutilise la
# géométrie/les opérateurs de CH. Toute la chaîne est reconfigurable en
# cascade via CS.apply_config() → CVN.apply_config() → CH.apply_config().
from core.experiments.Hessian_evaluation import Comparaison_hessian as CH
from core.experiments.Hessian_evaluation import Comparaison_vesselness as CVN
from core.experiments.Hessian_evaluation.Comparaison_hessian import (
    AnalyticalVessel,
    add_nyquist_noise,
)
from core.experiments.Hessian_evaluation.Comparaison_vesselness import compute_vesselness
from core.processing.derivator import Derivator
from core.processing.enhancer import Enhancer
from core.processing.segmenter import Segmenter
from core.utils.helpers import normalize_data
from core.experiments.metrics import detailed_metrics


SEED = 42
np.random.seed(SEED)
warnings.filterwarnings('ignore')

SKIP_BIFURCATION = False


# 0bis. CONFIGURATION DEPUIS YAML

def apply_config(*, seed=None, shape=None, sigma_vessel=None, vessel_radius=None,
                  methods_custom=None, methods_skimg=None, cases=None,
                  scales=None, filters_to_test=None, black_ridges=None,
                  skip_bifurcation=None):
    """Répercute les paramètres YAML sur ce module ET, en cascade, sur
    Comparaison_vesselness puis Comparaison_hessian via CVN.apply_config()."""
    global SEED, SKIP_BIFURCATION

    CVN.apply_config(
        seed=seed, shape=shape, sigma_vessel=sigma_vessel,
        vessel_radius=vessel_radius,
        methods_custom=methods_custom, methods_skimg=methods_skimg,
        cases=cases, scales=scales, filters_to_test=filters_to_test,
        black_ridges=black_ridges,
    )
    if seed is not None:
        SEED = seed
        np.random.seed(SEED)
    if skip_bifurcation is not None:
        SKIP_BIFURCATION = skip_bifurcation


def run_study(config):
    """Point d'entrée piloté par YAML (config: SegmentationStudyConfig)."""
    apply_config(
        seed=config.seed,
        shape=config.geometry.shape,
        sigma_vessel=config.geometry.sigma_vessel,
        vessel_radius=config.geometry.vessel_radius,
        methods_custom=config.methods.custom,
        methods_skimg=config.methods.skimage,
        cases=config.cases,
        scales=config.scales,
        filters_to_test=config.filters_to_test,
        black_ridges=config.black_ridges,
        skip_bifurcation=config.skip_bifurcation,
    )
    df, out_dir = run_benchmark(output_dir=config.results_dir, verbose=config.verbose)
    print_report(df)
    if config.plot:
        plot_results(df, out_dir, save=True)
    return df, out_dir

# Colonnes du dict detailed_metrics() reportées dans le classement principal.
RANKING_METRICS = {
    'dice': True,
    'mcc': True,
    'cldice': True,
    'bifurcation_f1': True,
    'fragmentation_ratio': False, # proche de 1 = idéal, mais on pénalise l'écart ailleurs
    'largest_gt_recall': True,
}



# 1. SEGMENTATION + MÉTRIQUES POUR UN (méthode, filtre) SUR UN CAS

def segment_and_evaluate(volume, gt_binary, method, filter_name, derivator, enhancer, segmenter):
    vesselness, hessian_time, wall_time = compute_vesselness(
        volume, method, filter_name, derivator, enhancer
    )

    # Même chemin que Processor.process_data() → segmentation_function(...)
    _data_segmented, threshold = segmenter.thresholding(
        data=vesselness,
        ground_truth=gt_binary,
        mask=None,
    )

    data_normalized = normalize_data(vesselness)

    metrics = detailed_metrics(
        data_normalized, gt_binary, mask=None,
        threshold=threshold, skip_bifurcation=SKIP_BIFURCATION,
    )

    metrics['threshold'] = threshold
    metrics['hessian_time_s'] = hessian_time
    metrics['wall_time_s'] = wall_time
    return metrics


# 2. BENCHMARK PRINCIPAL

def run_benchmark(output_dir='segmentation_benchmark', verbose=True):
    gen = AnalyticalVessel(shape=CH.SHAPE)
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    derivator = Derivator(use_gpu=False)
    enhancer = Enhancer(use_gpu=False)
    segmenter = Segmenter()

    all_rows = []

    for case_name, method_name in CVN.CASES.items():
        builder = getattr(gen, method_name)
        vol, _H_gt, mask, _transverse = builder()
        gt_binary = mask.astype(np.uint8)

        if verbose:
            print(f"\n[{case_name}] voxels_vessel={mask.sum()} volume_shape={vol.shape}")

        variants = {'clean': vol, 'nyquist': add_nyquist_noise(vol)}

        for vname, volume in variants.items():
            for filter_name in CVN.FILTERS_TO_TEST:
                for method in tqdm(
                    CH.METHODS, desc=f"{case_name}/{vname}/{filter_name}",
                    disable=not verbose
                ):
                    try:
                        row = segment_and_evaluate(
                            volume, gt_binary, method, filter_name,
                            derivator, enhancer, segmenter,
                        )
                        row['method'] = method
                        row['filter'] = filter_name
                        row['has_own_smooth'] = method in CH.METHODS_SKIMG
                        row['case'] = case_name
                        row['variant'] = vname
                        row['noisy'] = (vname == 'nyquist')
                        row['test_case'] = f"{case_name}_{vname}"
                        all_rows.append(row)
                    except Exception as exc:
                        if verbose:
                            print(f"{filter_name}/{method}: {exc}")
                        import traceback
                        traceback.print_exc()

    df = pd.DataFrame(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"segmentation_benchmark_{ts}.csv"
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\nCSV → {csv_path}")
    return df, out_dir


# 3. RAPPORT TEXTE

def print_report(df: pd.DataFrame) -> None:
    sep = "="* 100
    clean = df[df['variant'] == 'clean']

    print(f"\n{sep}")
    print("BENCHMARK SEGMENTATION - Segmenter + detailed_metrics par opérateur")
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    print(sep)

    for filter_name in CVN.FILTERS_TO_TEST:
        fclean = clean[(clean['filter'] == filter_name) & (~clean['has_own_smooth'])]
        if fclean.empty:
            continue

        print(f"\n{sep}")
        print(f"FILTRE : {filter_name.upper()} - dérivation pure")
        print("(farid, cubic, trigonometric, catmull, bspline, bezier)")
        print(sep)

        agg = fclean.groupby('method')[list(RANKING_METRICS.keys())].mean().round(5)

        ranks = pd.DataFrame(index=agg.index)
        for metric, higher_is_better in RANKING_METRICS.items():
            ranks[f'rk_{metric}'] = agg[metric].rank(ascending=not higher_is_better)
        agg['score'] = ranks.mean(axis=1)
        agg = agg.sort_values('score')

        hdr = (f"{'Rg':<3} {'Methode':<16} {'Dice':>7} {'MCC':>7} {'clDice':>7} "
               f"{'BifF1':>7} {'FragR':>7} {'GTrec':>7} {'Score':>7}")
        print(hdr)
        print("-"* len(hdr))
        for rank, (m, r) in enumerate(agg.iterrows(), 1):
            print(f"{rank:<3} {m:<16} {r['dice']:>7.4f} {r['mcc']:>7.4f} "
                  f"{r['cldice']:>7.4f} {r['bifurcation_f1']:>7.4f} "
                  f"{r['fragmentation_ratio']:>7.2f} {r['largest_gt_recall']:>7.4f} "
                  f"{r['score']:>7.2f}")

        fskim = clean[(clean['filter'] == filter_name) & (clean['has_own_smooth'])]
        if not fskim.empty:
            print(f"\n Informatif (lissage interne, non classé) :")
            for m, g in fskim.groupby('method'):
                print(f"{m:<10} dice={g['dice'].mean():.4f} mcc={g['mcc'].mean():.4f} "
                      f"cldice={g['cldice'].mean():.4f} bifF1={g['bifurcation_f1'].mean():.4f}")

        print(f"\n ROBUSTESSE AU BRUIT (Delta Dice : nyquist - clean)")
        sub = df[(df['filter'] == filter_name) & (df['method'].isin(CH.METHODS_CUSTOM))]
        piv = sub.groupby(['method', 'variant'])['dice'].mean().unstack()
        if 'nyquist'in piv and 'clean'in piv:
            delta = (piv['nyquist'] - piv['clean']).sort_values()
            for m, d in delta.items():
                print(f"{m:<16} Delta Dice = {d:+.4f}")

        print(f"\n FRAGMENTATION (fragmentation_ratio moyen, 1.0 = idéal)")
        for m, v in fclean.groupby('method')['fragmentation_ratio'].mean().sort_values().items():
            print(f"{m:<16} {v:.2f}")


# 4. VISUALISATION

def plot_results(df: pd.DataFrame, out_dir: Path, save: bool = True):
    sns.set_style('whitegrid')
    clean = df[df['variant'] == 'clean']

    fig, axes = plt.subplots(2, len(CVN.FILTERS_TO_TEST), figsize=(9 * len(CVN.FILTERS_TO_TEST), 11))
    if len(CVN.FILTERS_TO_TEST) == 1:
        axes = axes.reshape(2, 1)

    for col, filter_name in enumerate(CVN.FILTERS_TO_TEST):
        fair = clean[(clean['filter'] == filter_name) & (~clean['has_own_smooth'])]

        ax = axes[0, col]
        fair.pivot_table('dice', 'method', 'case').plot(
            kind='bar', ax=ax, colormap='viridis', width=0.7)
        ax.set_title(f'{filter_name.upper()} - Dice par cas', fontweight='bold')
        ax.set_ylabel('Dice')
        ax.tick_params(axis='x', rotation=35)
        ax.legend(title='Cas', fontsize=8)

        ax = axes[1, col]
        agg = fair.groupby('method')[list(RANKING_METRICS.keys())].mean()
        rk = pd.DataFrame(index=agg.index)
        for metric, higher_is_better in RANKING_METRICS.items():
            rk[metric] = agg[metric].rank(ascending=not higher_is_better)
        rk['score'] = rk.mean(axis=1)
        rk = rk.sort_values('score')
        sns.heatmap(rk, ax=ax, annot=True, fmt='.1f', cmap='RdYlGn_r',
                    linewidths=0.5, cbar_kws={'label': 'Rang (1=meilleur)'})
        ax.set_title(f'{filter_name.upper()} - classement global (clean)', fontweight='bold')

    plt.suptitle('Benchmark Segmentation - Segmenter + detailed_metrics par opérateur',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"segmentation_plot_{ts}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Figure → {out}")
    return fig


# 5. POINT D'ENTRÉE

if __name__ == '__main__':
    df, out_dir = run_benchmark(verbose=True)
    print_report(df)
    plot_results(df, out_dir, save=True)
    plt.show()