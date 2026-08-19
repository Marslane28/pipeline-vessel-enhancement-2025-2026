import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

# Configuration du chemin - À METTRE EN PREMIER
import sys
import os

# Ajoute le chemin racine du projet au PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Chemin ajouté au PYTHONPATH: {project_root}")

from core.experiments.Hessian_evaluation import Comparaison_hessian as CH
from core.experiments.Hessian_evaluation.Comparaison_hessian import (
    AnalyticalVessel,
    add_nyquist_noise,
)
from core.processing.derivator import Derivator
from core.processing.enhancer import Enhancer

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings('ignore')


SCALES = [1.0, 1.5, 2.0, 2.5, 3.0]
FILTERS_TO_TEST = ['frangi', 'jerman', 'mfat']
BLACK_RIDGES = False

CASES = {
    'simple': 'simple_vessel',
    'curved': 'curved_vessel',
    'bifurcation': 'bifurcation',
}


# 0bis. CONFIGURATION DEPUIS YAML (core.config.operator_study.VesselnessStudyConfig)

def apply_config(*, seed=None, shape=None, sigma_vessel=None, vessel_radius=None,
                  methods_custom=None, methods_skimg=None, cases=None,
                  scales=None, filters_to_test=None, black_ridges=None):
    """Répercute les paramètres YAML sur ce module ET, en cascade, sur
    Comparaison_hessian (géométrie, opérateurs) via CH.apply_config()."""
    global SEED, SCALES, FILTERS_TO_TEST, BLACK_RIDGES, CASES

    CH.apply_config(
        seed=seed, shape=shape, sigma_vessel=sigma_vessel,
        vessel_radius=vessel_radius,
        methods_custom=methods_custom, methods_skimg=methods_skimg,
    )
    if seed is not None:
        SEED = seed
        np.random.seed(SEED)
    if scales is not None:
        SCALES = list(scales)
    if filters_to_test is not None:
        FILTERS_TO_TEST = list(filters_to_test)
    if black_ridges is not None:
        BLACK_RIDGES = black_ridges
    if cases is not None:
        CASES = dict(cases)


def run_study(config):
    """Point d'entrée piloté par YAML (config: VesselnessStudyConfig)."""
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
    )
    df, out_dir = run_benchmark(output_dir=config.results_dir, verbose=config.verbose)
    print_report(df)
    if config.plot:
        plot_results(df, out_dir, save=True)
    return df, out_dir


# 1. ÉVALUATION D'UN (méthode, filtre) SUR UN VOLUME

def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def best_f1_threshold(vesselness: np.ndarray, gt_binary: np.ndarray) -> float:
    """F1 max sur la vesselness seuillée, balayage via precision_recall_curve."""
    v = vesselness.ravel()
    g = gt_binary.ravel().astype(np.uint8)
    if len(np.unique(g)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(g, v)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    return float(f1.max()) if len(f1) > 0 else 0.0


def compute_vesselness(volume, method, filter_name, derivator, enhancer):
    """Appelle Enhancer.frangi/jerman/mfat avec le Hessien piloté par `method`."""
    hessian_function = derivator.select_hessian_function(method)
    enhancement_function = enhancer.select_enhancement_function(filter_name)

    t0 = time.perf_counter()
    result = enhancement_function(
        volume,
        hessian_function=hessian_function,
        scales=SCALES,
        black_ridges=BLACK_RIDGES,
        mask=None,
    )
    wall_time = time.perf_counter() - t0

    if isinstance(result, tuple) and len(result) == 2:
        vesselness, hessian_time = result
    else:
        vesselness, hessian_time = result, None

    return np.asarray(vesselness, dtype=np.float64), hessian_time, wall_time


def evaluate(volume, gt_continuous, gt_binary, method, filter_name, derivator, enhancer):
    vesselness, hessian_time, wall_time = compute_vesselness(
        volume, method, filter_name, derivator, enhancer
    )

    v_flat = vesselness.ravel()
    g_flat = gt_continuous.ravel()

    if np.std(v_flat) < 1e-12 or np.std(g_flat) < 1e-12:
        corr = 0.0
    else:
        corr, _ = pearsonr(v_flat, g_flat)
        corr = float(corr)

    v_norm = _minmax(vesselness)
    rmse = float(np.sqrt(np.mean((v_norm - gt_continuous) ** 2)))

    f1 = best_f1_threshold(vesselness, gt_binary)

    return {
        'method': method,
        'filter': filter_name,
        'has_own_smooth': method in CH.METHODS_SKIMG,
        'vesselness_corr': corr,
        'vesselness_rmse': rmse,
        'f1_best': f1,
        'v_max': float(vesselness.max()),
        'v_mean': float(vesselness.mean()),
        'hessian_time_s': hessian_time,
        'wall_time_s': wall_time,
    }


# 2. BENCHMARK PRINCIPAL

def run_benchmark(output_dir='vesselness_benchmark', verbose=True):
    gen = AnalyticalVessel(shape=CH.SHAPE)
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    derivator = Derivator(use_gpu=False)
    enhancer = Enhancer(use_gpu=False)

    all_rows = []

    for case_name, method_name in CASES.items():
        builder = getattr(gen, method_name)
        vol, _H_gt, mask, _transverse = builder()

        gt_continuous = vol / vol.max() if vol.max() > 0 else vol
        gt_binary = mask.astype(np.uint8)

        if verbose:
            print(f"\n[{case_name}] voxels_vessel={mask.sum()} "
                  f"volume_shape={vol.shape}")

        variants = {'clean': vol, 'nyquist': add_nyquist_noise(vol)}

        for vname, volume in variants.items():
            for filter_name in FILTERS_TO_TEST:
                for method in tqdm(
                    CH.METHODS, desc=f"{case_name}/{vname}/{filter_name}",
                    disable=not verbose
                ):
                    try:
                        row = evaluate(
                            volume, gt_continuous, gt_binary,
                            method, filter_name, derivator, enhancer,
                        )
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
    csv_path = out_dir / f"vesselness_benchmark_{ts}.csv"
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\nCSV → {csv_path}")
    return df, out_dir


# 3. RAPPORT TEXTE

def print_report(df: pd.DataFrame) -> None:
    sep = "="* 92
    clean = df[df['variant'] == 'clean']

    print(f"\n{sep}")
    print("BENCHMARK VESSELNESS - Enhancer (Frangi / Jerman) par opérateur")
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    print(sep)

    for filter_name in FILTERS_TO_TEST:
        fclean = clean[(clean['filter'] == filter_name) & (~clean['has_own_smooth'])]
        if fclean.empty:
            continue

        print(f"\n{sep}")
        print(f"FILTRE : {filter_name.upper()} - dérivation pure")
        print("(farid, cubic, trigonometric, catmull, bspline, bezier)")
        print(sep)

        agg = (fclean.groupby('method')
               .agg(corr=('vesselness_corr', 'mean'),
                    rmse=('vesselness_rmse', 'mean'),
                    f1=('f1_best', 'mean'),
                    temps=('wall_time_s', 'mean'))
               .round(5))

        agg['rk_c'] = agg['corr'].rank(ascending=False)
        agg['rk_r'] = agg['rmse'].rank()
        agg['rk_f'] = agg['f1'].rank(ascending=False)
        agg['score'] = agg[['rk_c', 'rk_r', 'rk_f']].mean(axis=1)
        agg = agg.sort_values('score')

        hdr = f"{'Rg':<3} {'Methode':<16} {'Corr':>8} {'RMSE':>8} {'F1best':>8} {'t(s)':>8} {'Score':>7}"
        print(hdr)
        print("-"* len(hdr))
        for rank, (m, r) in enumerate(agg.iterrows(), 1):
            print(f"{rank:<3} {m:<16} {r['corr']:>8.4f} {r['rmse']:>8.4f} "
                  f"{r['f1']:>8.4f} {r['temps']:>8.4f} {r['score']:>7.2f}")

        fskim = clean[(clean['filter'] == filter_name) & (clean['has_own_smooth'])]
        if not fskim.empty:
            print(f"\n Informatif (lissage interne, non classé) :")
            for m, g in fskim.groupby('method'):
                print(f"{m:<10} corr={g['vesselness_corr'].mean():.4f} "
                      f"rmse={g['vesselness_rmse'].mean():.4f} "
                      f"f1={g['f1_best'].mean():.4f}")

        print(f"\n ROBUSTESSE AU BRUIT (Delta F1 : nyquist - clean)")
        pure = [m for m in CH.METHODS_CUSTOM]
        sub = df[(df['filter'] == filter_name) & (df['method'].isin(pure))]
        piv = sub.groupby(['method', 'variant'])['f1_best'].mean().unstack()
        if 'nyquist'in piv and 'clean'in piv:
            delta = (piv['nyquist'] - piv['clean']).sort_values()
            for m, d in delta.items():
                print(f"{m:<16} Delta F1 = {d:+.4f}")


# 4. VISUALISATION

def plot_results(df: pd.DataFrame, out_dir: Path, save: bool = True):
    sns.set_style('whitegrid')
    clean = df[df['variant'] == 'clean']

    fig, axes = plt.subplots(2, len(FILTERS_TO_TEST), figsize=(9 * len(FILTERS_TO_TEST), 11))
    if len(FILTERS_TO_TEST) == 1:
        axes = axes.reshape(2, 1)

    for col, filter_name in enumerate(FILTERS_TO_TEST):
        fair = clean[(clean['filter'] == filter_name) & (~clean['has_own_smooth'])]

        ax = axes[0, col]
        fair.pivot_table('f1_best', 'method', 'case').plot(
            kind='bar', ax=ax, colormap='viridis', width=0.7)
        ax.set_title(f'{filter_name.upper()} - F1 (seuil optimal) par cas', fontweight='bold')
        ax.set_ylabel('F1 best')
        ax.tick_params(axis='x', rotation=35)
        ax.legend(title='Cas', fontsize=8)

        ax = axes[1, col]
        agg = fair.groupby('method').agg(
            corr=('vesselness_corr', 'mean'),
            rmse=('vesselness_rmse', 'mean'),
            f1=('f1_best', 'mean'),
        )
        rk = agg.rank(ascending=False)
        rk['rmse'] = agg['rmse'].rank(ascending=True)
        rk['score'] = rk.mean(axis=1)
        rk = rk.sort_values('score')
        sns.heatmap(rk, ax=ax, annot=True, fmt='.1f', cmap='RdYlGn_r',
                    linewidths=0.5, cbar_kws={'label': 'Rang (1=meilleur)'})
        ax.set_title(f'{filter_name.upper()} - classement global (clean)', fontweight='bold')

    plt.suptitle('Benchmark Vesselness - Enhancer (Frangi/Jerman) par opérateur',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"vesselness_plot_{ts}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Figure → {out}")
    return fig


# 5. POINT D'ENTRÉE

if __name__ == '__main__':
    df, out_dir = run_benchmark(verbose=True)
    print_report(df)
    plot_results(df, out_dir, save=True)
    plt.show()