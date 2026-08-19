"""
Benchmark des opérateurs Hessiens métriques sur valeurs propres uniquement.
 : GT analytique exacte + normalisation par gain directionnel sur bande.

Infrastructure identique à benchmark_hessian_v2.py :
  - GT : hessienne analytique forme fermée (cylindre gaussien 3D, zéro FD)
  - Opérateurs : apply_custom / apply_skimage, aucun lissage résiduel
  - Normalisation : α directionnel sur bande [0.5·ω_v, 1.5·ω_v], par cas

Métriques (toutes sur valeurs propres après normalisation α) :
  RMSE λ₂ : valeur propre transverse mineure vs GT
  Noise amplification : variance de λ₂ sur N réalisations de bruit Nyquist
  Instability rate : % voxels où λ₁≥0 ou λ₂≥0 après normalisation
  Rotation angle : erreur orientation vecteur axial v₃ (λ₃≈0)

Convention pour un vaisseau tubulaire :
  λ₁ ≤ λ₂ < 0 (courbures transverses, négatives)
  λ₃ ≈ 0 (direction axiale)
  v₃ = vecteur propre de λ₃ = axe du vaisseau
"""
# je voudrais

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
import warnings
import sys
import os

# Ajoute le chemin racine du projet au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.experiments.Hessian_evaluation import Comparaison_hessian as CH
from core.experiments.Hessian_evaluation.Comparaison_hessian import (
    AnalyticalVessel,
    apply_operator,
    compute_gain_on_band,
    add_nyquist_noise,
)

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings('ignore')

N_NOISE_REALIZATIONS = 20
SNR_DB = 20.0

CASES = {
    'simple': 'simple_vessel',
    'curved': 'curved_vessel',
    'bifurcation': 'bifurcation',
}


# 0bis. CONFIGURATION DEPUIS YAML 

def apply_config(*, seed=None, shape=None, sigma_vessel=None, vessel_radius=None,
                  methods_custom=None, methods_skimg=None, cases=None,
                  snr_db=None, n_noise_realizations=None):
    """Répercute les paramètres YAML sur ce module ET, en cascade, sur
    Comparaison_hessian (géométrie, opérateurs) via CH.apply_config()."""
    global SEED, SNR_DB, N_NOISE_REALIZATIONS, CASES

    CH.apply_config(
        seed=seed, shape=shape, sigma_vessel=sigma_vessel,
        vessel_radius=vessel_radius,
        methods_custom=methods_custom, methods_skimg=methods_skimg,
    )
    if seed is not None:
        SEED = seed
        np.random.seed(SEED)
    if snr_db is not None:
        SNR_DB = snr_db
    if n_noise_realizations is not None:
        N_NOISE_REALIZATIONS = n_noise_realizations
    if cases is not None:
        CASES = dict(cases)


def run_study(config):
    """Point d'entrée piloté par YAML (config: EigenvalueStudyConfig)."""
    apply_config(
        seed=config.seed,
        shape=config.geometry.shape,
        sigma_vessel=config.geometry.sigma_vessel,
        vessel_radius=config.geometry.vessel_radius,
        methods_custom=config.methods.custom,
        methods_skimg=config.methods.skimage,
        cases=config.cases,
        snr_db=config.noise.snr_db,
        n_noise_realizations=config.n_noise_realizations,
    )
    df, out_dir = run_benchmark(output_dir=config.results_dir, verbose=config.verbose)
    print_report(df)
    if config.plot:
        plot_results(df, out_dir, save=True)
    return df, out_dir


# 1. VALEURS ET VECTEURS PROPRES

def eigvals_sorted(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    λ₁ ≤ λ₂ ≤ λ₃ (eigvalsh garantit l'ordre croissant).
    Entrée H : (*shape, 3, 3).
    """
    sh = H.shape[:-2]
    vals = np.linalg.eigvalsh(H.reshape(-1, 3, 3)) # (n, 3)
    return (vals[:, 0].reshape(sh),
            vals[:, 1].reshape(sh),
            vals[:, 2].reshape(sh))


def eigvecs_sorted(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vecteurs propres associés à λ₁ ≤ λ₂ ≤ λ₃.
    eigh trie les colonnes par valeur propre croissante :
      col 0 → v₁ (λ₁), col 1 → v₂ (λ₂), col 2 → v₃ (λ₃≈0, axial).
    """
    sh = H.shape[:-2]
    _, vecs = np.linalg.eigh(H.reshape(-1, 3, 3)) # (n, 3, 3)

    v1 = vecs[:, :, 0]
    v2 = vecs[:, :, 1]
    v3 = vecs[:, :, 2]

    # Normalisation de sécurité (eigh produit déjà des vecteurs unitaires)
    for v in (v1, v2, v3):
        nrm = np.linalg.norm(v, axis=1, keepdims=True).clip(1e-8)
        v /= nrm

    shape3 = (*sh, 3)
    return v1.reshape(shape3), v2.reshape(shape3), v3.reshape(shape3)


# 2. MÉTRIQUES

def metric_rmse_l2(H_op: np.ndarray, H_gt: np.ndarray,
                   mask: np.ndarray) -> float:
    """RMSE de λ₂ après normalisation α (déjà appliquée dans H_op)."""
    _, l2_op, _ = eigvals_sorted(H_op)
    _, l2_gt, _ = eigvals_sorted(H_gt)
    err = l2_op[mask] - l2_gt[mask]
    return float(np.sqrt(np.mean(err ** 2)))


def metric_noise_amplification(
    volume_clean: np.ndarray,
    method: str,
    alpha: float,
    mask: np.ndarray,
    n_real: int = N_NOISE_REALIZATIONS,
    snr_db: float = SNR_DB,
) -> Tuple[float, float]:
    """
    Variance et std de λ₂ sur n_real réalisations de bruit Nyquist.
    Chaque réalisation est normalisée par le même α.
    Retourne (var_l2, std_l2).
    """
    rng = np.random.default_rng(SEED)
    sp = float(np.var(volume_clean[volume_clean > 0.15 * volume_clean.max()]))
    sn = np.sqrt(sp / 10 ** (snr_db / 10.0))

    # Bruit HF identique à add_nyquist_noise mais avec graine contrôlée
    from scipy.ndimage import gaussian_filter as _gf
    all_l2 = []
    for _ in range(n_real):
        noise_raw = rng.standard_normal(volume_clean.shape)
        noise_hf = noise_raw - _gf(noise_raw, sigma=1.5)
        scale = sn / (float(np.std(noise_hf)) + 1e-12)
        vol_noisy = np.clip(volume_clean + noise_hf * scale, 0.0, None)

        H_noisy = apply_operator(vol_noisy, method) * alpha
        _, l2_n, _ = eigvals_sorted(H_noisy)
        all_l2.append(l2_n[mask])

    stack = np.stack(all_l2, axis=0) # (n_real, n_voxels)
    var = float(np.mean(np.var(stack, axis=0, ddof=1)))
    std = float(np.mean(np.std(stack, axis=0, ddof=1)))
    return var, std


def metric_instability_rate(H_op: np.ndarray, mask: np.ndarray) -> float:
    """
    % voxels (dans le masque) où λ₁ ≥ 0 ou λ₂ ≥ 0 après normalisation α.
    Incompatible avec une structure tubulaire perturberait Frangi directement.
    Intrinsèque : ne dépend pas de la GT.
    """
    vals = np.linalg.eigvalsh(H_op.reshape(-1, 3, 3)[mask.ravel()])
    unstable = (vals[:, 0] >= 0) | (vals[:, 1] >= 0)
    return float(100.0 * np.mean(unstable))


def metric_rotation_angle(H_op: np.ndarray, H_gt: np.ndarray,
                           mask: np.ndarray) -> float:
    """
    Erreur d'orientation du vecteur axial v₃ (associé à λ₃ ≈ 0).
    Ambiguïté de signe gérée via |dot product|.
    """
    _, _, v3_gt = eigvecs_sorted(H_gt)
    _, _, v3_op = eigvecs_sorted(H_op)

    dot = np.sum(v3_op[mask] * v3_gt[mask], axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    return float(np.mean(np.nan_to_num(angle, nan=0.0)))


def refine_mask(mask: np.ndarray, l2_gt: np.ndarray,
                pct: int = 10) -> np.ndarray:
    """
    Élimine les voxels où |λ₂_gt| < percentile `pct` (bords à faible signal).
    pct=10 préserve 90 % du masque.
    """
    l2_m = l2_gt[mask]
    threshold = np.percentile(np.abs(l2_m), pct)
    refined = mask & (np.abs(l2_gt) > threshold)
    print(f"Masque : {mask.sum()} → {refined.sum()} voxels "
          f"({100 * refined.sum() / mask.sum():.1f} %)")
    return refined if refined.sum() >= 100 else mask


# 3. ÉVALUATION D'UN OPÉRATEUR SUR UN CAS

def evaluate_method(
    volume_clean: np.ndarray,
    H_gt: np.ndarray,
    mask: np.ndarray,
    method: str,
    alpha: float,
    verbose: bool = False,
) -> Dict:
    """
    Calcule les 4 métriques pour `method` sur un cas.
    H_op = apply_operator(volume_clean, method) * alpha.
    """
    H_op = apply_operator(volume_clean, method) * alpha

    rmse = metric_rmse_l2(H_op, H_gt, mask)
    var_n, std_n = metric_noise_amplification(
        volume_clean, method, alpha, mask)
    instab = metric_instability_rate(H_op, mask)
    rot = metric_rotation_angle(H_op, H_gt, mask)

    if verbose:
        print(f"RMSE λ₂ : {rmse:.5f}")
        print(f"Noise var : {var_n:.3e}")
        print(f"Instability : {instab:.1f} %")
        print(f"Rotation v₃ : {rot:.2f} °")

    return {
        'rmse_l2': rmse,
        'noise_var': var_n,
        'noise_std': std_n,
        'instability_rate': instab,
        'rotation_angle': rot,
        'alpha': alpha,
        'has_own_smooth': method in CH.METHODS_SKIMG,
    }


# 4. BENCHMARK PRINCIPAL

def run_benchmark(output_dir: str = 'eigenvalue_benchmark_results',
                  verbose: bool = True) -> Tuple[pd.DataFrame, Path]:
    gen = AnalyticalVessel(shape=CH.SHAPE)
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    all_rows = []

    for case_name, method_name in CASES.items():
        builder = getattr(gen, method_name)
        volume, H_gt, mask, transverse = builder()

        _, l2_gt, _ = eigvals_sorted(H_gt)
        mask = refine_mask(mask, l2_gt, pct=10)

        if verbose:
            print(f"\n[{case_name}] voxels={mask.sum()}"
                  f"λ₂_gt ∈ [{l2_gt[mask].min():.4f}, {l2_gt[mask].max():.4f}]"
                  f"transverse={np.round(transverse, 3)}")

        # Calibration α une valeur par opérateur par cas
        alphas = {}
        for m in CH.METHODS:
            alphas[m] = compute_gain_on_band(m, transverse)
            if verbose:
                print(f"α[{m:>14s}] = {alphas[m]:.4f}")

        for method in tqdm(CH.METHODS, desc=f"{case_name}", disable=not verbose):
            try:
                row = evaluate_method(
                    volume, H_gt, mask, method, alphas[method], verbose=verbose)
                row['method'] = method
                row['case'] = case_name
                all_rows.append(row)
            except Exception as exc:
                if verbose:
                    print(f"{method}: {exc}")
                import traceback; traceback.print_exc()

    df = pd.DataFrame(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(out_dir / f"eigenvalue_results_{ts}.csv", index=False)
    return df, out_dir


# 5. RAPPORT TEXTE

def print_report(df: pd.DataFrame) -> None:
    sep = "="* 100
    fair = df[~df['has_own_smooth']]

    print(f"\n{sep}")
    print("BENCHMARK HESSIENNE - MÉTRIQUES VALEURS PROPRES")
    print(f"GT analytique exacte + normalisation α directionnel sur bande")
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    print(sep)

    # ── Par cas ──────────────────────────────────────────────────────────────
    for case in sorted(fair['case'].unique()):
        sub = fair[fair['case'] == case]
        print(f"\n CAS : {case.upper()}")
        print("-"* 60)
        for metric, label in [
            ('rmse_l2', 'Meilleur RMSE λ₂ '),
            ('noise_var', 'Bruit minimal '),
            ('instability_rate', 'Instabilité min '),
            ('rotation_angle', 'Rotation v₃ min '),
        ]:
            best = sub.loc[sub[metric].idxmin()]
            print(f"{label}: {best['method']:<16} {best[metric]:.5f}")

    # ── Classement global méthodes custom ──────────────────────────────────
    print(f"\n{sep}")
    print("CLASSEMENT GLOBAL (custom uniquement)")
    print(sep)

    agg = (fair.groupby('method')
           .agg(rmse =('rmse_l2', 'mean'),
                noise =('noise_var', 'mean'),
                instability=('instability_rate', 'mean'),
                rotation =('rotation_angle', 'mean'))
           .round(6))

    agg['rk_r'] = agg['rmse'].rank()
    agg['rk_n'] = agg['noise'].rank()
    agg['rk_i'] = agg['instability'].rank()
    agg['rk_a'] = agg['rotation'].rank()
    agg['score'] = agg[['rk_r', 'rk_n', 'rk_i', 'rk_a']].mean(axis=1)
    agg = agg.sort_values('score')

    hdr = (f"{'Rg':<3} {'Méthode':<16} {'RMSE_λ₂':>10} "
           f"{'NoiseVar':>12} {'Instab%':>9} {'RotV₃°':>9} {'Score':>7}")
    print(hdr)
    print("-"* 65)
    for rank, (m, r) in enumerate(agg.iterrows(), 1):
        print(f"{rank:<3} {m:<16} {r['rmse']:>10.5f} "
              f"{r['noise']:>12.3e} {r['instability']:>9.2f} "
              f"{r['rotation']:>9.2f} {r['score']:>7.2f}")

    # ── Skimage informatif ────────────────────────────────────────────────────
    own = df[df['has_own_smooth']]
    if not own.empty:
        print(f"\n{sep}")
        print("SKIMAGE (lissage propre σ=0.5, informatif)")
        print(sep)
        for m, g in own.groupby('method'):
            print(f"{m:<14} RMSE={g['rmse_l2'].mean():.5f} "
                  f"Noise={g['noise_var'].mean():.3e} "
                  f"Instab={g['instability_rate'].mean():.1f}% "
                  f"Rot={g['rotation_angle'].mean():.2f}° "
                  f"α={g['alpha'].mean():.4f}")


# 6. VISUALISATION

def plot_results(df: pd.DataFrame, out_dir: Path, save: bool = True) -> plt.Figure:
    fair = df[~df['has_own_smooth']]
    pal = sns.color_palette('tab10', n_colors=len(CH.METHODS_CUSTOM))
    mcol = dict(zip(CH.METHODS_CUSTOM, pal))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        'Benchmark Hessienne - Métriques valeurs propres\n'
        '(GT analytique exacte + normalisation α directionnel)',
        fontsize=13, fontweight='bold'
    )

    def barh(ax, metric, title, xlabel, log=False):
        avg = fair.groupby('method')[metric].mean().sort_values()
        colors = [mcol.get(m, 'gray') for m in avg.index]
        ax.barh(avg.index, avg.values, color=colors)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        if log:
            ax.set_xscale('log')

    barh(axes[0, 0], 'rmse_l2',
         'RMSE λ₂\n(erreur valeur propre transverse)', 'RMSE')

    barh(axes[0, 1], 'noise_var',
         'Noise amplification\n(var λ₂ sous bruit Nyquist)', 'Variance', log=True)

    barh(axes[0, 2], 'instability_rate',
         'Instability rate\n(% λ₁≥0 ou λ₂≥0)', 'Rate (%)')

    barh(axes[1, 0], 'rotation_angle',
         'Rotation angle v₃\n(erreur orientation axiale)', 'Degrés')

    # Radar / score composite
    ax = axes[1, 1]
    agg = fair.groupby('method').agg(
        rmse =('rmse_l2', 'mean'),
        noise =('noise_var', 'mean'),
        instability=('instability_rate','mean'),
        rotation =('rotation_angle', 'mean'),
    )
    for col in agg.columns:
        agg[f'rk_{col}'] = agg[col].rank()
    agg['score'] = agg[[c for c in agg.columns if c.startswith('rk_')]].mean(axis=1)
    agg_sorted = agg.sort_values('score')

    colors_sorted = [mcol.get(m, 'gray') for m in agg_sorted.index]
    ax.barh(agg_sorted.index, agg_sorted['score'], color=colors_sorted)
    ax.set_title('Score composite (rang moyen 4 métriques)\n1=meilleur',
                 fontweight='bold')
    ax.set_xlabel('Score (bas = meilleur)')
    ax.axvline(agg_sorted['score'].min(), color='k', lw=0.8, ls='--')

    # RMSE par cas
    ax = axes[1, 2]
    fair.pivot_table('rmse_l2', 'method', 'case').plot(
        kind='bar', ax=ax, colormap='viridis', width=0.7)
    ax.set_title('RMSE λ₂ par cas\n(clean, après α)', fontweight='bold')
    ax.set_ylabel('RMSE')
    ax.tick_params(axis='x', rotation=35)
    ax.legend(title='Cas', fontsize=8)

    plt.tight_layout()

    if save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"eigenvalue_benchmark_{ts}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Figure → {out}")

    return fig


# 7. POINT D'ENTRÉE

if __name__ == '__main__':
    df, out_dir = run_benchmark(verbose=True)
    print_report(df)
    plot_results(df, out_dir, save=True)
    plt.show()