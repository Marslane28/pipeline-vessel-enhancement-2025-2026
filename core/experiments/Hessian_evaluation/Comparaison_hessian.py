"""
  1. GT analytique analytical closed-form reference. (forme fermée, cylindre gaussien 3D).
     Aucun gaussian_filter, aucune FD numérique dans la GT.
  2. Normalisation par gain directionnel sur bande [0.5·ω_v, 1.5·ω_v]
     dans la direction transverse au vaisseau - recalculée par cas.
     Élimine le biais d'amplitude de bezier/trigonométrique.
  3. Deux variantes : signal pur (clean) + bruit de Nyquist (nyquist, SNR=20dB).
  4. Opérateurs à lissage propre (default/gaussian skimage) inclus
     avec flag has_own_smooth - normalisés de la même façon.
  5. Aucun monkey-patch : Derivator est appelé via une sous-classe
     qui désactive le lissage interne proprement.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import time
from scipy.ndimage import convolve1d
from tqdm import tqdm
import seaborn as sns

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings('ignore')

N_REPEATS = 7
WARMUP = 1

SHAPE = (64, 64, 64)
SIGMA_VESSEL = 2.0 # largeur gaussienne du vaisseau (voxels)
VESSEL_RADIUS = 4.0 # rayon de troncature
SNR_DB = 20.0 # rapport signal/bruit (Nyquist)

METHODS_CUSTOM = ['farid', 'cubic', 'trigonometric', 'catmull', 'bspline', 'bezier', 'scharr']
METHODS_SKIMG = ['default', 'gaussian']
METHODS = METHODS_CUSTOM + METHODS_SKIMG

def apply_config(*, seed=None, shape=None, sigma_vessel=None, vessel_radius=None,
                  snr_db=None, methods_custom=None, methods_skimg=None,
                  cases=None, n_repeats=None, warmup=None):

    global SEED, SHAPE, SIGMA_VESSEL, VESSEL_RADIUS, SNR_DB
    global METHODS_CUSTOM, METHODS_SKIMG, METHODS, CASES, N_REPEATS, WARMUP

    if seed is not None:
        SEED = seed
        np.random.seed(SEED)
    if shape is not None:
        SHAPE = tuple(shape)
    if sigma_vessel is not None:
        SIGMA_VESSEL = sigma_vessel
    if vessel_radius is not None:
        VESSEL_RADIUS = vessel_radius
    if snr_db is not None:
        SNR_DB = snr_db
    if methods_custom is not None:
        METHODS_CUSTOM = list(methods_custom)
    if methods_skimg is not None:
        METHODS_SKIMG = list(methods_skimg)
    if methods_custom is not None or methods_skimg is not None:
        METHODS = METHODS_CUSTOM + METHODS_SKIMG
    if cases is not None:
        CASES = dict(cases)
    if n_repeats is not None:
        N_REPEATS = n_repeats
    if warmup is not None:
        WARMUP = warmup


def run_study(config):
    """Point d'entrée piloté par YAML (config: HessianStudyConfig).
    Appelé depuis main.py via --run_operator_study --study_type hessian."""
    apply_config(
        seed=config.seed,
        shape=config.geometry.shape,
        sigma_vessel=config.geometry.sigma_vessel,
        vessel_radius=config.geometry.vessel_radius,
        snr_db=config.noise.snr_db,
        methods_custom=config.methods.custom,
        methods_skimg=config.methods.skimage,
        cases=config.cases,
        n_repeats=config.timing.n_repeats,
        warmup=config.timing.warmup,
    )
    df, out_dir = run_benchmark(output_dir=config.results_dir, verbose=config.verbose)
    print_report(df)
    if config.plot:
        plot_results(df, out_dir, save=True)
    return df, out_dir

def normalize_kernel_d1(k):
    """Normalise un noyau de dérivée première (test sur rampe)."""
    x = np.arange(len(k)) - len(k)//2
    gain = np.sum(k * x)
    return k / gain if abs(gain) > 1e-12 else k

def normalize_kernel_d2(k):
    """Normalise un noyau de dérivée seconde (test sur parabole)."""
    x = np.arange(len(k)) - len(k)//2
    gain = np.sum(k * (x**2 / 2))
    return k / gain if abs(gain) > 1e-12 else k

# 1. FILTRES - définition centralisée
#On normalise les noyaux de dérivée première et seconde pour que la réponse à une rampe (d1) ou une parabole (d2) soit correcte.

FILTERS = {
    'farid7': {
        'p': np.array([0.0047, 0.0693, 0.2454, 0.3611, 0.2454, 0.0693, 0.0047]),
        'd1': normalize_kernel_d1(np.array([0.0187, 0.1253, 0.1930, 0.0, -0.1930, -0.1253, -0.0187])), # np.array([0.0187, 0.1253, 0.1930, 0.0, -0.1930, -0.1253, -0.0187]),
        'd2': normalize_kernel_d2(np.array([0.0553, 0.1378, -0.0566, -0.2731, -0.0566, 0.1378, 0.0553])),
    },
    'cubic': {
        'p': np.array([0.0039, -0.0703, 0.2461, 0.6406, 0.2461, -0.0703, 0.0039]),
        'd1': normalize_kernel_d1(np.array([-0.0026, 0.0938, -0.6797, 0.0, 0.6797, -0.0938, 0.0026])),
        'd2': normalize_kernel_d2(np.array([-0.0312, 0.3125, 0.0312, -0.6250, 0.0312, 0.3125, -0.0312])),
    },
    'trigonometric': {
        'p': np.array([0.0043, -0.0745, 0.2457, 0.6490, 0.2457, -0.0745, 0.0043]),
        'd1': normalize_kernel_d1(np.array([-0.0061, 0.1966, -1.3282, 0.0, 1.3282, -0.1966, 0.0061])),
        'd2': normalize_kernel_d2(np.array([-0.1272, 1.2203, 0.1272, -2.4405, 0.1272, 1.2203, -0.1272])),
    },
    'catmull': {
        'p': np.array([0.0039, -0.0703, 0.2461, 0.6406, 0.2461, -0.0703, 0.0039]),
        'd1': normalize_kernel_d1(np.array([-0.0078, 0.1562, -0.7891, 0.0, 0.7891, -0.1562, 0.0078])),
        'd2': normalize_kernel_d2(np.array([-0.0312, 0.3125, 0.0312, -0.6250, 0.0312, 0.3125, -0.0312])),
    },
    'bspline': {
        'p': np.array([0.0004, 0.0200, 0.2496, 0.4601, 0.2496, 0.0200, 0.0004]),
        'd1': normalize_kernel_d1(np.array([-0.0026, -0.0729, -0.3464, 0.0, 0.3464, 0.0729, 0.0026])),
        'd2': normalize_kernel_d2(np.array([0.0104, 0.2292, -0.0104, -0.4583, -0.0104, 0.2292, 0.0104])),
    },
    'bezier': {
        'p': np.array([0.0156, 0.0938, 0.2344, 0.3125, 0.2344, 0.0938, 0.0156]),
        'd1': normalize_kernel_d1(np.array([0.0938, 0.3750, 0.4688, 0.0, -0.4688, -0.3750, -0.0938])),
        'd2': normalize_kernel_d2(np.array([0.3750, 0.7500, -0.3750, -1.5000, -0.3750, 0.7500, 0.3750])),
    },
    'scharr': {
        'p': np.array([0.01554, 0.23204, 0.50484, 0.23204, 0.01554]),
        'd1': normalize_kernel_d1(np.array([0.06368, 0.37263, 0.0, -0.37263, -0.06368])),
        'd2': normalize_kernel_d2(np.array([0.20786, 0.16854, -0.75282, 0.16854, 0.20786])),
    },
}

METHOD_TO_FILTER = {
    'farid': 'farid7', 'cubic': 'cubic', 'trigonometric': 'trigonometric',
    'catmull': 'catmull', 'bspline': 'bspline', 'bezier': 'bezier', 'scharr': 'scharr',
}


# 2. APPLICATION DES OPÉRATEURS


def apply_custom(volume: np.ndarray, method: str, mode: str = 'reflect') -> tuple:
    """
    Applique un opérateur custom directement (farid, cubic, …).
    Aucun lissage gaussien - le signal est supposé déjà dans l'état voulu.
    Retourne (Hxx, Hxy, Hxz, Hyy, Hyz, Hzz).
    """
    fkey = METHOD_TO_FILTER[method]
    p = FILTERS[fkey]['p'].astype(np.float64)
    d1 = FILTERS[fkey]['d1'].astype(np.float64)
    d2 = FILTERS[fkey]['d2'].astype(np.float64)
    v = volume.astype(np.float64)

    def c(arr, k, ax): return convolve1d(arr, k, axis=ax, mode=mode)

    Hxx = c(c(c(v, d2, 0), p, 1), p, 2)
    Hyy = c(c(c(v, p, 0), d2, 1), p, 2)
    Hzz = c(c(c(v, p, 0), p, 1), d2, 2)
    Hxy = c(c(c(v, d1, 0), d1, 1), p, 2)
    Hyz = c(c(c(v, p, 0), d1, 1), d1, 2)
    Hxz = c(c(c(v, d1, 0), p, 1), d1, 2)

    return Hxx, Hxy, Hxz, Hyy, Hyz, Hzz

def test_axes():
    """Test pour vérifier l'ordre des axes dans convolve1d"""
    # Créer un volume avec une impulsion au centre
    size = 7
    v = np.zeros((size, size, size))
    v[size//2, size//2, size//2] = 1.0
    
    # Noyau simple pour tester
    k = np.array([-1, 0, 1]) # dérivée première simple
    
    from scipy.ndimage import convolve1d
    
    # Tester chaque axe
    res0 = convolve1d(v, k, axis=0)
    res1 = convolve1d(v, k, axis=1)
    res2 = convolve1d(v, k, axis=2)
    
    # Trouver où est la réponse
    for ax, res in enumerate([res0, res1, res2]):
        max_pos = np.unravel_index(np.argmax(np.abs(res)), res.shape)
        print(f"Axe {ax}: réponse maximale à {max_pos}")
    
    # Normalement, axis=0 doit donner variation sur X
    # Si axis=0 donne variation sur Z, alors l'ordre est inversé




def apply_skimage(volume: np.ndarray, method: str,
                  sigma: float = 0.5, mode: str = 'reflect') -> tuple:
    """
    Applique default ou gaussian de skimage.
    sigma=0.5 : minimum technique pour neutraliser au maximum
    le lissage interne tout en évitant les artefacts numériques.
    """
    from skimage.feature import hessian_matrix
    use_gauss = (method == 'gaussian')
    raw = hessian_matrix(volume, sigma=sigma, mode=mode,
                         order='rc', use_gaussian_derivatives=use_gauss)
    # skimage retourne (Hxx, Hxy, Hxz, Hyy, Hyz, Hzz) en ordre 'rc'
    return tuple(raw)


def apply_operator(volume: np.ndarray, method: str) -> np.ndarray:
    """Appelle l'opérateur et retourne le tenseur (*shape, 3, 3)."""
    if method in METHODS_CUSTOM:
        raw = apply_custom(volume, method)
    else:
        raw = apply_skimage(volume, method)
    return _to_tensor(raw, volume.shape)


def _to_tensor(raw, shape) -> np.ndarray:
    assert len(raw) == 6
    H = np.zeros((*shape, 3, 3), dtype=np.float64)
    H[..., 0, 0] = raw[0] # Hxx
    H[..., 0, 1] = H[..., 1, 0] = raw[1] # Hxy
    H[..., 0, 2] = H[..., 2, 0] = raw[2] # Hxz
    H[..., 1, 1] = raw[3] # Hyy
    H[..., 1, 2] = H[..., 2, 1] = raw[4] # Hyz
    H[..., 2, 2] = raw[5] # Hzz
    return H


# 3. GROUND TRUTH ANALYTIQUE EXACTE

class AnalyticalVessel:
    """
    Cylindre gaussien 3D : f(x,y,z) = exp(-r_perp² / 2σ²)
    r_perp = distance au segment central (axe du vaisseau)

    Hessienne analytique exacte calculée en forme fermée.
    Aucune approximation numérique.
    """

    def __init__(self, shape=(64, 64, 64)):
        self.shape = shape
        c = np.array(shape) / 2.0
        x = np.arange(shape[0]) - c[0]
        y = np.arange(shape[1]) - c[1]
        z = np.arange(shape[2]) - c[2]
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')

    def _geometry(self, start, end):
        """Retourne (r_perp, u, v, w, t) pour un segment."""
        start = np.asarray(start, float)
        end = np.asarray(end, float)
        w = end - start
        L = np.linalg.norm(w)
        w /= L

        pts = np.stack([self.X + self.shape[0]/2,
                        self.Y + self.shape[1]/2,
                        self.Z + self.shape[2]/2], axis=-1)
        dp = pts - start
        t = np.clip(np.tensordot(dp, w, axes=([-1],[0])), 0.0, L)
        closest = start + t[..., np.newaxis] * w
        diff = pts - closest # vecteur transverse

        r_perp = np.linalg.norm(diff, axis=-1)
        return diff, r_perp, w

    def _single_segment_signal_and_hessian(self, start, end, sigma):
        """
        Retourne (f, H_analytic) pour un segment gaussien.
        H_analytic shape = (*shape, 3, 3).
        """
        diff, r, w = self._geometry(start, end)
        s2 = sigma ** 2
        s4 = sigma ** 4

        f = np.exp(-r**2 / (2.0 * s2))
        f[r > VESSEL_RADIUS * 1.5] = 0.0

        # Coordonnées transverses : diff = (dx, dy, dz)
        dx = diff[..., 0]
        dy = diff[..., 1]
        dz = diff[..., 2]

        # Hessienne analytique d'un cylindre gaussien :
        # H_ij = f · [ (r_i · r_j / σ⁴) - δ_ij_perp / σ² ]
        # où δ_ij_perp = δ_ij - w_i·w_j (projection sur plan transverse)
        H = np.zeros((*self.shape, 3, 3), dtype=np.float64)
        coords = [dx, dy, dz]
        for i in range(3):
            for j in range(3):
                delta_perp = (1.0 if i == j else 0.0) - w[i] * w[j]
                H[..., i, j] = f * (coords[i] * coords[j] / s4
                                    - delta_perp / s2)

        return f, H

    def build(self, segments, sigma=SIGMA_VESSEL):
        """
        Construit (volume, H_gt, mask) depuis une liste de segments.
        segments : liste de (start, end)
        Retourne le signal discrétisé directement depuis la formule
        analytique aucun gaussian_filter.
        """
        vol = np.zeros(self.shape, dtype=np.float64)
        H_gt = np.zeros((*self.shape, 3, 3), dtype=np.float64)

        for (start, end) in segments:
            f_seg, H_seg = self._single_segment_signal_and_hessian(
                start, end, sigma)
            vol += f_seg
            H_gt += H_seg

        vol = np.clip(vol, 0.0, None)
        mask = vol > 0.15 * vol.max()
        return vol, H_gt, mask



    def simple_vessel(self):
        c = np.array(self.shape) / 2.0
        segs = [((c[0]-20, c[1], c[2]-20), (c[0]+20, c[1], c[2]+20))]
        axis = np.array([1.0, 0.0, -1.0]) # direction axiale
        axis /= np.linalg.norm(axis)
        # Transverse : n'importe quel vecteur ⊥ à axis
        transverse = np.array([1.0, 0.0, 1.0])
        transverse -= transverse.dot(axis) * axis
        transverse /= np.linalg.norm(transverse)
        vol, H, mask = self.build(segs)
        return vol, H, mask, transverse

    def curved_vessel(self):
        c = np.array(self.shape) / 2.0
        segs = [
            ((c[0]-25, c[1], c[2]), (c[0], c[1], c[2]+15)),
            ((c[0], c[1], c[2]+15), (c[0]+15, c[1], c[2]+25)),
            ((c[0]+15, c[1], c[2]+25), (c[0]+25, c[1], c[2]+10)),
        ]
        # Direction transverse moyenne (vaisseau courbe → moyenne des axes)
        axes = []
        for s, e in segs:
            v = np.array(e) - np.array(s)
            axes.append(v / np.linalg.norm(v))
        mean_axis = np.mean(axes, axis=0)
        mean_axis /= np.linalg.norm(mean_axis)
        transverse = np.array([1.0, 0.0, 0.0])
        transverse -= transverse.dot(mean_axis) * mean_axis
        transverse /= np.linalg.norm(transverse)
        vol, H, mask = self.build(segs)
        return vol, H, mask, transverse

    def bifurcation(self):
        c = np.array(self.shape) / 2.0
        segs = [
            ((c[0]-25, c[1], c[2]), (c[0], c[1], c[2])),
            ((c[0], c[1], c[2]), (c[0]-10, c[1], c[2]+25)),
            ((c[0], c[1], c[2]), (c[0]+10, c[1], c[2]+25)),
        ]
        transverse = np.array([0.0, 1.0, 0.0]) # direction y toujours transverse
        vol, H, mask = self.build(segs)
        return vol, H, mask, transverse


# 4. CALIBRATION PAR GAIN DIRECTIONNEL SUR BANDE

def compute_gain_on_band(method: str, transverse: np.ndarray,
                         sigma_vessel: float = SIGMA_VESSEL,
                         n_cal: int = 256, mode: str = 'reflect') -> float:
    """
    Calcule le facteur de normalisation α pour un opérateur sur la bande
    fréquentielle [0.5·ω_vessel, 1.5·ω_vessel] dans la direction transverse.

    Protocole :
      1. Générer un signal 1D : somme de sinusoïdes à fréquences ω ∈ bande
      2. Appliquer la dérivée seconde 1D de l'opérateur (filtre d2·p·p)
      3. Comparer à la réponse théorique -ω² · sin(ωx)
      4. α = mean(|réponse_théorique| / |réponse_opérateur|) sur la bande

    Le signal 1D est orienté selon `transverse` c'est la direction
    de courbure maximale du vaisseau.
    """
    omega_vessel = 1.0 / (2.0 * np.pi * sigma_vessel) # cycles/voxel
    omega_lo = 0.5 * omega_vessel
    omega_hi = 1.5 * omega_vessel

    x = np.arange(n_cal, dtype=np.float64)

    # Fréquences dans la bande
    freqs_all = np.fft.rfftfreq(n_cal) # cycles/voxel
    band_mask = (freqs_all >= omega_lo) & (freqs_all <= omega_hi)
    if band_mask.sum() == 0:
        # Bande trop étroite pour n_cal : prendre les 3 fréquences les plus proches
        idx = np.argsort(np.abs(freqs_all - omega_vessel))[:3]
        band_mask = np.zeros(len(freqs_all), dtype=bool)
        band_mask[idx] = True

    # Signal de calibration : somme des sinusoïdes dans la bande
    omegas_band = freqs_all[band_mask] * 2.0 * np.pi # rad/voxel
    signal_1d = np.zeros(n_cal)
    for om in omegas_band:
        signal_1d += np.sin(om * x)

    # Réponse théorique d²/dx²
    theoretical_1d = np.zeros(n_cal)
    for om in omegas_band:
        theoretical_1d += -om**2 * np.sin(om * x)

    # Réponse de l'opérateur (dérivée seconde dans la direction transverse)
    # On projette transverse sur les axes canoniques pour choisir
    # la composante de hessienne la plus alignée
    dominant_axis = int(np.argmax(np.abs(transverse)))

    if method in METHODS_CUSTOM:
        fkey = METHOD_TO_FILTER[method]
        p = FILTERS[fkey]['p'].astype(np.float64)
        d2 = FILTERS[fkey]['d2'].astype(np.float64)

        # Dérivée seconde 1D effective = d2 * p * p (dans la direction dominant)
        # Pour les deux autres axes on applique p (lissage)
        # Sur signal 1D on ne peut appliquer que d2 seul ou d2 convolved with p
        # On simule la réponse sur l'axe dominant_axis via convolution 1D
        resp_1d = convolve1d(signal_1d, d2, mode=mode)
        # Les deux autres axes contribuent via p·p mais sur signal constant
        # (signal 1D → pas de variation transverse) → gain = (sum p)² = 1
        # car les filtres p sont normalisés : sum(p) ≈ 1
        # Pas de correction nécessaire pour signal purement 1D.

    elif method == 'default':
        from skimage.feature import hessian_matrix
        # Appliquer sur signal 1D étendu en 3D (dimension minimale)
        vol3 = np.zeros((n_cal, 5, 5))
        vol3[:, 2, 2] = signal_1d
        H3 = hessian_matrix(vol3, sigma=0.5, mode=mode, order='rc',
                             use_gaussian_derivatives=False)
        resp_1d = H3[dominant_axis][..., 2, 2] if hasattr(H3[0], 'ndim') else \
                  H3[dominant_axis][:, 2, 2]
        # H3 retourne (Hxx,Hxy,Hxz,Hyy,Hyz,Hzz) → index dominant_axis
        idx_map = {0: 0, 1: 3, 2: 5}
        resp_1d = H3[idx_map[dominant_axis]][:, 2, 2]

    elif method == 'gaussian':
        from skimage.feature import hessian_matrix
        vol3 = np.zeros((n_cal, 5, 5))
        vol3[:, 2, 2] = signal_1d
        H3 = hessian_matrix(vol3, sigma=0.5, mode=mode, order='rc',
                             use_gaussian_derivatives=True)
        idx_map = {0: 0, 1: 3, 2: 5}
        resp_1d = H3[idx_map[dominant_axis]][:, 2, 2]

    else:
        raise ValueError(f"Méthode inconnue : {method}")

    # Gain = FFT(réponse_opérateur) / FFT(réponse_théorique) sur la bande
    R_op = np.fft.rfft(resp_1d)
    R_theo = np.fft.rfft(theoretical_1d)

    amp_op = np.abs(R_op[band_mask])
    amp_theo = np.abs(R_theo[band_mask])

    # Éviter division par zéro
    valid = amp_op > 1e-10
    if valid.sum() == 0:
        return 1.0 # opérateur silencieux sur la bande → α=1 par défaut

    gain = np.mean(amp_theo[valid] / amp_op[valid])
    return float(gain)


# 5. MÉTRIQUES

def frobenius_relative(H_op, H_gt, mask):
    """||H_op - H_gt||_F / ||H_gt||_F par voxel."""
    diff = H_op[mask] - H_gt[mask]
    e = np.sqrt(np.sum(diff**2, axis=(-2,-1)))
    n = np.sqrt(np.sum(H_gt[mask]**2, axis=(-2,-1)))
    return e / np.where(n > 1e-12, n, 1e-12)


def component_relative_errors_symmetric(H_op, H_gt, mask):
    labels = ['Hxx', 'Hxy', 'Hxz', 'Hyy', 'Hyz', 'Hzz']
    idxs = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
    results = {}
    
    for lbl, (i, j) in zip(labels, idxs):
        gt = H_gt[mask][..., i, j]
        op = H_op[mask][..., i, j]
        
        # Erreur symétrique bornée entre 0 et 1
        # (a-b)/(|a|+|b|+ε) est une mesure relative robuste
        denominator = np.abs(op) + np.abs(gt) + 1e-12
        err = np.abs(op - gt) / denominator
        results[lbl] = err
    
    return results


def amplitude_bias(H_op, H_gt, mask):
    """Biais d'amplitude : median(H_op/H_gt) - 1 sur les voxels du masque."""
    gt_flat = H_gt[mask].ravel()
    op_flat = H_op[mask].ravel()
    valid = np.abs(gt_flat) > 1e-12
    return float(np.median(op_flat[valid] / gt_flat[valid])) - 1.0


def symmetry_residual(H_op, mask):
    """||H - H^T||_F / ||H||_F par voxel."""
    Hm = H_op[mask]
    asym = Hm - np.transpose(Hm, (0,2,1))
    res = np.sqrt(np.sum(asym**2, axis=(-2,-1)))
    nrm = np.sqrt(np.sum(Hm**2, axis=(-2,-1)))
    return res / np.where(nrm > 1e-12, nrm, 1e-12)


def eigvals_sorted(H):
    v = np.linalg.eigvalsh(H.reshape(-1,3,3))
    sh = H.shape[:-2]
    return v[:,0].reshape(sh), v[:,1].reshape(sh), v[:,2].reshape(sh)


# 6. ÉVALUATION D'UN OPÉRATEUR SUR UN CAS

def evaluate(volume, H_gt, mask, method, transverse, alpha):
    """
    Évalue `method` sur `volume`.
    alpha : facteur de normalisation (gain directionnel sur bande).
    """
    H_raw = apply_operator(volume, method)
    H_op = H_raw * alpha # normalisation par gain

    frob = frobenius_relative(H_op, H_gt, mask)
    comp = component_relative_errors_symmetric(H_op, H_gt, mask)
    bias = amplitude_bias(H_op, H_gt, mask)
    sym = symmetry_residual(H_op, mask)

    row = {
        'method': method,
        'has_own_smooth': method in METHODS_SKIMG,
        'alpha': alpha,
        'frob_mean': float(np.mean(frob)),
        'frob_median': float(np.median(frob)),
        'frob_std': float(np.std(frob)),
        'frob_p95': float(np.percentile(frob, 95)),
        'bias': bias,
        'sym_mean': float(np.mean(sym)),
        'n_voxels': int(mask.sum()),
    }
    for lbl, vals in comp.items():
        row[f'comp_{lbl}_mean'] = float(np.mean(vals))
        row[f'comp_{lbl}_p95'] = float(np.percentile(vals, 95))

    return row, H_op


def measure_time(volume, method, n_repeats=N_REPEATS, warmup=WARMUP):
    times = []
    for i in range(warmup + n_repeats):
        t0 = time.perf_counter()
        apply_operator(volume, method)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return float(np.median(times)), float(np.std(times))


# 7. BRUIT DE NYQUIST

def add_nyquist_noise(volume, snr_db=SNR_DB):
    """
    Bruit blanc dont la puissance est concentrée à la fréquence de Nyquist.
    Cible les hautes fréquences que les opérateurs doivent gérer.
    SNR mesuré sur les voxels du vaisseau à la fréquence de Nyquist.
    """
    rng = np.random.default_rng(SEED)
    # Bruit blanc classique spectre plat jusqu'à Nyquist
    noise_raw = rng.standard_normal(volume.shape)

    # Amplifier les hautes fréquences (filtre passe-haut simple)
    from scipy.ndimage import gaussian_filter as gf
    noise_hf = noise_raw - gf(noise_raw, sigma=1.5)

    sp = float(np.var(volume[volume > 0.15 * volume.max()]))
    sn = np.sqrt(sp / 10**(snr_db / 10.0))
    scale = sn / (float(np.std(noise_hf)) + 1e-12)
    return np.clip(volume + noise_hf * scale, 0.0, None)


# 8. BENCHMARK PRINCIPAL

CASES = {
    'simple': 'simple_vessel',
    'curved': 'curved_vessel',
    'bifurcation': 'bifurcation',
}


def run_benchmark(output_dir='hessian_benchmark', verbose=True):
    gen = AnalyticalVessel(shape=SHAPE)
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    all_rows = []

    for case_name, method_name in CASES.items():
        builder = getattr(gen, method_name)
        vol, H_gt, mask, transverse = builder()

        if verbose:
            frob_gt = np.sqrt(np.sum(H_gt[mask]**2, axis=(-2,-1))).mean()
            print(f"\n[{case_name}] voxels={mask.sum()} "
                  f"||H_gt||_F_mean={frob_gt:.4f} "
                  f"transverse={np.round(transverse,3)}")

        # Calibration : α par opérateur par cas
        alphas = {}
        for m in METHODS:
            alphas[m] = compute_gain_on_band(m, transverse)
            if verbose:
                print(f"α[{m:>14s}] = {alphas[m]:.4f}")

        variants = {'clean': vol, 'nyquist': add_nyquist_noise(vol)}

        for vname, volume in variants.items():
            for method in tqdm(METHODS, desc=f"{case_name}/{vname}",
                               disable=not verbose):
                try:
                    row, _ = evaluate(volume, H_gt, mask, method,
                                      transverse, alphas[method])
                    t_med, t_std = measure_time(volume, method)
                    row['test_case'] = f"{case_name}_{vname}"
                    row['case'] = case_name
                    row['variant'] = vname
                    row['noisy'] = (vname == 'nyquist')
                    row['time_median_s'] = t_med
                    row['time_std_s'] = t_std
                    all_rows.append(row)
                except Exception as exc:
                    if verbose:
                        print(f"{method}: {exc}")
                    import traceback; traceback.print_exc()

    df = pd.DataFrame(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"benchmark_{ts}.csv"
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\nCSV → {csv_path}")
    return df, out_dir


# 9. RAPPORT TEXTE

def print_report(df):
    sep = "="* 72
    clean = df[df['variant'] == 'clean']
    
    # Separation des familles
    pure_derivatives = [m for m in METHODS_CUSTOM if m in clean['method'].values]
    skimage_methods = [m for m in METHODS_SKIMG if m in clean['method'].values]
    
    pure_clean = clean[clean['method'].isin(pure_derivatives)]
    skimage_clean = clean[clean['method'].isin(skimage_methods)]

    print(f"\n{sep}")
    print("BENCHMARK HESSIENNE - GT ANALYTIQUE + NORMALISATION GAIN")
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    print(sep)

    # 1. RESULTATS PAR CAS (toutes methodes, informatif)
    print("\n RESULTATS PAR CAS")
    print("-"* 60)
    for case in sorted(clean['case'].unique()):
        sub = clean[clean['case'] == case]
        bf = sub.loc[sub['frob_mean'].idxmin()]
        ba = sub.loc[sub['bias'].abs().idxmin()]
        bt = sub.loc[sub['time_median_s'].idxmin()]
        print(f"\n CAS : {case.upper()}")
        print(f"Meilleur Frobenius : {bf['method']:<16} {bf['frob_mean']:.4f} +- {bf['frob_std']:.4f}")
        print(f"Biais minimal : {ba['method']:<16} {ba['bias']:+.4f}")
        print(f"Plus rapide : {bt['method']:<16} {bt['time_median_s']:.4f} s")

    # 2. CLASSEMENT PRINCIPAL - DERIVATION PURE
    print(f"\n{sep}")
    print("CLASSEMENT PRINCIPAL - DERIVATION PURE")
    print("(farid, cubic, trigonometric, catmull, bspline, bezier)")
    print(sep)

    if not pure_clean.empty:
        agg = (pure_clean.groupby('method')
               .agg(frobenius=('frob_mean', 'mean'),
                    frob_p95=('frob_p95', 'mean'),
                    bias=('bias', lambda x: x.abs().mean()),
                    symetrie=('sym_mean', 'mean'),
                    temps=('time_median_s', 'mean'))
               .round(5))
        
        agg['rk_f'] = agg['frobenius'].rank()
        agg['rk_b'] = agg['bias'].rank()
        agg['rk_t'] = agg['temps'].rank()
        agg['score'] = agg[['rk_f', 'rk_b', 'rk_t']].mean(axis=1)
        agg = agg.sort_values('score')

        hdr = f"{'Rg':<3} {'Methode':<16} {'Frob':>8} {'p95':>8} {'|Biais|':>8} {'Sym':>10} {'t(s)':>8} {'Score':>7}"
        print(hdr)
        print("-"* len(hdr))
        for rank, (m, r) in enumerate(agg.iterrows(), 1):
            print(f"{rank:<3} {m:<16} {r['frobenius']:>8.4f} {r['frob_p95']:>8.4f} "
                  f"{r['bias']:>8.4f} {r['symetrie']:>10.6f} {r['temps']:>8.4f} {r['score']:>7.2f}")
    else:
        print("Aucune methode de derivation pure trouvee.")

    # 3. SECTION INFORMATIVE - METHODES SKIMAGE
    print(f"\n{sep}")
    print("METHODES SKIMAGE (lissage interne)")
    print("(default, gaussian) - Non comparables directement")
    print(sep)

    if not skimage_clean.empty:
        for m, g in skimage_clean.groupby('method'):
            print(f"\n {m.upper()}:")
            print(f"Frobenius : {g['frob_mean'].mean():.4f} +- {g['frob_std'].mean():.4f}")
            print(f"Biais : {g['bias'].mean():+.4f}")
            print(f"alpha : {g['alpha'].mean():.4f}")
            print(f"Temps : {g['time_median_s'].mean():.4f} s")
    else:
        print("Aucune methode skimage trouvee.")

    # 4. ROBUSTESSE AU BRUIT
    print(f"\n{sep}")
    print("ROBUSTESSE AU BRUIT (Delta Frobenius : nyquist - clean)")
    print(sep)
    
    custom_noise = df[df['method'].isin(pure_derivatives)]
    piv = custom_noise.groupby(['method', 'variant'])['frob_mean'].mean().unstack()
    if 'nyquist'in piv and 'clean'in piv:
        delta = (piv['nyquist'] - piv['clean']).sort_values()
        for m, d in delta.items():
            print(f"{m:<16} Delta = {d:+.4f}")
    else:
        print("Donnees de bruit non disponibles.")


# 10. VISUALISATION

def plot_results(df, out_dir, save=True):
    sns.set_style('whitegrid')
    pal = sns.color_palette('tab10', n_colors=len(METHODS))
    mcol = dict(zip(METHODS, pal))

    clean = df[df['variant'] == 'clean']
    fair = clean[~clean['has_own_smooth']]

    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.40)

    # (0,0) Frobenius par cas
    ax = fig.add_subplot(gs[0, 0])
    fair.pivot_table('frob_mean', 'method', 'case').plot(
        kind='bar', ax=ax, colormap='viridis', width=0.7)
    ax.set_title('Frobenius relatif par cas\n(après normalisation gain)',
                 fontweight='bold')
    ax.set_ylabel('||H_op − H_gt||_F / ||H_gt||_F')
    ax.tick_params(axis='x', rotation=35)
    ax.legend(title='Cas', fontsize=8)

    # (0,1) Biais d'amplitude
    ax = fig.add_subplot(gs[0, 1])
    bias_data = fair.groupby(['method','case'])['bias'].mean().unstack()
    bias_data.plot(kind='bar', ax=ax, colormap='coolwarm', width=0.7)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_title('Biais d\'amplitude\n(median(H_op/H_gt)−1, 0=parfait)',
                 fontweight='bold')
    ax.set_ylabel('Biais relatif')
    ax.tick_params(axis='x', rotation=35)

    # (0,2) Temps médian
    ax = fig.add_subplot(gs[0, 2])
    tdf = fair.groupby('method')['time_median_s'].mean().sort_values()
    bars = ax.barh(tdf.index, tdf.values,
                   xerr=fair.groupby('method')['time_std_s'].mean().reindex(tdf.index),
                   color=[mcol[m] for m in tdf.index], capsize=4)
    ax.set_title('Temps médian (s)', fontweight='bold')
    for bar, val in zip(bars, tdf.values):
        ax.text(val*1.02, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}s', va='center', fontsize=8)

    # (1,0) Heatmap rangs
    ax = fig.add_subplot(gs[1, 0])
    agg = fair.groupby('method').agg(
        frobenius=('frob_mean', 'mean'),
        biais =('bias', lambda x: x.abs().mean()),
        temps =('time_median_s', 'mean'),
    )
    rk = agg.rank()
    rk['score'] = rk.mean(axis=1)
    rk = rk.sort_values('score')
    sns.heatmap(rk[['frobenius','biais','temps','score']], ax=ax,
                annot=True, fmt='.1f', cmap='RdYlGn_r', linewidths=0.5,
                xticklabels=['Frobenius','|Biais|','Temps','Score'],
                cbar_kws={'label': 'Rang (1=meilleur)'})
    ax.set_title('Classement global (clean, sans lissage propre)',
                 fontweight='bold')

    # (1,1) Erreurs par composante
    ax = fig.add_subplot(gs[1, 1])
    comp_cols = [c for c in fair.columns
                 if c.startswith('comp_') and c.endswith('_mean')]
    comp_data = fair.groupby('method')[comp_cols].mean()
    comp_data.columns = [c.replace('comp_','').replace('_mean','')
                         for c in comp_data.columns]
    comp_data.T.plot(kind='bar', ax=ax, width=0.75)
    ax.set_title('Erreur relative par composante\n(clean, moyenne sur cas)',
                 fontweight='bold')
    ax.set_ylabel('Erreur relative')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=7, ncol=2)

    # (1,2) Dégradation bruit
    ax = fig.add_subplot(gs[1, 2])
    piv = df[~df['has_own_smooth']].groupby(
        ['method','variant'])['frob_mean'].mean().unstack()
    if 'nyquist'in piv and 'clean'in piv:
        delta = (piv['nyquist'] - piv['clean']).sort_values()
        colors = ['tomato'if v > 0 else 'steelblue'for v in delta.values]
        ax.barh(delta.index, delta.values, color=colors)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_title('Δ Frobenius (nyquist − clean)\n(robustesse bruit HF)',
                     fontweight='bold')
        ax.set_xlabel('Δ erreur relative')

    # (2,0) Frobenius p95
    ax = fig.add_subplot(gs[2, 0])
    fair.groupby('method')['frob_p95'].mean().sort_values().plot(
        kind='barh', ax=ax, color='steelblue')
    ax.set_title('Frobenius p95\n(robustesse voxels difficiles)',
                 fontweight='bold')

    # (2,1) Résidu symétrie
    ax = fig.add_subplot(gs[2, 1])
    fair.groupby('method')['sym_mean'].mean().sort_values().plot(
        kind='barh', ax=ax, color='mediumseagreen')
    ax.set_title('Résidu de symétrie moyen\n(||H−Hᵀ||/||H|| ≈ 0 attendu)',
                 fontweight='bold')

    # (2,2) Facteurs α par méthode
    ax = fig.add_subplot(gs[2, 2])
    alpha_data = (df[~df['has_own_smooth']]
                  .groupby(['method','case'])['alpha'].mean().unstack())
    alpha_data.plot(kind='bar', ax=ax, colormap='Set2', width=0.7)
    ax.axhline(1.0, color='k', lw=0.8, ls='--', label='α=1 (gain unitaire)')
    ax.set_title('Facteurs de normalisation α\n(1=gain correct, >1=sous-estimation)',
                 fontweight='bold')
    ax.set_ylabel('α')
    ax.tick_params(axis='x', rotation=35)
    ax.legend(fontsize=7)

    plt.suptitle(
        'Benchmark Hessienne GT analytique exacte + normalisation gain directionnel',
        fontsize=14, fontweight='bold', y=1.01
    )

    if save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"benchmark_plot_{ts}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Figure → {out}")
    return fig


# 11. POINT D'ENTRÉE

if __name__ == '__main__':
    for method in METHODS_CUSTOM:
        fkey = METHOD_TO_FILTER[method]
        d2 = FILTERS[fkey]['d2']
        print(f"{method}: d2 = {d2}")
        print(f"sum(d2) = {d2.sum()}")
        print(f"max(d2) = {d2.max()}")
    test_axes()
    df, out_dir = run_benchmark(verbose=True)
    print_report(df)
    plot_results(df, out_dir, save=True)
    plt.show()