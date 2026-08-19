# inspiré de l'article Lamy et al. (ICPR 2020)
# GT = seuillage > 0 de l'image brute, PAS de rastérisation

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
from scipy.stats import rice


# =========================================================================== #
# Constantes exactes de l'article
# =========================================================================== #
NOISE_LEVELS: Tuple[float, ...] = (5.0, 10.0, 20.0)   # sigma du bruit Rician
BIAS_SIGMA: float = 40.0                               # taille du bias field
INTENSITY_IMIN: float = 50.0
INTENSITY_IMAX: float = 100.0

# Complexités ciblées : data7→31, data9→41, data11→51
TARGET_DATA_INDICES = (7, 9, 11)


# =========================================================================== #
# Extraction des nœuds (.mat) -
# =========================================================================== #
def _extract_node_fields(mat_path: Path):
    """Charge le .mat et retourne (coords, parents) pour chaque noeud."""
    data = loadmat(str(mat_path), squeeze_me=False, struct_as_record=True)
    if "node" not in data:
        raise KeyError(
            f"Clé 'node' introuvable dans {mat_path.name}. "
            f"Clés disponibles : {[k for k in data if not k.startswith('__')]}"
        )
    node = data["node"].reshape(-1)
    n = node.shape[0]

    coords = np.zeros((n, 3), dtype=np.float64)
    parents = np.full(n, -1, dtype=np.int64)

    for i in range(n):
        rec = node[i]

        coord = np.asarray(rec["coord"]).reshape(-1).astype(np.float64)
        coords[i] = coord[:3]

        parent_arr = np.asarray(rec["parent"]).reshape(-1)
        if parent_arr.size > 0 and parent_arr[0] > 0:
            parents[i] = int(parent_arr[0]) - 1

    return coords, parents


# =========================================================================== #
# Fonctions de pré-traitement (répliques exactes du code des auteurs)
# =========================================================================== #
def _gauss3d(x, y, z, mx, my, mz, sx, sy, sz):
    """Réplique exacte de la gaussienne du code original."""
    return (
        1.0
        / (sx * sy * sz * np.sqrt(2.0 * np.pi) * np.sqrt(2.0 * np.pi))
        * np.exp(
            -(
                (x - mx) ** 2 / (2.0 * sx ** 2)
                + (y - my) ** 2 / (2.0 * sy ** 2)
                + (z - mz) ** 2 / (2.0 * sz ** 2)
            )
        )
    )


def vessels_and_background(dat: np.ndarray, Imin: float = INTENSITY_IMIN, Imax: float = INTENSITY_IMAX) -> np.ndarray:
    """
    Réplique ligne à ligne de vasculature.Generator.vesselsAndBackground.
    """
    dat = dat.astype(np.float64)
    dat = dat / np.max(dat) * (Imax - Imin) + Imin

    # Remplir le fond (voxels = 0) avec la valeur positive minimale
    min_value = np.min(dat[dat > 0]) if np.any(dat > 0) else 0.0
    dat[dat == 0] = min_value

    return np.clip(dat, 0, 255).astype(np.uint8)


def vessels_illumination(dat: np.ndarray, sigma: float = BIAS_SIGMA,
                          Imin: float = INTENSITY_IMIN, Imax: float = INTENSITY_IMAX) -> np.ndarray:
    """
    Réplique fidèle de Generator.vesselsIllumination.
    Ajoute un artefact d'illumination (bias field) avec 3 gaussiennes.
    """
    dat = dat.astype(np.float32)
    end_x, end_y, end_z = dat.shape
    start_x = start_y = start_z = 0

    step_x = end_x - start_x
    step_y = end_y - start_y
    step_z = end_z - start_z

    x = np.linspace(start_x, end_x, step_x)
    y = np.linspace(start_y, end_y, step_y)
    z = np.linspace(start_z, end_z, step_z)

    half_x, half_y, half_z = end_x / 2, end_y / 2, end_z / 2

    # indexing='xy' comme dans le code original (comportement par défaut de np.meshgrid)
    x, y, z = np.meshgrid(x, y, z, indexing='xy')

    d1 = _gauss3d(x, y, z, mx=0, my=0, mz=0, sx=sigma, sy=sigma, sz=sigma)
    d2 = _gauss3d(x, y, z, mx=half_x, my=half_y, mz=half_z, sx=sigma, sy=sigma, sz=sigma)
    d3 = _gauss3d(x, y, z, mx=end_x, my=end_y, mz=end_z, sx=sigma, sy=sigma, sz=sigma)

    d = d1 + d2 + d3
    d = d / d.max() * (Imax - Imin) + Imin

    dat = np.maximum(d, dat)
    dat = np.clip(dat, 0, 255)
    return dat.astype(np.float32)


def rician_noise_variants(dat: np.ndarray, noise_levels: Tuple[float, ...] = NOISE_LEVELS) -> List[Tuple[float, np.ndarray]]:
    """
    Réplique exacte de Generator.noisyImage(..., noiseType="rician").
    """
    dat = dat.astype(np.float32)
    out = []
    for sigma in noise_levels:
        noisy = rice.rvs(dat / sigma, scale=sigma)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        out.append((sigma, noisy))
    return out


def ground_truth_from_raw_image(raw_dat: np.ndarray) -> np.ndarray:
    """
    GT = (image brute > 0) comme décrit dans l'article :
    "the ground-truths simply correspond to the binary volumes used for simulating the images"
    """
    return (raw_dat > 0).astype(np.uint8)


# =========================================================================== #
# Masque des bifurcations (ROI 3)
# =========================================================================== #
def bifurcation_mask(mat_path: Path, volume_shape_xyz: Tuple[int, int, int]) -> np.ndarray:
    """
    Réplique de Generator.bifurcationsPositionsGT.
    Noyau gaussien 7x7x7 (sigma=1, normalisé par son max) à chaque bifurcation.

    IMPORTANT (fidélité bit-exacte) : le code original des auteurs contient un
    off-by-one dans la construction du noyau :
        for i in range( -halfKS, halfKS ):   # PAS de +1 !
    Avec halfKS=3 sur un noyau 7x7x7, ceci ne couvre QUE i,j,k in [-3, 2]
    (6 valeurs, pas 7) : la dernière tranche du noyau (indice 6 après
    décalage) n'est jamais remplie et reste à 0. Ce n'est pas une erreur
    d'arrondi anodine : ça change la forme réelle du blob déposé à chaque
    bifurcation. Pour rester bit-exact avec l'article, on réplique CE
    comportement (bug inclus) plutôt que de le "corriger" avec range(-3, 4).
    """
    coords, parents = _extract_node_fields(mat_path)

    # Trouver les nœuds de bifurcation (ceux qui ont 2 enfants)
    n = len(parents)
    child_count = np.zeros(n, dtype=np.int64)
    for idx in range(n):
        p = parents[idx]
        if p >= 0:
            child_count[p] += 1
    bif_indices = [i for i in range(n) if child_count[i] == 2]

    shape = tuple(int(s) for s in volume_shape_xyz)
    out = np.zeros(shape, dtype=np.float32)

    # Noyau gaussien 7x7x7 -- bug d'origine répliqué (voir docstring ci-dessus)
    kernel_size = 7
    sigma = 1.0
    half_k = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size, kernel_size), dtype=np.float32)
    for i in range(-half_k, half_k):        # PAS de +1 : réplique le off-by-one des auteurs
        for j in range(-half_k, half_k):
            for k in range(-half_k, half_k):
                kernel[i + half_k, j + half_k, k + half_k] = np.exp(
                    -(i ** 2 + j ** 2 + k ** 2) / (3.0 * sigma * sigma)
                )
    kernel = kernel / np.max(kernel)

    # Déposer le noyau à chaque bifurcation
    for idx in bif_indices:
        x, y, z = (int(round(c)) for c in coords[idx])
        x0, x1 = x - half_k, x + half_k + 1
        y0, y1 = y - half_k, y + half_k + 1
        z0, z1 = z - half_k, z + half_k + 1

        if x0 < 0 or y0 < 0 or z0 < 0 or x1 > shape[0] or y1 > shape[1] or z1 > shape[2]:
            continue

        patch = np.maximum(out[x0:x1, y0:y1, z0:z1], kernel)
        out[x0:x1, y0:y1, z0:z1] = patch

    return out


def vessel_neighborhood_mask(gt: np.ndarray, dilation_iterations: int = 3) -> np.ndarray:
    """
    Crée le masque du voisinage des vaisseaux (ROI 2) par dilatation du GT.

    ATTENTION -- CE N'EST PAS UNE REPRODUCTION VÉRIFIÉE :
    le rayon de dilatation exact utilisé par les auteurs (dans l'exécutable
    C++ fermé MakeVascuSynthGT, qui produit "gtDilated.nii") n'est documenté
    nulle part publiquement (ni dans le papier, ni dans le code source
    disponible). La valeur `dilation_iterations=3` ci-dessous est un DÉFAUT
    ARBITRAIRE, pas une valeur confirmée. Si ce ROI est utilisé pour publier
    des résultats, il faut soit la calibrer visuellement contre la Fig. 2(h)
    de l'article, soit la documenter explicitement comme une hypothèse.
    """
    return binary_dilation(gt.astype(bool), iterations=dilation_iterations).astype(np.uint8)


# =========================================================================== #
# (Déprécié) Gardé pour compatibilité mais non utilisé par l'article
# =========================================================================== #
def _draw_segment(c1, c2, volume, shape):
    """Déprécié - gardé pour compatibilité."""
    c1, c2 = np.array(c1, float), np.array(c2, float)
    n_steps = int(np.ceil(np.linalg.norm(c2 - c1))) + 1
    for t in np.linspace(0, 1, max(n_steps, 2)):
        p = np.round(c1 + t * (c2 - c1)).astype(int)
        if all(0 <= p[i] < shape[i] for i in range(3)):
            volume[p[0], p[1], p[2]] = 1


def rasterize_tree_to_mask(mat_path: Path, volume_shape_xyz: Tuple[int, int, int],
                            dilation_iterations: int = 3) -> np.ndarray:
    """
    DÉPRÉCIÉ - Cette méthode de rastérisation n'est PAS utilisée par l'article.
    L'article utilise ground_truth_from_raw_image() à la place.
    Gardé uniquement pour compatibilité.
    """
    print(f"[WARNING] rasterize_tree_to_mask est déprécié. "
          f"L'article utilise ground_truth_from_raw_image() à la place.")
    coords, parents = _extract_node_fields(mat_path)
    shape = tuple(int(s) for s in volume_shape_xyz)
    mask = np.zeros(shape, dtype=np.uint8)

    for idx, coord in enumerate(coords):
        xi, yi, zi = int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))
        if all(0 <= v < shape[i] for i, v in enumerate([xi, yi, zi])):
            mask[xi, yi, zi] = 1

    for idx in range(len(coords)):
        parent_idx = parents[idx]
        if parent_idx < 0:
            continue
        _draw_segment(coords[parent_idx], coords[idx], mask, shape)

    if dilation_iterations > 0:
        mask = binary_dilation(mask.astype(bool), iterations=dilation_iterations).astype(np.uint8)

    return mask