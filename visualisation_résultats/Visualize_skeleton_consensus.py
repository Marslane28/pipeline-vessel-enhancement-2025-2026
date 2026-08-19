#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import pickle
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import binary_dilation as ndi_dilation
from scipy.ndimage import label, find_objects

warnings.filterwarnings("ignore")

# CONFIGURATION

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "visualisations_consensus_MFAT_mip"
OUTPUT_DIR.mkdir(exist_ok=True)

OPERATORS = ['default', 'gaussian', 'farid', 'cubic', 'trigonometric',
             'catmull', 'bspline', 'bezier', 'scharr']

DATASET_LABELS = {
    "ircad": "IRCAD",
    "bullitt": "Bullitt",
    "vascusynth": "VascuSynth",
}
BENCHMARK_ROOT = PROJECT_ROOT / "outputs"

@dataclass
class DatasetConfig:
    enhancer_dir: Path
    labels_dir: Path
    masks_dir: Optional[Path]
    case_prefix: str
    case_suffix: str
    result_prefix: str
    result_suffix: str
    patients: List[str]
    has_subdirs: bool = False


DATASETS: Dict[str, DatasetConfig] = {
    "bullitt": DatasetConfig(
        enhancer_dir=BENCHMARK_ROOT / "Jerman"/ "bullitt_enhancer_jerman_2026-07-11_14-25-54",
        labels_dir=PROJECT_ROOT / "data/bullitt/labels",
        masks_dir=PROJECT_ROOT / "data/bullitt/masks",
        case_prefix="patient_", case_suffix="_images.nii",
        result_prefix="results_patient_", result_suffix="_images.nii",
        patients=[f"{i:02d}"for i in range(1, 34)],
    ),
    "ircad": DatasetConfig(
        enhancer_dir=BENCHMARK_ROOT / "Jerman"/ "ircad_enhancer_jerman_2026-07-11_20-39-32",
        labels_dir=PROJECT_ROOT / "data/ircad/3d-échantillonnées/labels",
        masks_dir=PROJECT_ROOT / "data/ircad/3d-échantillonnées/masks",
        case_prefix="patient_", case_suffix="_images.nii",
        result_prefix="results_patient_", result_suffix="_images.nii",
        patients=[f"{i:02d}"for i in range(1, 21)],
    ),
    "vascusynth": DatasetConfig(
        enhancer_dir=BENCHMARK_ROOT / "Jerman"/ "vascusynth_enhancer_jerman_2026-07-18_09-32-01",
        labels_dir=PROJECT_ROOT / "data/vascusynth_preprocessed/labels",
        masks_dir=None,
        case_prefix="", case_suffix="",
        result_prefix="results_", result_suffix="",
        patients=[], has_subdirs=True,
    ),
}

DPI = 180
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


DILATION_SIZE = 1 
# Marge (en pixels) ajoutée autour de la bounding box lors du recadrage
CROP_MARGIN = 15

# Bornes des 4 catégories, exprimées en FRACTION du nombre d'opérateurs (n_ops)
# - unique : count == 1
# - minoritaire : 1 < count <= n_ops/2
# - majoritaire : n_ops/2 < count < n_ops
# - consensus_total : count == n_ops
CATEGORY_COLORS = {
    "unique": "#d7191c", # rouge franc
    "minoritaire": "#fdae61", # orange
    "majoritaire": "#a6d96a", # vert clair
    "consensus": "#1a9641", # vert foncé
}
CATEGORY_LABELS = {
    "unique": "1 seul opérateur",
    "minoritaire": "Minorité (2 à n/2)",
    "majoritaire": "Majorité (n/2+1 à n-1)",
    "consensus": "Consensus total (tous)",
}


BACKGROUND_CAT = -1

METRIC_NAMES = [
    'dice', 'mcc', 'roc', 'pr', 'cldice',
    'components_ratio', 'excess_components',
    'skeleton_component_connectivity', 'largest_component_recall',
    'gt_fragmentation', 'pred_small_components', 'pred_medium_components',
    'pred_large_components', 'largest_gt_recall', 'largest_component_overlap',
    'fragmentation_ratio', 'hessian_time_seconds',
    'bifurcation_detection_rate', 'bifurcation_precision',
    'n_bifurcations_gt', 'n_bifurcations_pred', 'n_bifurcations_detected',
]

# FONCTIONS DE BASE
def load_text_metrics(dataset: str, case_id: str, operator: str) -> Dict[str, Any]:
    cfg = DATASETS[dataset]
    patient_dir = (cfg.enhancer_dir / case_id if dataset == "vascusynth"
                    else cfg.enhancer_dir / f"{cfg.case_prefix}{case_id}{cfg.case_suffix}")
    metrics_file = patient_dir / "text_metrics.txt"
    if not metrics_file.exists():
        return {}
    try:
        text = metrics_file.read_text()
    except Exception:
        return {}
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if len(lines) < 3:
        return {}
    for line in lines[2:]:
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if parts and parts[0] == operator:
            metrics = {}
            for i, name in enumerate(METRIC_NAMES):
                if i + 1 < len(parts):
                    try:
                        metrics[name] = float(parts[i + 1])
                    except (ValueError, IndexError):
                        metrics[name] = None
            return metrics
    return {}

def load_text_config(dataset: str, case_id: str, operator: str) -> Dict[str, Any]:
    cfg = DATASETS[dataset]
    patient_dir = (cfg.enhancer_dir / case_id if dataset == "vascusynth"
                    else cfg.enhancer_dir / f"{cfg.case_prefix}{case_id}{cfg.case_suffix}")
    config_file = patient_dir / "text_configs.txt"
    if not config_file.exists():
        return {}
    try:
        text = config_file.read_text()
    except Exception:
        return {}
    pattern = rf"{operator.upper()}\s*\(best[^)]*\)\s*:"
    for match in re.finditer(pattern, text, re.IGNORECASE):
        start = match.end()
        brace_match = re.search(r'\s*(\{)', text[start:])
        if not brace_match:
            continue
        dict_start = start + brace_match.start()
        brace_count, dict_end = 0, dict_start
        for i, ch in enumerate(text[dict_start:], dict_start):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    dict_end = i + 1
                    break
        if dict_end <= dict_start:
            continue
        threshold_match = re.search(r"'threshold':\s*([0-9.]+)", text[dict_start:dict_end])
        if threshold_match:
            return {'segmentation': {'threshold': float(threshold_match.group(1))}}
    return {}

def _results_dir_for(dataset: str, case_id: str) -> Optional[Path]:
    cfg = DATASETS[dataset]
    if dataset == "vascusynth":
        enhancer_dir = cfg.enhancer_dir / case_id
    else:
        enhancer_dir = cfg.enhancer_dir
    if not enhancer_dir.exists():
        return None
    results_dir = enhancer_dir / "results"
    return results_dir if results_dir.exists() else None

def _find_result_file(dataset: str, case_id: str) -> Optional[Path]:
    
    cfg = DATASETS[dataset]
    results_dir = _results_dir_for(dataset, case_id)
    if results_dir is None:
        return None
    if dataset == "vascusynth":
        matching = list(results_dir.glob(f"results_{case_id}_rician_*.nii"))
        if not matching:
            matching = list(results_dir.glob(f"results_{case_id}.nii"))
        if not matching:
            return None
        return matching[0]
    result_file = results_dir / f"{cfg.result_prefix}{case_id}{cfg.result_suffix}"
    return result_file if result_file.exists() else None

def load_pickle_once(dataset: str, case_id: str, debug: bool = False) -> Optional[Dict[str, Any]]:

    result_file = _find_result_file(dataset, case_id)
    if result_file is None:
        return None
    try:
        with open(result_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        if debug:
            print(f"Erreur chargement {result_file}: {e}")
        return None
    return data.get('derivator', {})

def extract_segmentation(derivator_dict: Optional[Dict[str, Any]], operator: str) -> Optional[np.ndarray]:
    if not derivator_dict:
        return None
    for key in (operator, 'default'):
        if key not in derivator_dict:
            continue
        op_data = derivator_dict[key]
        if hasattr(op_data, 'data_segmented'):
            seg = op_data.data_segmented
            if seg is not None:
                return np.asarray(seg).astype(bool)
        elif isinstance(op_data, dict):
            seg = op_data.get('data_segmented')
            if seg is not None:
                return np.asarray(seg).astype(bool)
    return None

def load_ground_truth(dataset: str, case_id: str) -> Optional[np.ndarray]:
    import nibabel as nib
    cfg = DATASETS[dataset]
    if not cfg.labels_dir.exists():
        return None

    if dataset == "vascusynth":
        patient_folder = case_id.split("_rician")[0] if "_rician"in case_id else case_id
        gt_path = cfg.labels_dir / patient_folder / f"{patient_folder}_vessels_gt.nii.gz"
        if not gt_path.exists():
            gt_path = cfg.labels_dir / patient_folder / f"{patient_folder}_vessels_gt.nii"
        if not gt_path.exists():
            candidates = list((cfg.labels_dir / patient_folder).glob("*vessels_gt.nii*"))
            if not candidates:
                return None
            gt_path = candidates[0]
    else:
        candidates = list(cfg.labels_dir.glob(f"*{case_id}*.nii*"))
        if not candidates:
            return None
        gt_path = candidates[0]

    try:
        img = nib.load(str(gt_path))
        return np.asarray(img.get_fdata()) > 0.5
    except Exception as e:
        print(f"Erreur chargement GT {gt_path}: {e}")
        return None
    
def get_vascusynth_patients(enhancer_dir: Path) -> List[str]:
    if not enhancer_dir.exists():
        return []
    patients = []
    for item in enhancer_dir.iterdir():
        if item.is_dir() and item.name.startswith("Group") and "_data"in item.name:
            if (item / "text_metrics.txt").exists():
                patients.append(item.name)

    def sort_key(name):
        m = re.search(r"Group(\d+)_data(\d+)", name)
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)

    patients.sort(key=sort_key)
    return patients

def init_vascusynth_config():
    enhancer_dir = DATASETS["vascusynth"].enhancer_dir
    DATASETS["vascusynth"].patients = get_vascusynth_patients(enhancer_dir)

    
def load_all_metrics(debug: bool = False, with_gt: bool = True) -> pd.DataFrame:
    init_vascusynth_config()
    all_rows = []
    gt_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}

    for dataset_name, cfg in DATASETS.items():
        print(f"Chargement {dataset_name} ({len(cfg.patients)} patients)...")
        for case_id in tqdm(cfg.patients, desc=f"{dataset_name}"):
            patient_dir = (cfg.enhancer_dir / case_id if dataset_name == "vascusynth"
                            else cfg.enhancer_dir / f"{cfg.case_prefix}{case_id}{cfg.case_suffix}")
            if not (patient_dir / "text_metrics.txt").exists():
                continue

            gt = None
            if with_gt:
                key = (dataset_name, case_id)
                if key not in gt_cache:
                    gt_cache[key] = load_ground_truth(dataset_name, case_id)
                gt = gt_cache[key]

            # un seul chargement + dé-picklage pour les 9 opérateurs
            derivator_dict = load_pickle_once(dataset_name, case_id, debug=debug)

            for operator in OPERATORS:
                metrics = load_text_metrics(dataset_name, case_id, operator)
                if not metrics:
                    continue
                config = load_text_config(dataset_name, case_id, operator)
                threshold = config.get('segmentation', {}).get('threshold') if config else None
                seg = extract_segmentation(derivator_dict, operator)

                all_rows.append({
                    'dataset': dataset_name,
                    'case_id': case_id,
                    'operator': operator,
                    'threshold': threshold,
                    'segmentation': seg,
                    'gt': gt,
                    **metrics
                })

        n_loaded = len([r for r in all_rows if r['dataset'] == dataset_name])
        n_with_gt = len([r for r in all_rows if r['dataset'] == dataset_name and r['gt'] is not None])
        print(f"{dataset_name}: {n_loaded} lignes chargées ({n_with_gt} avec GT trouvé)")

    print(f"Total: {len(all_rows)} lignes chargées")
    return pd.DataFrame(all_rows)

def count_components(mask: np.ndarray) -> int:
    if mask is None or mask.sum() == 0:
        return 0
    labeled, n = label(mask, structure=CONNECTIVITY_26)
    return n


def compute_skeleton(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.zeros((1, 1, 1), dtype=bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    try:
        return skeletonize(mask)
    except Exception as e:
        print(f"Warning: squelette impossible: {e}")
        return np.zeros_like(mask, dtype=bool)


def load_segmentation_for_case(dataset: str, case_id: str) -> Dict[str, Optional[np.ndarray]]:
    cfg = DATASETS[dataset]

    if dataset == "vascusynth":
        if "_rician"in case_id:
            patient_folder = case_id.split("_rician")[0]
        else:
            patient_folder = case_id
        patient_dir = cfg.mfat_dir / patient_folder
    else:
        patient_dir = cfg.mfat_dir / f"{cfg.case_prefix}{case_id}{cfg.case_suffix}"

    results_dir = patient_dir.parent / "results"
    if not results_dir.exists():
        return {}

    if dataset == "vascusynth":
        pattern = f"results_{patient_folder}_rician_*.nii"
        candidates = list(results_dir.glob(pattern))
        if not candidates:
            candidates = list(results_dir.glob(f"results_{patient_folder}.nii"))
        if not candidates:
            candidates = list(results_dir.glob(f"results_{case_id}*.nii"))
    else:
        candidates = list(results_dir.glob(f"{cfg.result_prefix}{case_id}{cfg.result_suffix}"))

    if not candidates:
        return {}

    try:
        with open(candidates[0], "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Warning: Erreur pickle {candidates[0].name}: {e}")
        return {}

    derivator_dict = data.get("derivator", data)
    out: Dict[str, Optional[np.ndarray]] = {}

    for op in OPERATORS:
        if op not in derivator_dict:
            out[op] = None
            continue
        op_data = derivator_dict[op]

        seg = getattr(op_data, "data_segmented", None)
        if seg is None and isinstance(op_data, dict):
            seg = op_data.get("data_segmented")
        if seg is None:
            seg = getattr(op_data, "segmented", None)

        out[op] = np.asarray(seg).astype(bool) if seg is not None else None

    return out


def compute_operator_diversity_on_vrais_positifs(
    dataset: str,
    case_id: str,
    segs: Dict,
    gt: np.ndarray
) -> Tuple[float, Dict[str, int], Dict[str, np.ndarray]]:
    skel_gt = compute_skeleton(gt)

    skeletons = {}
    components = {}

    for op, mask in segs.items():
        if mask is not None:
            vrais_positifs = np.logical_and(mask, gt)
            skel_vp = compute_skeleton(vrais_positifs)
            skeletons[op] = skel_vp
            components[op] = count_components(vrais_positifs)

    if len(skeletons) < 2:
        return 0.0, components, skeletons

    total_diff = 0.0
    n_pairs = 0
    ops = list(skeletons.keys())

    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            skel1 = skeletons[ops[i]]
            skel2 = skeletons[ops[j]]
            if skel1 is None or skel2 is None or skel1.size == 0 or skel2.size == 0:
                continue
            diff = np.logical_xor(skel1, skel2).sum()
            total_gt_skel = skel_gt.sum() + 1e-8
            diff_ratio = diff / total_gt_skel
            total_diff += diff_ratio
            n_pairs += 1

    return total_diff / n_pairs if n_pairs > 0 else 0.0, components, skeletons


def pick_case_max_diversity_vp(
    dataset: str,
    df_metrics: pd.DataFrame
) -> Optional[Tuple[str, Dict[str, int], Dict[str, np.ndarray]]]:
    sub = df_metrics[df_metrics["dataset"] == dataset]
    if sub.empty:
        return None

    complete_cases = sub["case_id"].unique().tolist()

    best_case = None
    best_diversity = -1.0
    best_components = {}
    best_skeletons = {}

    for case_id in complete_cases:
        print(f"Test case {case_id}...")

        clean_case_id = case_id
        if dataset == "vascusynth"and "_rician"in case_id:
            clean_case_id = case_id.split("_rician")[0]

        gt = load_ground_truth(dataset, clean_case_id)
        if gt is None:
            print(f"GT non trouvée pour {clean_case_id}")
            continue

        segs = load_segmentation_for_case(dataset, case_id)
        if not segs or len(segs) < 5:
            print(f"Segmentations insuffisantes")
            continue

        diversity, components, skeletons = compute_operator_diversity_on_vrais_positifs(
            dataset, case_id, segs, gt
        )
        print(f"Diversity: {diversity:.4f}")

        if diversity > best_diversity:
            best_diversity = diversity
            best_case = case_id
            best_components = components
            best_skeletons = skeletons

    return (best_case, best_components, best_skeletons) if best_case else None


# CONSENSUS : STACK + STATS 4 CATÉGORIES

def compute_consensus_stack(skeletons: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], List[str]]:
    ops = [op for op, s in skeletons.items() if s is not None and s.size > 1]
    if not ops:
        return None, []

    shape = skeletons[ops[0]].shape
    count = np.zeros(shape, dtype=np.int16)
    for op in ops:
        s = skeletons[op]
        if s.shape != shape:
            continue
        count += s.astype(np.int16)

    return count, ops


def categorize_counts(counts: np.ndarray, n_ops: int) -> np.ndarray:

    half = n_ops / 2.0
    cat = np.full(counts.shape, BACKGROUND_CAT, dtype=np.int8)
    cat[counts == 1] = 0 # unique
    cat[(counts > 1) & (counts <= half)] = 1 # minoritaire
    cat[(counts > half) & (counts < n_ops)] = 2 # majoritaire
    cat[counts == n_ops] = 3 # consensus total
    return cat


def operator_consensus_stats(skel_op: np.ndarray, consensus_count: np.ndarray, n_ops: int) -> Dict[str, float]:
    if skel_op is None or skel_op.sum() == 0 or n_ops < 2:
        return {"unique": 0.0, "minoritaire": 0.0, "majoritaire": 0.0, "consensus": 0.0,
                "mean_agreement_pct": 0.0}

    counts_at_op = consensus_count[skel_op]
    total_voxels = counts_at_op.size
    half = n_ops / 2.0

    pct_unique = 100.0 * np.sum(counts_at_op == 1) / total_voxels
    pct_minor = 100.0 * np.sum((counts_at_op > 1) & (counts_at_op <= half)) / total_voxels
    pct_major = 100.0 * np.sum((counts_at_op > half) & (counts_at_op < n_ops)) / total_voxels
    pct_consensus = 100.0 * np.sum(counts_at_op == n_ops) / total_voxels

    mean_count = counts_at_op.mean()
    mean_agreement_pct = 100.0 * (mean_count - 1) / max(1, (n_ops - 1))

    return {
        "unique": pct_unique,
        "minoritaire": pct_minor,
        "majoritaire": pct_major,
        "consensus": pct_consensus,
        "mean_agreement_pct": mean_agreement_pct,
    }


def maximum_intensity_projection_3d(volume: np.ndarray, axis: int = 2) -> Optional[np.ndarray]:
    if volume is None or volume.size == 0:
        return None
    return np.max(volume, axis=axis)


def get_crop_bbox(mask: np.ndarray, margin: int = CROP_MARGIN) -> Tuple[slice, slice]:

    if mask is None or mask.sum() == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin + 1)
    return slice(y0, y1), slice(x0, x1)


# FIGURE : CONSENSUS CATÉGORIEL + RECADRAGE

def plot_operator_consensus_mip(
    dataset: str,
    df_metrics: pd.DataFrame,
) -> Tuple[Optional[str], Optional[Dict]]:
    print(f"\n[ {DATASET_LABELS[dataset]} ] recherche du cas avec le plus de differences...")

    result = pick_case_max_diversity_vp(dataset, df_metrics)
    if result is None:
        print(f"Aucun cas exploitable pour {dataset}")
        return None, None

    case_id, components, skeletons = result
    print(f"Cas selectionne : {case_id}")

    gt = load_ground_truth(dataset, case_id)
    if gt is None:
        print(f"GT introuvable pour {case_id}")
        return None, None

    segs = load_segmentation_for_case(dataset, case_id)
    if not segs:
        print(f"Segmentations introuvables pour {case_id}")
        return None, None

    consensus_count, ops_available = compute_consensus_stack(skeletons)
    if consensus_count is None or len(ops_available) < 2:
        print(f"Pas assez d'operateurs disponibles pour un consensus")
        return case_id, components

    n_ops = len(ops_available)
    ops_to_show = sorted(ops_available, key=lambda op: components.get(op, 999), reverse=True)

    n_panels = len(ops_to_show) + 1
    n_cols = 3
    n_rows = max(1, int(np.ceil(n_panels / n_cols)))

    # --- Projections + recadrage commun à toutes les figures du cas ---
    gt_proj = maximum_intensity_projection_3d(gt, axis=2)
    count_proj = maximum_intensity_projection_3d(consensus_count, axis=2)
    if count_proj is None:
        print(f"Projection consensus échouée")
        return case_id, components

    crop_y, crop_x = get_crop_bbox(gt_proj > 0)
    gt_proj = gt_proj[crop_y, crop_x]
    count_proj = count_proj[crop_y, crop_x]
    gt_proj_norm = gt_proj / (gt_proj.max() + 1e-8)

    # Masque des voxels réellement détectés par au moins un opérateur.
    # Sert à empêcher la dilatation de "peindre"du fond pur.
    real_detection_mask = count_proj > 0

    # Colormap catégorielle discrète (4 classes utiles + 1 case fond jamais affichée)
    cat_proj = categorize_counts(count_proj, n_ops)

    cat_cmap = ListedColormap([CATEGORY_COLORS["unique"], CATEGORY_COLORS["minoritaire"],
                                CATEGORY_COLORS["majoritaire"], CATEGORY_COLORS["consensus"]])
    cat_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cat_cmap.N)

    cat_proj_for_color = np.clip(cat_proj, 0, 3)
    color_proj = cat_cmap(cat_norm(cat_proj_for_color))[:, :, :3]

    def gray_background():
        bg = np.zeros((gt_proj.shape[0], gt_proj.shape[1], 3))
        bg[:, :, 0] = 0.25 * gt_proj_norm
        bg[:, :, 1] = 0.25 * gt_proj_norm
        bg[:, :, 2] = 0.25 * gt_proj_norm
        return bg

    dil_struct = np.ones((1 + 2 * DILATION_SIZE, 1 + 2 * DILATION_SIZE))

    def dilate_2d(m):
        if m is None or m.sum() == 0:
            return np.zeros_like(gt_proj, dtype=bool)
        return ndi_dilation(m, dil_struct)

    def paintable_mask(skel_mask_2d: np.ndarray) -> np.ndarray:

        dilated = dilate_2d(skel_mask_2d)
        return dilated & real_detection_mask

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.6 * n_rows + 1.2), dpi=DPI)
    if n_rows == 1:
        axes = np.atleast_2d(axes)

    fig.suptitle(
        f"{DATASET_LABELS[dataset]} - Cas {case_id} - Vrais positifs uniquement\n"
        f"Consensus de détection entre les {n_ops} opérateurs (projection maximale 3D, recadrée)-Filtre: MFAT",
        fontsize=12, fontweight="bold"
    )

    # --- Panneau 0 : consensus global ---
    ax0 = axes[0, 0]
    display0 = gray_background()
    union_mask = paintable_mask(real_detection_mask)
    display0[union_mask] = color_proj[union_mask]
    ax0.imshow(np.clip(display0, 0, 1), origin="lower")
    ax0.set_title(f"CONSENSUS GLOBAL\n({n_ops} opérateurs)", fontsize=9, fontweight="bold")
    ax0.axis("off")

    # --- Panneaux opérateurs ---
    for idx, op in enumerate(ops_to_show, start=1):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        skel_op = skeletons.get(op)
        if skel_op is None or skel_op.sum() == 0:
            ax.text(0.5, 0.5, f"{op.upper()}\nSquelette vide", ha="center", va="center",
                     transform=ax.transAxes, fontsize=10)
            ax.axis("off")
            continue

        stats = operator_consensus_stats(skel_op, consensus_count, n_ops)

        skel_proj_full = maximum_intensity_projection_3d(skel_op, axis=2)
        skel_proj = skel_proj_full[crop_y, crop_x] if skel_proj_full is not None else None
        if skel_proj is None:
            ax.text(0.5, 0.5, f"{op.upper()}\nProjection échouée", ha="center", va="center",
                     transform=ax.transAxes, fontsize=10)
            ax.axis("off")
            continue

        mask_op = paintable_mask(skel_proj)
        display = gray_background()
        display[mask_op] = color_proj[mask_op]
        ax.imshow(np.clip(display, 0, 1), origin="lower")

        n_components = components.get(op, 0)

        # Couleur du titre basée sur unique + minoritaire (désaccord réel)
        disagreement = stats["unique"] + stats["minoritaire"]
        title_color = "red"if disagreement > 40 else ("darkorange"if disagreement > 20 else "darkgreen")

        ax.set_title(
            f"{op.upper()} (comp={n_components})\n"
            f"Unique={stats['unique']:.0f}% | Minor.={stats['minoritaire']:.0f}% | "
            f"Major.={stats['majoritaire']:.0f}% | Consensus={stats['consensus']:.0f}%",
            fontsize=7.5, fontweight="bold", color=title_color
        )
        ax.axis("off")

    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")

    # Légende catégorielle (remplace la colorbar continue peu lisible)
    legend_patches = [Patch(facecolor=CATEGORY_COLORS[k], label=CATEGORY_LABELS[k])
                       for k in ["unique", "minoritaire", "majoritaire", "consensus"]]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    out_path = OUTPUT_DIR / f"consensus_mip_v2_{dataset}_{case_id}.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée: {out_path.name}")

    return case_id, components


# MAIN

def main() -> int:
    print("="* 80)
    print("VISUALISATION CONSENSUS - CATÉGORIELLE + RECADRÉE")
    print("="* 80)

    df_metrics = load_all_metrics(debug=False)
    if df_metrics.empty:
        print("Aucune metrique chargee.")
        return 1

    if "vascusynth"in df_metrics["dataset"].unique():
        init_vascusynth_config()

    for dataset in ["bullitt", "ircad", "vascusynth"]:
        if dataset not in df_metrics["dataset"].unique():
            continue
        plot_operator_consensus_mip(dataset, df_metrics)

    print("\nTermine. Fichiers dans :", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())