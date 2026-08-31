from __future__ import annotations

import itertools
import pickle
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr, friedmanchisquare, wilcoxon
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config.builder import ConfigBuilder
from core.config.postbench_tests import PostBenchTestsDatasetConfig
# Import the specific functions you need from the metrics module
from core.experiments.metrics import (
    dice,
    _skeletonize_3d,
    detect_bifurcations,
    connected_components_metrics,
    bifurcation_detection_rate,
    detailed_metrics
)
BENCHMARK_ROOT = PROJECT_ROOT / "outputs"

OPERATORS = ['default', 'gaussian', 'farid', 'cubic', 'trigonometric',
             'catmull', 'bspline', 'bezier', 'scharr']

SMALL_COMPONENT_THRESHOLD = 50

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


# CHARGEMENT patients, métriques texte, config, segmentations, GT

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


def _to_runtime_config(raw: PostBenchTestsDatasetConfig) -> DatasetConfig:
    """Convertit un PostBenchTestsDatasetConfig (chemins str, tel que chargé
    depuis un YAML) en DatasetConfig (chemins Path, résolus depuis
    PROJECT_ROOT) même convention que le dict DATASETS codé en dur
    ci-dessus (BENCHMARK_ROOT / PROJECT_ROOT)."""
    return DatasetConfig(
        enhancer_dir=PROJECT_ROOT / raw.enhancer_dir,
        labels_dir=PROJECT_ROOT / raw.labels_dir,
        masks_dir=(PROJECT_ROOT / raw.masks_dir) if raw.masks_dir else None,
        case_prefix=raw.case_prefix,
        case_suffix=raw.case_suffix,
        result_prefix=raw.result_prefix,
        result_suffix=raw.result_suffix,
        patients=list(raw.patients),
        has_subdirs=raw.has_subdirs,
    )


def load_datasets(config_dir: Union[str, Path]) -> Dict[str, DatasetConfig]:
    config_dir = Path(config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"Dossier de configs post-tests introuvable : {config_dir}")

    datasets: Dict[str, DatasetConfig] = {}
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        dataset_name = yaml_file.stem
        raw_config: PostBenchTestsDatasetConfig = ConfigBuilder(yaml_file, PostBenchTestsDatasetConfig)
        datasets[dataset_name] = _to_runtime_config(raw_config)
    return datasets


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


def extract_response(derivator_dict: Optional[Dict[str, Any]], operator: str) -> Optional[np.ndarray]:
    if not derivator_dict:
        return None
    for key in (operator, 'default'):
        if key not in derivator_dict:
            continue
        op_data = derivator_dict[key]
        if hasattr(op_data, 'data_enhanced'):
            v = op_data.data_enhanced
            if v is not None:
                return np.asarray(v)
        elif isinstance(op_data, dict):
            v = op_data.get('data_enhanced') or op_data.get('data_segmented')
            if v is not None:
                return np.asarray(v)
    return None


def load_ground_truth(dataset: str, case_id: str) -> Optional[np.ndarray]:
    """Utilisr par tous les tests, pas seulement le sweep (a)."""
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


def load_mask(dataset: str, case_id: str) -> Optional[np.ndarray]:
    import nibabel as nib
    cfg = DATASETS[dataset]
    if cfg.masks_dir is None or not cfg.masks_dir.exists() or dataset == "vascusynth":
        return None
    candidates = list(cfg.masks_dir.glob(f"*{case_id}*.nii*"))
    if not candidates:
        return None
    try:
        img = nib.load(str(candidates[0]))
        return np.asarray(img.get_fdata()) > 0.5
    except Exception:
        return None


# MÉTRIQUES vs GT (recalculées quand text_metrics.txt 
# ne suffit pas)

def _components_ratio_vs_gt(pred: np.ndarray, gt: np.ndarray) -> float:
    _, n_pred = ndimage.label(pred, structure=np.ones((3, 3, 3)))
    _, n_gt = ndimage.label(gt, structure=np.ones((3, 3, 3)))
    return n_pred / n_gt if n_gt > 0 else np.nan


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


# TEST 1 DISTRIBUTION DES TAILLES DE COMPOSANTES  (déjà vs GT
# via pred_small/medium/large_components qui viennent de text_metrics.txt)

def analyze_filtering_impact(df_metrics: pd.DataFrame) -> Dict[str, Any]:
    results = {}
    for dataset in df_metrics['dataset'].unique():
        sub = df_metrics[df_metrics['dataset'] == dataset]
        op_dist, ratios, small_pcts = {}, [], []
        for op in OPERATORS:
            op_data = sub[sub['operator'] == op]
            if op_data.empty:
                continue
            small = op_data['pred_small_components'].mean()
            medium = op_data['pred_medium_components'].mean()
            large = op_data['pred_large_components'].mean()
            total = small + medium + large
            op_dist[op] = {
                'small_pct': small / total * 100 if total > 0 else 0,
                'medium_pct': medium / total * 100 if total > 0 else 0,
                'large_pct': large / total * 100 if total > 0 else 0,
            }
            ratios.append(op_data['components_ratio'].mean())
            small_pcts.append(op_dist[op]['small_pct'])

        corr = None
        if len(small_pcts) > 2 and len(ratios) > 2:
            try:
                corr, _ = pearsonr(small_pcts, ratios)
            except Exception:
                corr = None

        results[dataset] = {
            'operator_distribution': op_dist,
            'components_ratio_range': max(ratios) - min(ratios) if ratios else 0,
            'correlation_small_vs_ratio': corr,
        }
    return results


# TEST 1bis FILTRAGE < 50 VOXELS
#dice/cldice recalculés sur seg_filtered vs gt.

def remove_small_components_3d(mask: np.ndarray, min_size: int = SMALL_COMPONENT_THRESHOLD) -> np.ndarray:
    if mask is None or mask.sum() == 0:
        return mask
    labeled, _ = ndimage.label(mask)
    sizes = np.bincount(labeled.ravel())
    keep = np.isin(labeled, np.where(sizes >= min_size)[0][1:] if len(sizes) > 1 else [])
    return keep


def compute_components_after_filter(seg: np.ndarray, gt: Optional[np.ndarray],
                                     min_size: int = SMALL_COMPONENT_THRESHOLD) -> Dict[str, Any]:
    """
    Calcule les métriques de composantes avant/après filtrage.
    Utilise connected_components_metrics du module metrics.
    """
    if seg is None or seg.sum() == 0:
        return {
            'n_components_raw': 0, 'n_components_filtered': 0, 'n_removed': 0,
            'removed_ratio': 0.0, 'volume_raw': 0, 'volume_filtered': 0,
            'dice_raw': np.nan, 'dice_filtered': np.nan,
            'cldice_raw': np.nan, 'cldice_filtered': np.nan,
        }

    # Filtrage
    seg_filtered = remove_small_components_3d(seg, min_size)

    conn_raw = connected_components_metrics(seg, gt) if gt is not None else connected_components_metrics(seg, seg)
    conn_filtered = connected_components_metrics(seg_filtered, gt) if gt is not None else connected_components_metrics(seg_filtered, seg_filtered)

    result = {
        'n_components_raw': conn_raw.get('n_components_pred', 0),
        'n_components_filtered': conn_filtered.get('n_components_pred', 0),
        'n_removed': conn_raw.get('n_components_pred', 0) - conn_filtered.get('n_components_pred', 0),
        'removed_ratio': (conn_raw.get('n_components_pred', 0) - conn_filtered.get('n_components_pred', 0)) / max(conn_raw.get('n_components_pred', 1), 1),
        'volume_raw': int(seg.sum()),
        'volume_filtered': int(seg_filtered.sum()),
    }

    # Calcul des métriques de qualité avec detailed_metrics
    if gt is not None and gt.shape == seg.shape:
        metrics_raw = detailed_metrics(seg, gt, metrics=["dice", "cldice"])
        metrics_filtered = detailed_metrics(seg_filtered, gt, metrics=["dice", "cldice"])

        result['dice_raw'] = metrics_raw['dice']
        result['dice_filtered'] = metrics_filtered['dice']
        result['cldice_raw'] = metrics_raw['cldice']
        result['cldice_filtered'] = metrics_filtered['cldice']
    else:
        result['dice_raw'] = result['dice_filtered'] = np.nan
        result['cldice_raw'] = result['cldice_filtered'] = np.nan

    return result


def analyze_filtering_impact_small_components(df_metrics: pd.DataFrame) -> Dict[str, Any]:
    results = {}
    for dataset in df_metrics['dataset'].unique():
        sub = df_metrics[df_metrics['dataset'] == dataset]
        dataset_results = []

        # [INSTRUMENTATION] boucle potentiellement lente (detailed_metrics
        # est appelé 2x par ligne, dont clDice qui squelettise) : tqdm rend
        # visible la progression et le case_id/operator courant au lieu de
        # rester silencieux. Aucun changement de calcul.
        rows = list(sub.iterrows())
        pbar = tqdm(rows, desc=f"Test 1bis {dataset}", total=len(rows))
        for _, row in pbar:
            pbar.set_postfix_str(f"case={row['case_id']} op={row['operator']}")
            seg = row.get('segmentation')
            if seg is None:
                continue
            gt = row.get('gt')
            comp_stats = compute_components_after_filter(seg, gt, SMALL_COMPONENT_THRESHOLD)
            dataset_results.append({
                'case_id': row['case_id'],
                'operator': row['operator'],
                **comp_stats,
            })

        if dataset_results:
            df_results = pd.DataFrame(dataset_results)
            # Maintenant dice_raw, dice_filtered, cldice_raw, cldice_filtered existent
            df_results['dice_delta'] = df_results['dice_filtered'] - df_results['dice_raw']
            df_results['cldice_delta'] = df_results['cldice_filtered'] - df_results['cldice_raw']

            agg = df_results.groupby('operator').agg({
                'n_components_raw': 'mean', 'n_components_filtered': 'mean',
                'removed_ratio': 'mean',
                'dice_raw': 'mean', 'dice_filtered': 'mean', 'dice_delta': 'mean',
                'cldice_raw': 'mean', 'cldice_filtered': 'mean', 'cldice_delta': 'mean',
            }).round(4)
            agg['components_reduction_pct'] = 100.0 * (
                agg['n_components_raw'] - agg['n_components_filtered']
            ) / agg['n_components_raw']

            n_with_gt = df_results['dice_raw'].notna().sum()
            results[dataset] = {
                'df': df_results, 'agg': agg,
                'n_with_gt': int(n_with_gt),
                'summary': {
                    'mean_dice_delta': df_results['dice_delta'].mean(),
                    'mean_cldice_delta': df_results['cldice_delta'].mean(),
                    'pct_cases_improved_dice': (df_results['dice_delta'] > 0).mean() * 100
                    if n_with_gt > 0 else np.nan,
                }
            }
        else:
            results[dataset] = None
    return results


def report_filtering_impact_small_components(results: Dict[str, Any]) -> str:
    lines = ["-"* 80,
             f"TEST 1bis - FILTRAGE DES PETITES COMPOSANTES (< {SMALL_COMPONENT_THRESHOLD} voxels)",
             "-"* 80, ""]
    for dataset, result in results.items():
        if result is None:
            lines.append(f"\n{dataset.upper()}: Aucune donnée")
            continue
        agg = result['agg']
        lines.append(f"\n{dataset.upper()} (GT disponible pour {result['n_with_gt']} lignes)")
        lines.append(f"Réduction moyenne du nombre de composantes : "
                      f"{agg['components_reduction_pct'].mean():.1f}%")
        lines.append(f"Delta Dice moyen (filtré - brut, vs GT) : {result['summary']['mean_dice_delta']:+.4f}")
        lines.append(f"Delta clDice moyen (filtré - brut, vs GT) : {result['summary']['mean_cldice_delta']:+.4f}")
        if not np.isnan(result['summary']['pct_cases_improved_dice']):
            lines.append(f"% de cas où le filtrage améliore le Dice : "
                          f"{result['summary']['pct_cases_improved_dice']:.1f}%")
        lines.append("")
        lines.append(f"{'Opérateur':<16} {'Réduc.%':>9} {'Dice brut':>10} {'Dice filt.':>11} {'Delta Dice':>10}")
        lines.append(""+ "-"* 62)
        for op in OPERATORS:
            if op not in agg.index:
                continue
            r = agg.loc[op]
            lines.append(f"{op:<16} {r['components_reduction_pct']:>8.1f}% "
                          f"{r['dice_raw']:>10.4f} {r['dice_filtered']:>11.4f} {r['dice_delta']:>+10.4f}")
        if result['summary']['mean_dice_delta'] < -0.005:
            lines.append("\n Le filtrage DÉGRADE le Dice en moyenne : les petites composantes"
                          "retirées contenaient en partie du vrai vaisseau (faux négatifs induits).")
        elif result['summary']['mean_dice_delta'] > 0.005:
            lines.append("\n Le filtrage AMÉLIORE le Dice en moyenne : les petites composantes"
                          "retirées étaient majoritairement du bruit (faux positifs).")
        else:
            lines.append("\n Effet du filtrage sur le Dice négligeable (< 0.005) en moyenne.")
    return "\n".join(lines)


# TEST 2 PROXIMITÉ AU SEUIL 
# on garde dice1/dice2 (distance individuelle au GT) et on
# regarde si le ratio de seuil corrèle avec la distance moyenne au GT et/ou
# avec le fait que l'un des deux soit nettement pire.

def analyze_threshold_proximity(df_metrics: pd.DataFrame) -> Dict[str, Any]:
    comparisons = [('gaussian', 'bspline'), ('gaussian', 'bezier'), ('gaussian', 'farid'),
                   ('gaussian', 'scharr'), ('bspline', 'bezier')]
    results = {}

    for dataset in df_metrics['dataset'].unique():
        sub = df_metrics[df_metrics['dataset'] == dataset]
        dataset_results = []

        for op1, op2 in comparisons:
            data1 = sub[sub['operator'] == op1]
            data2 = sub[sub['operator'] == op2]
            if data1.empty or data2.empty:
                continue
            merged = pd.merge(
                data1[['case_id', 'threshold', 'dice', 'components_ratio']],
                data2[['case_id', 'threshold', 'dice', 'components_ratio']],
                on='case_id', suffixes=('_1', '_2')
            )
            for _, row in merged.iterrows():
                t1, t2 = row['threshold_1'], row['threshold_2']
                if pd.isna(t1) or pd.isna(t2) or t1 <= 0 or t2 <= 0:
                    continue
                threshold_ratio = max(t1, t2) / min(t1, t2)
                dataset_results.append({
                    'case_id': row['case_id'], 'op1': op1, 'op2': op2,
                    'threshold_ratio': threshold_ratio,
                    'dice_1': row['dice_1'], 'dice_2': row['dice_2'],
                    'mean_gt_error': 1 - (row['dice_1'] + row['dice_2']) / 2,
                    'worse_dice': min(row['dice_1'], row['dice_2']),
                    'dice_diff_between_ops': abs(row['dice_1'] - row['dice_2']),
                })

        if dataset_results:
            results[dataset] = pd.DataFrame(dataset_results)
        else:
            results[dataset] = pd.DataFrame()
    return results


def report_threshold_proximity(results: Dict[str, pd.DataFrame]) -> str:
    lines = ["-"* 80, "TEST 2 - PROXIMITÉ AU SEUIL ", "-"* 80, ""]
    for dataset, df in results.items():
        lines.append(f"\n{dataset.upper()}")
        if df.empty:
            lines.append("Aucune donnée")
            continue
        lines.append(f"Ratio seuil moyen : {df['threshold_ratio'].mean():.2f} ± {df['threshold_ratio'].std():.2f}")
        if len(df) > 2:
            r_gt, p_gt = pearsonr(df['threshold_ratio'], df['mean_gt_error'])
            r_worse, p_worse = pearsonr(df['threshold_ratio'], 1 - df['worse_dice'])
            r_between, p_between = pearsonr(df['threshold_ratio'], df['dice_diff_between_ops'])
            lines.append(f"Corrélation ratio seuil / erreur moyenne vs GT : r={r_gt:.3f}, p={p_gt:.4f}")
            lines.append(f"Corrélation ratio seuil / erreur du pire des deux : r={r_worse:.3f}, p={p_worse:.4f}")
            lines.append(f"(indicatif, ne mesure pas la distance au GT) "
                          f"ratio seuil / écart entre opérateurs : r={r_between:.3f}, p={p_between:.4f}")
            if p_gt < 0.05 and r_gt > 0.2:
                lines.append("-> Un écart de seuil important est associé à une plus grande erreur vs GT.")
            else:
                lines.append("-> Pas de lien clair entre écart de seuil et erreur réelle vs GT.")
    return "\n".join(lines)


# TEST 3 / MÉDIATION 
# Y = excès de fragmentation réel de op2 par rapport à op1,
# mesuré vs GT via fragmentation_ratio (déjà une métrique vs GT dans
# text_metrics.txt), signé pour donner un sens causal à X->Y.

def analyze_mediation(df_metrics: pd.DataFrame) -> Dict[str, Any]:
    comparisons = [('gaussian', 'bspline'), ('gaussian', 'bezier'), ('gaussian', 'farid')]
    results = {}
    for dataset in df_metrics['dataset'].unique():
        sub = df_metrics[df_metrics['dataset'] == dataset]
        mediation_data = []
        for op1, op2 in comparisons:
            data1 = sub[sub['operator'] == op1]
            data2 = sub[sub['operator'] == op2]
            if data1.empty or data2.empty:
                continue
            for _, row1 in data1.iterrows():
                row2_match = data2[data2['case_id'] == row1['case_id']]
                if row2_match.empty:
                    continue
                row2 = row2_match.iloc[0]
                t1, t2 = row1.get('threshold'), row2.get('threshold')
                if t1 is None or t2 is None:
                    continue
                try:
                    t1, t2 = float(t1), float(t2)
                except (ValueError, TypeError):
                    continue
                if pd.isna(t1) or pd.isna(t2) or t1 <= 0 or t2 <= 0:
                    continue

                X = max(t1, t2) / min(t1, t2) - 1
                M = row2['dice'] - row1['dice'] 
                frag1 = row1.get('fragmentation_ratio')
                frag2 = row2.get('fragmentation_ratio')
                if frag1 is None or frag2 is None or pd.isna(frag1) or pd.isna(frag2):
                    continue
                Y = frag2 - frag1 

                mediation_data.append({'case_id': row1['case_id'], 'op1': op1, 'op2': op2,
                                        'X': X, 'M': M, 'Y': Y})

        if mediation_data:
            df_med = pd.DataFrame(mediation_data)
            r_xy, p_xy = pearsonr(df_med['X'], df_med['Y'])
            r_xm, p_xm = pearsonr(df_med['X'], df_med['M'])
            r_my, p_my = pearsonr(df_med['M'], df_med['Y'])
            denom = np.sqrt((1 - r_xm ** 2) * (1 - r_my ** 2))
            r_xy_m = (r_xy - r_xm * r_my) / denom if denom > 0 else float("nan")
            results[dataset] = {
                'n': len(df_med), 'r_total': r_xy, 'p_total': p_xy,
                'r_xm': r_xm, 'r_my': r_my, 'r_direct': r_xy_m, 'data': df_med,
                'interpretation': (
                    "médiation complète (compatible avec causalité)"
                    if abs(r_xy_m) < 0.1 and abs(r_xy) > 0.3
                    else "médiation partielle (compatible avec causalité)"
                    if abs(r_xy_m) < abs(r_xy) * 0.5 and abs(r_xy) > 0.3
                    else "pas de médiation détectée"
                )
            }
        else:
            results[dataset] = {'n': 0, 'interpretation': 'données insuffisantes'}
    return results


# TEST 4 - DISTRIBUTION DES SEUILS

def analyze_threshold_distribution(df_metrics: pd.DataFrame) -> Dict[str, Any]:
    results = {}
    for dataset in df_metrics['dataset'].unique():
        sub = df_metrics[df_metrics['dataset'] == dataset]
        thresholds = {}
        for op in OPERATORS:
            op_data = sub[sub['operator'] == op]
            if op_data.empty:
                continue
            thr = op_data['threshold'].dropna()
            if not thr.empty:
                thresholds[op] = {'mean': thr.mean(), 'std': thr.std()}
        means = [t['mean'] for t in thresholds.values() if not np.isnan(t['mean'])]
        results[dataset] = {
            'thresholds': thresholds,
            'threshold_range': max(means) - min(means) if means else 0,
        }
    return results


# TEST 5 - VOXELS CRITIQUES
# squelette/bifurcations calculés sur le GT ; on mesure si op2
# réduit ou augmente l'erreur (vs GT) sur ces voxels topologiquement
# importants, par rapport à op1.
# Le squelette et les points de bifurcation du GT ne dépendent que
# de case_id, pas de la paire (op1, op2). Auparavant ils étaient recalculés
# à chaque comparaison (6 paires x tous les cas) alors que le même case_id
# revient dans plusieurs paires -> mise en cache par case_id.
# Idem pour le squelette d'une SEGMENTATION prédite :
# _skeletonize_3d(seg1) / _skeletonize_3d(seg2) sont des fonctions pures
# (même entrée -> même sortie). Or un même (case_id, operator) revient
# comme seg1 et/ou seg2 dans plusieurs des 6 paires de comparaison.

def analyze_critical_changes(df_metrics: pd.DataFrame,
                              comparisons: List[Tuple[str, str]] = None) -> Dict[str, Any]:
    """
    Analyse les changements sur les voxels critiques (squelette + bifurcations).
    Utilise bifurcation_detection_rate pour des métriques plus précises.
    """
    if comparisons is None:
        comparisons = [('gaussian', 'bspline'), ('gaussian', 'bezier'), ('gaussian', 'farid'),
                       ('gaussian', 'scharr'), ('bspline', 'bezier'), ('farid', 'bezier')]

    results = {}
    for dataset in df_metrics['dataset'].unique():
        sub = df_metrics[df_metrics['dataset'] == dataset]
        critical_data = []
        critical_cache: Dict[Any, Tuple[np.ndarray, int, np.ndarray]] = {}
        seg_skel_cache: Dict[Tuple[Any, str], np.ndarray] = {}

        def _get_seg_skeleton(case_id: Any, operator: str, seg: np.ndarray) -> np.ndarray:
            key = (case_id, operator)
            if key not in seg_skel_cache:
                seg_skel_cache[key] = _skeletonize_3d(seg)
            return seg_skel_cache[key]

        for op1, op2 in tqdm(comparisons, desc=f"Test 5 {dataset} (paires)"):
            data1 = sub[sub['operator'] == op1]
            data2 = sub[sub['operator'] == op2]
            if data1.empty or data2.empty:
                continue

            rows1 = list(data1.iterrows())
            pbar = tqdm(rows1, desc=f"  {op1} vs {op2}", leave=False)
            for _, row1 in pbar:
                pbar.set_postfix_str(f"case={row1['case_id']}")
                row2_match = data2[data2['case_id'] == row1['case_id']]
                if row2_match.empty:
                    continue
                row2 = row2_match.iloc[0]
                seg1, seg2, gt = row1.get('segmentation'), row2.get('segmentation'), row1.get('gt')
                if seg1 is None or seg2 is None or gt is None or gt.shape != seg1.shape:
                    continue

                case_key = row1['case_id']
                if case_key not in critical_cache:
                    skel_gt = _skeletonize_3d(gt)
                    bp_gt = detect_bifurcations(skel_gt, threshold=3)
                    critical = skel_gt | bp_gt
                    critical_cache[case_key] = (critical, int(critical.sum()), skel_gt)
                critical, n_critical, skel_gt = critical_cache[case_key]
                if n_critical == 0:
                    continue

                # Calcul des erreurs sur voxels critiques
                error1 = np.logical_xor(seg1, gt) & critical
                error2 = np.logical_xor(seg2, gt) & critical
                error1_ratio = error1.sum() / n_critical
                error2_ratio = error2.sum() / n_critical
                delta_error = error2_ratio - error1_ratio

                # Utilisation de bifurcation_detection_rate pour des métriques précises
                # on utilise les squelettes déjà calculés via le cache
                bif_metrics1 = bifurcation_detection_rate(
                    seg1, gt,
                    _s_pred=_get_seg_skeleton(case_key, op1, seg1),
                    _s_gt=skel_gt,
                    tolerance_radius=3,
                    bifurcation_threshold=3
                )
                bif_metrics2 = bifurcation_detection_rate(
                    seg2, gt,
                    _s_pred=_get_seg_skeleton(case_key, op2, seg2),
                    _s_gt=skel_gt,
                    tolerance_radius=3,
                    bifurcation_threshold=3
                )

                delta_dice = row2['dice'] - row1['dice']
                delta_frag = (row2.get('fragmentation_ratio') or np.nan) - (row1.get('fragmentation_ratio') or np.nan)
                delta_bdr = bif_metrics2['bifurcation_detection_rate'] - bif_metrics1['bifurcation_detection_rate']

                critical_data.append({
                    'case_id': row1['case_id'], 'op1': op1, 'op2': op2,
                    'n_critical': int(n_critical),
                    'error1_ratio': error1_ratio, 'error2_ratio': error2_ratio,
                    'delta_error_critical': delta_error,
                    'delta_dice': delta_dice,
                    'delta_fragmentation': delta_frag,
                    'bdr_1': bif_metrics1['bifurcation_detection_rate'],
                    'bdr_2': bif_metrics2['bifurcation_detection_rate'],
                    'delta_bdr': delta_bdr,
                })

        if critical_data:
            df_crit = pd.DataFrame(critical_data).dropna(subset=['delta_fragmentation'])
            if len(df_crit) > 2:
                r_err_dice, p_err_dice = pearsonr(df_crit['delta_error_critical'], df_crit['delta_dice'])
                r_err_frag, p_err_frag = pearsonr(df_crit['delta_error_critical'], df_crit['delta_fragmentation'])
                r_err_bdr, p_err_bdr = pearsonr(df_crit['delta_error_critical'], df_crit['delta_bdr'])
            else:
                r_err_dice = p_err_dice = r_err_frag = p_err_frag = r_err_bdr = p_err_bdr = np.nan

            results[dataset] = {
                'n_comparisons': len(df_crit), 'df': df_crit,
                'correlations': {
                    'delta_error_critical_vs_delta_dice': {'r': r_err_dice, 'p': p_err_dice},
                    'delta_error_critical_vs_delta_fragmentation': {'r': r_err_frag, 'p': p_err_frag},
                    'delta_error_critical_vs_delta_bdr': {'r': r_err_bdr, 'p': p_err_bdr},
                },
                'summary': {
                    'mean_error1_ratio': df_crit['error1_ratio'].mean(),
                    'mean_error2_ratio': df_crit['error2_ratio'].mean(),
                    'mean_bdr_1': df_crit['bdr_1'].mean(),
                    'mean_bdr_2': df_crit['bdr_2'].mean(),
                }
            }
        else:
            results[dataset] = {'n_comparisons': 0}
    return results


def report_critical_changes(results: Dict[str, Any]) -> str:
    lines = ["-"* 80, "TEST 5 - VOXELS CRITIQUES (squelette + bifurcations DU GT) [AVEC BDR]",
             "-"* 80,
             "(on mesure si un changement d'opérateur réduit ou augmente l'erreur vs GT",
             "sur les voxels topologiquement importants, incluant les métriques BDR)", ""]
    for dataset, r in results.items():
        if r.get('n_comparisons', 0) == 0:
            lines.append(f"\n{dataset.upper()}: Aucune donnée (GT manquant ou squelette GT vide)")
            continue
        lines.append(f"\n{dataset.upper()} ({r['n_comparisons']} comparaisons)")
        lines.append(f"Erreur moyenne sur voxels critiques - op1 : {r['summary']['mean_error1_ratio']*100:.2f}%")
        lines.append(f"Erreur moyenne sur voxels critiques - op2 : {r['summary']['mean_error2_ratio']*100:.2f}%")
        lines.append(f"BDR moyen - op1 : {r['summary'].get('mean_bdr_1', np.nan)*100:.2f}%")
        lines.append(f"BDR moyen - op2 : {r['summary'].get('mean_bdr_2', np.nan)*100:.2f}%")

        c1 = r['correlations']['delta_error_critical_vs_delta_dice']
        c2 = r['correlations']['delta_error_critical_vs_delta_fragmentation']
        c3 = r['correlations'].get('delta_error_critical_vs_delta_bdr', {'r': np.nan, 'p': np.nan})

        lines.append(f"Corr. delta erreur_critique / delta Dice : r={c1['r']:.3f}, p={c1['p']:.4f}")
        lines.append(f"Corr. delta erreur_critique / delta fragmentation : r={c2['r']:.3f}, p={c2['p']:.4f}")
        lines.append(f"Corr. delta erreur_critique / delta BDR : r={c3['r']:.3f}, p={c3['p']:.4f}")

        if c1['r'] < -0.2 and c1['p'] < 0.05:
            lines.append("-> Réduire l'erreur sur les voxels critiques améliore bien le Dice réel (cohérent).")
        if c2['r'] > 0.2 and c2['p'] < 0.05:
            lines.append("-> Réduire l'erreur sur les voxels critiques réduit la fragmentation réelle vs GT.")
        if c3['r'] < -0.2 and c3['p'] < 0.05:
            lines.append("-> Réduire l'erreur sur les voxels critiques améliore la détection des bifurcations.")
    return "\n".join(lines)


# (b) FRIEDMAN + POST-HOC HOLM ( dice/mcc/cldice/components_ratio/
# fragmentation_ratio sont déjà des métriques vs GT)

def analyze_operator_differences(df_metrics: pd.DataFrame,
                                  metrics: Tuple[str, ...] = ("dice", "mcc", "cldice",
                                                               "components_ratio", "fragmentation_ratio"),
                                  alpha: float = 0.05) -> Dict:
    results = {}
    for dataset in df_metrics["dataset"].unique():
        sub = df_metrics[df_metrics["dataset"] == dataset]
        results[dataset] = {}
        for metric in metrics:
            if metric not in sub.columns:
                continue
            wide = sub.pivot_table(index="case_id", columns="operator", values=metric).dropna()
            wide = wide[[op for op in OPERATORS if op in wide.columns]]
            if wide.shape[0] < 3 or wide.shape[1] < 3:
                results[dataset][metric] = {"error": "pas assez de données complètes"}
                continue

            stat, p = friedmanchisquare(*[wide[op].values for op in wide.columns])
            n, k = wide.shape
            kendalls_w = stat / (n * (k - 1))

            posthoc = {}
            if p < alpha:
                pairs = list(itertools.combinations(wide.columns, 2))
                raw_p = []
                for op1, op2 in pairs:
                    try:
                        _, pw = wilcoxon(wide[op1], wide[op2])
                    except ValueError:
                        pw = np.nan
                    raw_p.append(pw)
                order = np.argsort(raw_p)
                m = len(raw_p)
                holm_p = np.empty(m)
                for rank, idx in enumerate(order):
                    holm_p[idx] = min(raw_p[idx] * (m - rank), 1.0)
                running_max = 0.0
                for idx in order:
                    running_max = max(running_max, holm_p[idx])
                    holm_p[idx] = running_max
                for (op1, op2), pw, ph in zip(pairs, raw_p, holm_p):
                    posthoc[f"{op1}_vs_{op2}"] = {"p_raw": pw, "p_holm": ph,
                                                   "significant": ph < alpha,
                                                   "median_diff": wide[op1].median() - wide[op2].median()}

            results[dataset][metric] = {
                "n_patients_complets": n, "n_operateurs": k,
                "friedman_stat": stat, "friedman_p": p, "kendalls_w": kendalls_w,
                "operators_differ": p < alpha, "posthoc_pairwise": posthoc,
            }
    return results


def summarize_operator_differences(results: Dict) -> str:
    lines = []
    for dataset, metrics_res in results.items():
        lines.append(f"\n{dataset.upper()}")
        for metric, r in metrics_res.items():
            if "error"in r:
                lines.append(f"{metric}: {r['error']}")
                continue
            lines.append(f"{metric}: Friedman chi2={r['friedman_stat']:.2f}, p={r['friedman_p']:.4g}, "
                          f"Kendall's W={r['kendalls_w']:.3f} "
                          f"({'les opérateurs diffèrent'if r['operators_differ'] else 'pas de différence'}, "
                          f"n={r['n_patients_complets']})")
            sig_pairs = [k for k, v in r["posthoc_pairwise"].items() if v["significant"]]
            if sig_pairs:
                lines.append(f"Paires significatives (Holm, p<0.05): {len(sig_pairs)}")
                for pair in sig_pairs:
                    v = r["posthoc_pairwise"][pair]
                    lines.append(f"{pair}: p_holm={v['p_holm']:.4g}, diff médiane={v['median_diff']:.4f}")
    return "\n".join(lines)


# (c) BOOTSTRAP DE L'EFFET INDIRECT

def build_mediation_dataframe(df_metrics: pd.DataFrame,
                               comparisons: Tuple[Tuple[str, str], ...] = (
                                   ("gaussian", "bspline"), ("gaussian", "bezier"), ("gaussian", "farid"))
                               ) -> pd.DataFrame:
    rows = []
    for dataset in df_metrics["dataset"].unique():
        sub = df_metrics[df_metrics["dataset"] == dataset]
        for op1, op2 in comparisons:
            d1 = sub[sub["operator"] == op1]
            d2 = sub[sub["operator"] == op2]
            for _, r1 in d1.iterrows():
                r2m = d2[d2["case_id"] == r1["case_id"]]
                if r2m.empty:
                    continue
                r2 = r2m.iloc[0]
                t1, t2 = r1.get("threshold"), r2.get("threshold")
                if t1 is None or t2 is None:
                    continue
                try:
                    t1, t2 = float(t1), float(t2)
                except (TypeError, ValueError):
                    continue
                if t1 <= 0 or t2 <= 0:
                    continue
                frag1, frag2 = r1.get('fragmentation_ratio'), r2.get('fragmentation_ratio')
                if frag1 is None or frag2 is None or pd.isna(frag1) or pd.isna(frag2):
                    continue

                X = max(t1, t2) / min(t1, t2) - 1
                M = r2["dice"] - r1["dice"]
                Y = frag2 - frag1 
                rows.append({"dataset": dataset, "case_id": f"{dataset}_{r1['case_id']}",
                             "op1": op1, "op2": op2, "X": X, "M": M, "Y": Y})
    return pd.DataFrame(rows)


def bootstrap_mediation(df_med: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> Dict:
    import statsmodels.api as sm
    rng = np.random.default_rng(seed)

    def _fit_paths(data: pd.DataFrame) -> Tuple[float, float, float]:
        X = sm.add_constant(data["X"])
        a = sm.OLS(data["M"], X).fit().params["X"]
        XM = sm.add_constant(data[["X", "M"]])
        model_b = sm.OLS(data["Y"], XM).fit()
        return a, model_b.params["M"], model_b.params["X"]

    a_obs, b_obs, c_prime_obs = _fit_paths(df_med)
    indirect_obs = a_obs * b_obs
    case_ids = df_med["case_id"].unique()
    n_cases = len(case_ids)

    #  regroupe les lignes par case_id une seule fois avant le
    # bootstrap : auparavant chaque itération refiltrait df_med par égalité
    # sur case_id pour chaque élément tiré (coût O(n_boot * n_cases * len(df_med))).
    groups = {cid: g for cid, g in df_med.groupby("case_id")}

    boot_indirect = np.empty(n_boot)
    for i in tqdm(range(n_boot), desc="Bootstrap médiation", leave=False):
        sampled = rng.choice(case_ids, size=n_cases, replace=True)
        boot_df = pd.concat([groups[c] for c in sampled], ignore_index=True)
        try:
            a_b, b_b, _ = _fit_paths(boot_df)
            boot_indirect[i] = a_b * b_b
        except Exception:
            boot_indirect[i] = np.nan
    boot_indirect = boot_indirect[~np.isnan(boot_indirect)]
    ci_low, ci_high = np.percentile(boot_indirect, [2.5, 97.5])
    total_effect = indirect_obs + c_prime_obs
    prop_mediated = indirect_obs / total_effect if total_effect != 0 else np.nan

    return {
        "n_clusters": n_cases, "n_boot": len(boot_indirect),
        "a_path": a_obs, "b_path": b_obs, "direct_effect_c_prime": c_prime_obs,
        "indirect_effect_ab": indirect_obs, "indirect_ci_95": (ci_low, ci_high),
        "indirect_significant": not (ci_low <= 0 <= ci_high),
        "proportion_mediated": prop_mediated,
        "interpretation": (
            "médiation significative (l'IC 95% de l'effet indirect exclut 0)"
            if not (ci_low <= 0 <= ci_high)
            else "pas de médiation significative (l'IC 95% inclut 0)"
        ),
    }


# (a) SWEEP DE SEUIL A OPÉRATEUR FIXE

def sweep_threshold_fixed_operator(dataset: str, operator: str, case_id: str,
                                    optimal_threshold: float, n_steps: int = 15,
                                    rel_range: Tuple[float, float] = (0.5, 1.5),
                                    debug: bool = False) -> Optional[pd.DataFrame]:
    cfg = DATASETS[dataset]
    if dataset == "vascusynth":
        patient_folder = case_id.split("_rician")[0] if "_rician"in case_id else case_id
        patient_dir = cfg.enhancer_dir / patient_folder
    else:
        patient_dir = cfg.enhancer_dir / f"{cfg.case_prefix}{case_id}{cfg.case_suffix}"
    if not patient_dir.exists():
        return None

    derivator_dict = load_pickle_once(dataset, case_id, debug)
    response = extract_response(derivator_dict, operator)
    if response is None:
        return None
    gt = load_ground_truth(dataset, case_id)
    if gt is None:
        return None
    if response.shape != gt.shape:
        if response.ndim == gt.ndim:
            m = tuple(min(a, b) for a, b in zip(response.shape, gt.shape))
            response, gt = response[:m[0], :m[1], :m[2]], gt[:m[0], :m[1], :m[2]]
        else:
            return None

    roi = load_mask(dataset, case_id)
    thresholds = np.linspace(optimal_threshold * rel_range[0], optimal_threshold * rel_range[1], n_steps)
    rows = []
    for t in thresholds:
        pred = response > t
        if roi is not None:
            pred = np.logical_and(pred, roi)
        ratio = _components_ratio_vs_gt(pred, gt)
        dice_val = dice(pred, gt)
        rows.append({"dataset": dataset, "operator": operator, "case_id": case_id,
                      "threshold": t, "rel_distance_to_optimal": (t - optimal_threshold) / optimal_threshold,
                      "dice": dice_val, "components_ratio": ratio})
    return pd.DataFrame(rows)


def run_threshold_sweep_all(df_metrics: pd.DataFrame, operator: str = "gaussian",
                             n_cases_per_dataset: int = 10, n_steps: int = 15,
                             debug: bool = False) -> pd.DataFrame:
    all_sweeps = []
    for dataset in df_metrics["dataset"].unique():
        sub = df_metrics[(df_metrics["dataset"] == dataset) & (df_metrics["operator"] == operator)]
        sub = sub.dropna(subset=["threshold"]).head(n_cases_per_dataset) if n_cases_per_dataset else sub.dropna(subset=["threshold"])
        for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"sweep {dataset}/{operator}"):
            res = sweep_threshold_fixed_operator(dataset, operator, row["case_id"], row["threshold"],
                                                  n_steps=n_steps, debug=debug)
            if res is not None:
                all_sweeps.append(res)
    return pd.concat(all_sweeps, ignore_index=True) if all_sweeps else pd.DataFrame()


def analyze_sweep_results(df_sweep: pd.DataFrame) -> Dict:
    if df_sweep.empty:
        return {}
    results = {}
    for dataset in df_sweep["dataset"].unique():
        sub = df_sweep[df_sweep["dataset"] == dataset]
        r_global, p_global = spearmanr(sub["rel_distance_to_optimal"].abs(), sub["components_ratio"])
        results[dataset] = {
            "n": len(sub), "spearman_r_distance_vs_fragmentation": r_global, "p_value": p_global,
            "interpretation": (
                "fragmentation sensible au seuil (compatible avec causalité seuil->fragmentation)"
                if p_global < 0.05 and r_global > 0.3
                else "fragmentation quasi-insensible au seuil à opérateur fixe "
                     "(compatible avec une cause structurelle plutôt que la proximité au seuil)"
            ),
        }
    return results


# RAPPORT GLOBAL

def generate_full_report(df_metrics, filtering_results, filtering_small_results,
                          proximity_results, mediation_results, threshold_results,
                          critical_results, op_diff_results) -> str:
    lines = ["="* 80, "RAPPORT POST-BENCHMARK -  ANCRÉE AU GT", "="* 80, "",
             f"Nombre total de lignes chargées : {len(df_metrics)}",
             f"Datasets : {df_metrics['dataset'].unique().tolist()}",
             f"Opérateurs : {df_metrics['operator'].unique().tolist()}",
             f"Lignes avec GT trouvé : {df_metrics['gt'].notna().sum()} / {len(df_metrics)}", ""]

    lines.append("-"* 80)
    lines.append("TEST 1 - DISTRIBUTION DES TAILLES DE COMPOSANTES")
    lines.append("-"* 80)
    for dataset, r in filtering_results.items():
        lines.append(f"\n{dataset.upper()}")
        lines.append(f"Écart inter-opérateurs sur components_ratio : {r['components_ratio_range']:.2f}")
        if r.get('correlation_small_vs_ratio') is not None:
            lines.append(f"Corrélation %petites composantes / components_ratio : "
                          f"r={r['correlation_small_vs_ratio']:.3f}")
        for op, dist in r['operator_distribution'].items():
            lines.append(f"{op}: small={dist['small_pct']:.1f}%, medium={dist['medium_pct']:.1f}%, "
                          f"large={dist['large_pct']:.1f}%")

    lines.append("")
    lines.append(report_filtering_impact_small_components(filtering_small_results))

    lines.append("")
    lines.append(report_threshold_proximity(proximity_results))

    lines.append("")
    lines.append("-"* 80)
    lines.append("TEST 3 - MÉDIATION (causalité)")
    lines.append("-"* 80)
    lines.append("Hypothèse : Proximité au seuil (X) -> Delta Dice vs GT (M) -> Delta fragmentation vs GT (Y)")
    for dataset, r in mediation_results.items():
        lines.append(f"\n{dataset.upper()}")
        if r.get('n', 0) == 0:
            lines.append("Données insuffisantes")
            continue
        lines.append(f"n = {r['n']}")
        lines.append(f"Effet total (X->Y) : r = {r['r_total']:.3f}, p = {r['p_total']:.4f}")
        lines.append(f"Chemin X->M : r = {r['r_xm']:.3f}")
        lines.append(f"Chemin M->Y : r = {r['r_my']:.3f}")
        lines.append(f"Effet direct (X->Y|M) : r = {r['r_direct']:.3f}")
        lines.append(f"Interprétation : {r['interpretation']}")

    lines.append("")
    lines.append("-"* 80)
    lines.append("TEST 4 - DISTRIBUTION DES SEUILS PAR OPÉRATEUR")
    lines.append("-"* 80)
    for dataset, r in threshold_results.items():
        lines.append(f"\n{dataset.upper()}")
        if not r['thresholds']:
            lines.append("Aucun seuil trouvé")
            continue
        lines.append(f"Écart des seuils : {r['threshold_range']:.4f}")
        for op, stats in r['thresholds'].items():
            lines.append(f"{op}: mean={stats['mean']:.4f} ± {stats['std']:.4f}")

    lines.append("")
    lines.append(report_critical_changes(critical_results))

    lines.append("")
    lines.append("-"* 80)
    lines.append("(b) DIFFÉRENCES ENTRE OPÉRATEURS (Friedman + post-hoc, déjà vs GT)")
    lines.append("-"* 80)
    lines.append(summarize_operator_differences(op_diff_results))

    return "\n".join(lines)


# MAIN

def run_post_benchmark_tests(dataset: Optional[str] = None, output: Optional[str] = None,
                              skip_sweep: bool = False, debug: bool = False,
                              config_dir: Optional[Union[str, Path]] = None) -> int:
    """Point d'entrée unique de ce module pas de CLI/`main()` local, tout
    passe par main.py (--run_post_tests), qui appelle cette fonction.

    Si config_dir est fourni, remplace DATASETS par le contenu des YAML de
    ce dossier avant de lancer les tests (cf. load_datasets). Sinon, utilise
    DATASETS tel quel soit le dict codé en dur ci-dessus, soit un dict déjà
    substitué par l'appelant (cf. main.py : DATASETS.clear(); DATASETS.update(...)).
    """
    if config_dir is not None:
        DATASETS.clear()
        DATASETS.update(load_datasets(config_dir))

    print("Chargement des métriques, segmentations et GT...")
    df_metrics = load_all_metrics(debug=debug)
    if df_metrics.empty:
        print("Aucune donnée chargée.")
        return 1
    if dataset:
        df_metrics = df_metrics[df_metrics['dataset'] == dataset]
        if df_metrics.empty:
            print(f"Aucune donnée pour le dataset {dataset}")
            return 1

    print("\nAnalyse des tests...")

    # chronométrage de chaque étape - permet de voir en
    # direct où le temps est passé, sans rien changer aux calculs.
    def _timed(label, fn, *args, **kwargs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        print(f"  [{label}] terminé en {time.time() - t0:.1f}s", flush=True)
        return result

    filtering_results = _timed("Test 1", analyze_filtering_impact, df_metrics)
    filtering_small_results = _timed("Test 1bis", analyze_filtering_impact_small_components, df_metrics)
    proximity_results = _timed("Test 2", analyze_threshold_proximity, df_metrics)
    mediation_results = _timed("Test 3", analyze_mediation, df_metrics)
    threshold_results = _timed("Test 4", analyze_threshold_distribution, df_metrics)
    critical_results = _timed("Test 5", analyze_critical_changes, df_metrics)
    op_diff_results = _timed("Test (b) Friedman", analyze_operator_differences, df_metrics)

    report = generate_full_report(df_metrics, filtering_results, filtering_small_results,
                                   proximity_results, mediation_results, threshold_results,
                                   critical_results, op_diff_results)
    print("\n"+ report)

    # (c) bootstrap de médiation, séparé car plus coûteux (statsmodels + bootstrap)
    print("\n"+ "="* 80)
    print("(c) BOOTSTRAP DE L'EFFET INDIRECT")
    print("="* 80)
    df_med = build_mediation_dataframe(df_metrics)
    for ds in df_med["dataset"].unique():
        sub_med = df_med[df_med["dataset"] == ds]
        if sub_med.empty:
            continue
        t0 = time.time()
        res = bootstrap_mediation(sub_med, n_boot=2000)
        print(f"\n{ds.upper()} (en {time.time() - t0:.1f}s)")
        print(f"n clusters = {res['n_clusters']}, effet indirect (a*b) = {res['indirect_effect_ab']:.4f}")
        print(f"IC 95% bootstrap = [{res['indirect_ci_95'][0]:.4f}, {res['indirect_ci_95'][1]:.4f}]")
        print(f"% médié = {res['proportion_mediated']*100:.1f}% -> {res['interpretation']}")

    if not skip_sweep:
        print("\n"+ "="* 80)
        print("(a) SWEEP DE SEUIL A OPÉRATEUR FIXE (gaussian)")
        print("="* 80)
        df_sweep = run_threshold_sweep_all(df_metrics, operator="gaussian",
                                            n_cases_per_dataset=10, n_steps=15, debug=debug)
        if not df_sweep.empty:
            for ds, r in analyze_sweep_results(df_sweep).items():
                print(f"\n{ds.upper()} n={r['n']}")
                print(f"Spearman r = {r['spearman_r_distance_vs_fragmentation']:.3f}, p = {r['p_value']:.4g}")
                print(f"-> {r['interpretation']}")
            df_sweep.to_csv("threshold_sweep_results.csv", index=False)
        else:
            print("Aucun sweep n'a pu être calculé.")

    if output:
        Path(output).write_text(report, encoding='utf-8')
        print(f"\nRapport sauvegardé dans {output}")

    return 0
