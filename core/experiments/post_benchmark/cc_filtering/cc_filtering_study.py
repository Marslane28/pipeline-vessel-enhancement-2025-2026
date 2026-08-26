import sys, os, pickle, time
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label as scipy_label
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.experiments.metrics import detailed_metrics
from core.config.cc_filtering import CCFilteringConfig
from core.experiments.post_benchmark.cc_filtering.cc_filtering_analysis import (plot_metrics_vs_min_size)

STRUCT_26 = np.ones((3, 3, 3), dtype=np.uint8)
SEED = 42
np.random.seed(SEED)


def resolve_patient_files(cfg: CCFilteringConfig, patient_id: str):
    if cfg.dataset == "vascusynth":
        img_filename = f"{patient_id}_rician_{cfg.noise_level:.1f}.nii.gz"
        img_file = cfg.images_dir / patient_id / img_filename
        if not img_file.exists():
            candidates = list((cfg.images_dir / patient_id).glob(f"{patient_id}_rician_*.nii.gz"))
            img_file = candidates[0] if candidates else None
        label_file = cfg.labels_dir / patient_id / f"{patient_id}_vessels_gt.nii.gz"
        mask = None
        result_file = cfg.results_dir / "results" / f"results_patient_{patient_id}_images.nii"
        if not result_file.exists():
            candidates = list(cfg.results_dir.glob(f"results_{patient_id}_*.nii"))
            result_file = candidates[0] if candidates else None
    else:  # bullitt / ircad - même convention de nommage
        img_file = cfg.images_dir / f"patient_{patient_id}_images.nii.gz"
        label_file = cfg.labels_dir / f"patient_{patient_id}_label.nii.gz"
        result_file = cfg.results_dir / "results" / f"results_patient_{patient_id}_images.nii"
        mask = None
        if cfg.masks_dir is not None:
            mask_file = cfg.masks_dir / f"patient_{patient_id}_liver.nii.gz"
            mask = nib.load(mask_file).get_fdata() > 0.5 if mask_file.exists() else None

    if (img_file is None or not Path(img_file).exists() or not label_file.exists()
            or result_file is None or not Path(result_file).exists()):
        return None, None, None, None
    return img_file, label_file, mask, result_file


def filter_by_component_size(binary_mask, min_size, gt_bin):
    if min_size == 0:
        return binary_mask.copy(), 0, 0, np.nan

    labeled, n_components = scipy_label(binary_mask, structure=STRUCT_26)
    filtered_mask = np.zeros_like(binary_mask, dtype=bool)
    n_removed = voxels_removed = 0
    removed_overlap_num, removed_overlap_den = 0.0, 0

    for comp_id in range(1, n_components + 1):
        comp_mask = (labeled == comp_id)
        comp_size = int(comp_mask.sum())
        if comp_size >= min_size:
            filtered_mask[comp_mask] = True
        else:
            n_removed += 1
            voxels_removed += comp_size
            overlap = int(np.logical_and(comp_mask, gt_bin).sum())
            removed_overlap_num += overlap
            removed_overlap_den += comp_size

    removed_gt_overlap_weighted = (
        removed_overlap_num / removed_overlap_den if removed_overlap_den > 0 else np.nan
    )
    return filtered_mask, n_removed, voxels_removed, removed_gt_overlap_weighted


def process_patient(cfg: CCFilteringConfig, patient_id: str) -> list:
    rows = []
    image_file, label_file, mask, result_file = resolve_patient_files(cfg, patient_id)
    if image_file is None:
        print(f"ERROR: Missing files for patient {patient_id}, skip")
        return rows

    gt_data = nib.load(label_file).get_fdata() > 0.5
    with open(result_file, 'rb') as f:
        results = pickle.load(f)
    derivators = results.get('derivator', results)

    print(f"\n{'='*60}\nPatient {patient_id}\n{'='*60}")

    for op_name in cfg.operators:
        if op_name not in derivators:
            print(f"  warning: operator {op_name} missing, skip")
            continue
        op_data = derivators[op_name]
        seg_data = getattr(op_data, 'data_segmented', None)
        if seg_data is None:
            seg_data = getattr(op_data, 'segmented', None)
            print(f"  warning: no segmentation for {op_name}, skip")

        threshold = getattr(op_data, 'threshold', 0.5)
        pred_bin = (seg_data > threshold).astype(bool)

        for min_size in cfg.min_sizes:
            filtered_bin, n_removed, voxels_removed, removed_gt_overlap = filter_by_component_size(
                pred_bin, min_size, gt_data)

            m = detailed_metrics(filtered_bin.astype(bool), gt_data.astype(bool),
                                  mask=mask, threshold=0.5, skip_bifurcation=False)

            rows.append({
                'dataset': cfg.dataset, 'patient': patient_id, 'operator': op_name,
                'min_size': min_size, 'n_cc_removed': n_removed,
                'voxels_removed': voxels_removed,
                'removed_gt_overlap_weighted': removed_gt_overlap,
                **{k: m[k] for k in (
                    'dice', 'mcc', 'cldice', 'roc', 'pr', 'fragmentation_ratio',
                    'components_ratio', 'n_components_pred', 'n_components_gt',
                    'skeleton_component_connectivity', 'largest_gt_recall',
                    'largest_component_overlap', 'bifurcation_detection_rate',
                    'bifurcation_precision', 'bifurcation_f1', 'tp', 'fp', 'fn', 'tn')},
            })
            print(f"  {op_name:14s} min_size={min_size:4d}  Dice={m['dice']:.4f}  "
                  f"clDice={m['cldice']:.4f}  Frag={m['fragmentation_ratio']:.2f}  "
                  f"CC removed={n_removed} (GT overlap={removed_gt_overlap:.3f})")
    return rows


def run_cc_filtering_study(cfg: CCFilteringConfig):
    from pathlib import Path
    patients = cfg.patient_list()
    print(f"\n{'='*80}\nCC FILTERING STUDY — dataset={cfg.dataset}\n{'='*80}")
    print(f"Patients : {len(patients)} | Opérateurs : {len(cfg.operators)} | "
          f"Seuils : {cfg.min_sizes}")

    start_time = time.time()
    df_all = pd.DataFrame()
    if cfg.output_csv.exists():
        df_all = pd.read_csv(cfg.output_csv)
        done = set(df_all['patient'].unique())
        remaining = [p for p in patients if p not in done]
        print(f"Existing file: {len(df_all)} rows, {len(remaining)} patients restants")
    else:
        remaining = patients

    new_rows = []
    for i, patient_id in enumerate(remaining, 1):
        print(f"\nPatient {patient_id} ({i}/{len(remaining)})")
        try:
            rows = process_patient(cfg, patient_id)
        except Exception as e:
            print(f"ERROR for patient {patient_id}: {e}")
            import traceback; traceback.print_exc()
            rows = []
        if rows:
            new_rows.extend(rows)
            df_new = pd.DataFrame(new_rows)
            df_all = pd.concat([df_all, df_new], ignore_index=True) if not df_all.empty else df_new
            df_all.to_csv(cfg.output_csv, index=False)
            print(f"  Progressive save: {len(df_all)} rows saved")
            new_rows = []

    if df_all.empty:
        print("\nWARNING: No data saved")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\nExported: {cfg.output_csv} ({len(df_all)} rows) in {elapsed/60:.1f} min")

    if cfg.generate_figure:
        plot_metrics_vs_min_size(df_all, cfg.output_dir)
        print(f"Figure saved in {cfg.output_dir}")

    return df_all
