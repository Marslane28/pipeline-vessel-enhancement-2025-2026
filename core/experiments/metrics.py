import numpy as np
from numpy import ndarray
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, roc_curve as sk_roc_curve, precision_recall_curve as sk_pr_curve
from scipy.ndimage import label as scipy_label, binary_dilation, convolve
from logging import getLogger
logger = getLogger(__name__)

try:
    from skimage.morphology import skeletonize
except ImportError:
    from skimage.morphology import skeletonize_3d as skeletonize


# Helpers

def _compute_skeletons_once(v_pred_bin, v_gt_bin):
    """
    Calcule les squelettes une seule fois.
    Fonction helper INTERNE pour éviter les recalculs.
    """
    s_pred = _skeletonize_3d(v_pred_bin)
    s_gt = _skeletonize_3d(v_gt_bin)
    return s_pred, s_gt


def _get_skeleton_or_compute(v, skeleton_param):
    """
    Si skeleton_param est None, calcule. Sinon, l'utilise.
    Utile pour backward compatibility.
    """
    if skeleton_param is None:
        return _skeletonize_3d(v)
    return skeleton_param


# clDice (topological metric)

def _skeletonize_3d(mask: np.ndarray) -> np.ndarray:
    """Squelette 3D d'un masque binaire."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return skeletonize(mask.astype(bool)) > 0


def cldice_fg(v_pred: np.ndarray, v_gt: np.ndarray, smooth: float = 1e-7) -> float:
    """
    Calcule le clDice pour le foreground (vascularisation).
    Mesure topologique qui évalue la préservation de la connectivité.

    clDice = 2 * (tprec * tsens) / (tprec + tsens)
    où tprec = (squelette_pred ∩ GT) / squelette_pred
    et tsens = (squelette_GT ∩ pred) / squelette_GT
    """
    v_pred = v_pred.astype(bool)
    v_gt = v_gt.astype(bool)

    s_pred = _skeletonize_3d(v_pred)
    s_gt = _skeletonize_3d(v_gt)

    inter_tprec = np.logical_and(s_pred, v_gt).sum()
    n_skel_pred = s_pred.sum()
    tprec = (inter_tprec + smooth) / (n_skel_pred + smooth) if n_skel_pred > 0 else 0.0

    inter_tsens = np.logical_and(s_gt, v_pred).sum()
    n_skel_gt = s_gt.sum()
    tsens = (inter_tsens + smooth) / (n_skel_gt + smooth) if n_skel_gt > 0 else 0.0

    if tprec + tsens == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))



def cldice(v_pred: np.ndarray, v_gt: np.ndarray, mask: ndarray = None,
           compute_background: bool = False, **kwargs) -> float:
    """
    Calcule le clDice (optionnellement avec background).
    """
    v_pred = v_pred.astype(bool)
    v_gt = v_gt.astype(bool)

    if mask is not None:
        mask_bin = (mask > 0).astype(bool)
        v_pred = v_pred & mask_bin
        v_gt = v_gt & mask_bin

    s_pred_cached = kwargs.get('_s_pred', None)
    s_gt_cached = kwargs.get('_s_gt', None)
    
    s_pred = _get_skeleton_or_compute(v_pred, s_pred_cached)
    s_gt = _get_skeleton_or_compute(v_gt, s_gt_cached)

    inter_tprec = np.logical_and(s_pred, v_gt).sum()
    n_skel_pred = s_pred.sum()
    tprec = (inter_tprec + 1e-7) / (n_skel_pred + 1e-7) if n_skel_pred > 0 else 0.0

    inter_tsens = np.logical_and(s_gt, v_pred).sum()
    n_skel_gt = s_gt.sum()
    tsens = (inter_tsens + 1e-7) / (n_skel_gt + 1e-7) if n_skel_gt > 0 else 0.0

    if tprec + tsens == 0:
        fg_score = 0.0
    else:
        fg_score = float(2 * tprec * tsens / (tprec + tsens))

    if not compute_background:
        return fg_score

    bg_pred = ~v_pred
    bg_gt = ~v_gt

    if mask is not None:
        bg_pred = bg_pred & mask_bin
        bg_gt = bg_gt & mask_bin

    bg_s_pred = _get_skeleton_or_compute(bg_pred, None)
    bg_s_gt = _get_skeleton_or_compute(bg_gt, None)
    
    inter_tprec_bg = np.logical_and(bg_s_pred, bg_gt).sum()
    n_skel_pred_bg = bg_s_pred.sum()
    tprec_bg = (inter_tprec_bg + 1e-7) / (n_skel_pred_bg + 1e-7) if n_skel_pred_bg > 0 else 0.0

    inter_tsens_bg = np.logical_and(bg_s_gt, bg_pred).sum()
    n_skel_gt_bg = bg_s_gt.sum()
    tsens_bg = (inter_tsens_bg + 1e-7) / (n_skel_gt_bg + 1e-7) if n_skel_gt_bg > 0 else 0.0

    if tprec_bg + tsens_bg == 0:
        bg_score = 0.0
    else:
        bg_score = float(2 * tprec_bg * tsens_bg / (tprec_bg + tsens_bg))

    return (fg_score + bg_score) / 2


# Composantes connexes

def largest_gt_recall(v_pred, v_gt, _gt_labels=None, _n_gt=None):
    """
    Recall du tronc principal GT : combien de voxels du tronc principal GT
    sont retrouvés dans la prédiction.
    """
    v_pred = v_pred.astype(bool)
    v_gt = v_gt.astype(bool)

    if _gt_labels is None:
        gt_labels, n_gt = scipy_label(v_gt, structure=np.ones((3, 3, 3)))
    else:
        gt_labels, n_gt = _gt_labels, _n_gt

    if n_gt == 0:
        return 0.0

    gt_sizes = np.bincount(gt_labels.ravel())[1:]
    largest_gt_idx = np.argmax(gt_sizes) + 1
    largest_gt = (gt_labels == largest_gt_idx)

    if largest_gt.sum() == 0:
        return 0.0

    intersection = np.logical_and(largest_gt, v_pred).sum()
    return float(intersection / largest_gt.sum())


def largest_component_overlap(v_pred, v_gt, _gt_labels=None, _n_gt=None,
                               _pred_labels=None, _n_pred=None):
    """
    Overlap entre les plus grandes composantes GT et prédiction.
    Mesure si le tronc principal GT est connecté dans la prédiction.
    """
    v_pred = v_pred.astype(bool)
    v_gt = v_gt.astype(bool)

    if _gt_labels is None:
        gt_labels, n_gt = scipy_label(v_gt, structure=np.ones((3, 3, 3)))
    else:
        gt_labels, n_gt = _gt_labels, _n_gt

    if n_gt == 0:
        return 0.0

    gt_sizes = np.bincount(gt_labels.ravel())[1:]
    largest_gt_idx = np.argmax(gt_sizes) + 1
    largest_gt = (gt_labels == largest_gt_idx)

    if _pred_labels is None:
        pred_labels, n_pred = scipy_label(v_pred, structure=np.ones((3, 3, 3)))
    else:
        pred_labels, n_pred = _pred_labels, _n_pred

    if n_pred == 0:
        return 0.0

    pred_sizes = np.bincount(pred_labels.ravel())[1:]
    largest_pred_idx = np.argmax(pred_sizes) + 1
    largest_pred = (pred_labels == largest_pred_idx)

    intersection = np.logical_and(largest_gt, largest_pred).sum()

    if largest_gt.sum() == 0:
        return 0.0

    return float(intersection / largest_gt.sum())


def fragmentation_ratio(v_pred, v_gt, _n_gt=None, _n_pred=None):
    """
    Ratio du nombre de composantes prédites sur GT.
    Plus le ratio est élevé, plus la prédiction est fragmentée.
    """
    if _n_gt is not None and _n_pred is not None:
        n_gt, n_pred = _n_gt, _n_pred
    else:
        v_pred = v_pred.astype(bool)
        v_gt = v_gt.astype(bool)
        _, n_gt = scipy_label(v_gt, structure=np.ones((3, 3, 3)))
        _, n_pred = scipy_label(v_pred, structure=np.ones((3, 3, 3)))

    if n_gt == 0:
        return float('inf') if n_pred > 0 else 0.0

    return float(n_pred / n_gt)


def skeleton_component_connectivity(v_pred, v_gt, sk_gt_cached=None):
    """
    Connectivité du squelette : pour chaque composante du squelette GT,
    proportion couverte par la prédiction.
    """
    v_pred = v_pred.astype(bool)
    v_gt = v_gt.astype(bool)

    sk_gt = _get_skeleton_or_compute(v_gt, sk_gt_cached)

    if not sk_gt.any():
        return 0.0

    gt_labels, n = scipy_label(sk_gt, structure=np.ones((3, 3, 3)))

    scores = []
    for i in range(1, n + 1):
        comp = (gt_labels == i)
        if comp.sum() == 0:
            continue
        covered = np.logical_and(comp, v_pred).sum()
        total = comp.sum()
        if total > 0:
            scores.append(covered / total)

    return float(np.mean(scores)) if scores else 0.0


def connected_components_metrics(v_pred: np.ndarray, v_gt: np.ndarray, mask: ndarray = None, **kwargs) -> dict:
    """
    Analyse les composantes connexes pour évaluer la continuité topologique.
    """
    v_pred = v_pred.astype(bool)
    v_gt = v_gt.astype(bool)

    if mask is not None:
        mask_bin = (mask > 0).astype(bool)
        v_pred = v_pred & mask_bin
        v_gt = v_gt & mask_bin

    struct = np.ones((3, 3, 3))

    pred_labels, n_pred = scipy_label(v_pred, structure=struct)
    gt_labels, n_gt = scipy_label(v_gt, structure=struct)

    pred_sizes = np.bincount(pred_labels.ravel())[1:]
    gt_sizes = np.bincount(gt_labels.ravel())[1:]

    pred_small = (pred_sizes < 10).sum()
    pred_medium = ((pred_sizes >= 10) & (pred_sizes < 100)).sum()
    pred_large = (pred_sizes >= 100).sum()

    gt_small = (gt_sizes < 10).sum()
    gt_medium = ((gt_sizes >= 10) & (gt_sizes < 100)).sum()
    gt_large = (gt_sizes >= 100).sum()

    sk_gt_cached = kwargs.get('_sk_gt', None)
    sk_gt = _get_skeleton_or_compute(v_gt, sk_gt_cached)

    frag_ratio = fragmentation_ratio(v_pred, v_gt, _n_gt=n_gt, _n_pred=n_pred)

    return {
        "n_components_pred": int(n_pred),
        "n_components_gt": int(n_gt),
        "components_ratio": float(n_pred / n_gt) if n_gt > 0 else float('inf'),
        "excess_components": int(max(0, n_pred - n_gt)),
        "missing_components": int(max(0, n_gt - n_pred)),
        "pred_comp_sizes_mean": float(pred_sizes.mean()) if len(pred_sizes) > 0 else 0.0,
        "pred_comp_sizes_median": float(np.median(pred_sizes)) if len(pred_sizes) > 0 else 0.0,
        "gt_comp_sizes_mean": float(gt_sizes.mean()) if len(gt_sizes) > 0 else 0.0,
        "gt_comp_sizes_median": float(np.median(gt_sizes)) if len(gt_sizes) > 0 else 0.0,
        "pred_small_components": int(pred_small),
        "pred_medium_components": int(pred_medium),
        "pred_large_components": int(pred_large),
        "gt_small_components": int(gt_small),
        "gt_medium_components": int(gt_medium),
        "gt_large_components": int(gt_large),
        "largest_component_pred": int(pred_sizes.max()) if len(pred_sizes) > 0 else 0,
        "largest_component_gt": int(gt_sizes.max()) if len(gt_sizes) > 0 else 0,
        "largest_ratio": float(pred_sizes.max() / gt_sizes.max()) if len(pred_sizes) > 0 and len(gt_sizes) > 0 and gt_sizes.max() > 0 else 0.0,
        "largest_component_overlap": float(largest_component_overlap(
            v_pred, v_gt, _gt_labels=gt_labels, _n_gt=n_gt, _pred_labels=pred_labels, _n_pred=n_pred
        )),
        "fragmentation_ratio": float(frag_ratio),
        "skeleton_component_connectivity": float(skeleton_component_connectivity(v_pred, v_gt, sk_gt)),
        "largest_gt_recall": float(largest_gt_recall(v_pred, v_gt, _gt_labels=gt_labels, _n_gt=n_gt)),
        "gt_fragmentation": float(frag_ratio),
    }


# Métriques originales

def dice(y_pred, y_true, mask=None):
    y_pred = y_pred.astype(bool)
    y_true = y_true.astype(bool)

    if mask is not None:
        m = mask > 0
        y_pred = y_pred[m]
        y_true = y_true[m]

    inter = np.logical_and(y_pred, y_true).sum()
    denom = y_pred.sum() + y_true.sum()

    return 2 * inter / denom if denom > 0 else 0.0


def mcc(y_pred: ndarray, y_true: ndarray, mask: ndarray = None) -> float:
    if mask is not None:
        y_pred = y_pred[mask > 0]
        y_true = y_true[mask > 0]
    return matthews_corrcoef(y_true.ravel(), y_pred.ravel())


def roc(y_pred: ndarray, y_true: ndarray, mask: ndarray = None) -> float:
    if mask is not None:
        y_pred = y_pred[mask > 0]
        y_true = y_true[mask > 0]
    return roc_auc_score(y_true.ravel(), y_pred.ravel())


def pr(y_pred: ndarray, y_true: ndarray, mask: ndarray = None) -> float:
    if mask is not None:
        y_pred = y_pred[mask > 0]
        y_true = y_true[mask > 0]
    return average_precision_score(y_true.ravel(), y_pred.ravel())


def roc_curve(y_true: ndarray, y_score: ndarray, pos_label: int = 1):
    return sk_roc_curve(y_true.ravel(), y_score.ravel(), pos_label=pos_label)


def precision_recall_curve(y_true: ndarray, y_score: ndarray, pos_label: int = 1):
    return sk_pr_curve(y_true.ravel(), y_score.ravel(), pos_label=pos_label)


def confusion_matrix(y_pred: ndarray, y_true: ndarray, mask: ndarray = None) -> dict:
    y_pred = np.array(y_pred).astype(int)
    y_true = np.array(y_true).astype(int)

    if mask is not None:
        mask_bool = mask > 0 if mask.dtype != bool else mask
        y_pred = y_pred[mask_bool]
        y_true = y_true[mask_bool]

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    return {
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'total': int(tp + fp + fn + tn),
        'gt_vessels': int(tp + fn),
        'pred_vessels': int(tp + fp)
    }


def print_confusion_matrix(y_pred: ndarray, y_true: ndarray, mask: ndarray = None,
                            name: str = "", logger=None) -> dict:
    cm = confusion_matrix(y_pred, y_true, mask)

    lines = [
        f"MATRICE DE CONFUSION - {name.upper() if name else 'RESULTATS'}",
        f"TP (vrais positifs) : {cm['tp']:>12,d} (bien détectés)",
        f"FP (faux positifs) : {cm['fp']:>12,d} (surdétection)",
        f"FN (faux négatifs) : {cm['fn']:>12,d} (sous-détection)",
        f"TN (vrais négatifs) : {cm['tn']:>12,d} (bien exclus)",
        f"Total voxels analysés : {cm['total']:>12,d}",
        f"Vaisseaux dans GT : {cm['gt_vessels']:>12,d}",
        f"Vaisseaux prédits : {cm['pred_vessels']:>12,d}",
        f"{'='*50}"
    ]

    for line in lines:
        if logger:
            logger.info(line)
        else:
            print(line)

    return cm


# DÉTECTION DE BIFURCATIONS

_EUCLIDEAN_STRUCT_CACHE = {}
def _euclidean_dilation_struct(radius: int) -> np.ndarray:
    """
    Génère un élément structurant sphérique (distance euclidienne ≤ radius).
    """
    d = 2 * radius + 1
    idx = np.arange(d) - radius
    x, y, z = np.meshgrid(idx, idx, idx, indexing='ij')
    return (x**2 + y**2 + z**2) <= radius**2


# Kernel 3×3×3 sans le voxel central, utilisé pour compter les voisins squelette.

_NEIGHBOR_KERNEL = np.ones((3, 3, 3), dtype=np.uint8)
_NEIGHBOR_KERNEL[1, 1, 1] = 0

def _get_euclidean_struct(radius: int) -> np.ndarray:
    if radius not in _EUCLIDEAN_STRUCT_CACHE:
        _EUCLIDEAN_STRUCT_CACHE[radius] = _euclidean_dilation_struct(radius)
    return _EUCLIDEAN_STRUCT_CACHE[radius]

def detect_bifurcations(skeleton: np.ndarray, mask: np.ndarray = None,
                         threshold: int = 3, sk_gt_cached: np.ndarray = None) -> np.ndarray:
    """
    Détecte les points de bifurcation dans un squelette 3D.
    """
    # Utiliser le squelette pré-calculé si disponible
    if sk_gt_cached is not None:
        skeleton = sk_gt_cached
    else:
        skeleton = skeleton.astype(bool)

    # Appliquer le masque si présent
    if mask is not None:
        skeleton = skeleton & (mask > 0)

    if not skeleton.any():
        return np.zeros_like(skeleton, dtype=bool)

    # Compter les voisins squelette (hors voxel central)
    neighbor_count = convolve(
        skeleton.astype(np.uint8),
        _NEIGHBOR_KERNEL,
        mode='constant',
        cval=0,
    )

    # Un voxel squelette avec >= threshold voisins squelette est une bifurcation
    bifurcations = skeleton & (neighbor_count >= threshold)

    return bifurcations


def compute_bdr(
    bif_pred: np.ndarray,
    bif_gt: np.ndarray,
    tolerance_radius: int = 3,
) -> dict:
    """
    Calcule le BDR (Bifurcation Detection Rate) entre deux ensembles de bifurcations.

    La tolérance est euclidienne : une bifurcation GT est détectée si une
    bifurcation prédite se trouve dans une sphère de rayon `tolerance_radius`.
    """
    gt_labels, n_bif_gt = scipy_label(bif_gt, structure=np.ones((3, 3, 3)))
    pred_labels, n_bif_pred = scipy_label(bif_pred, structure=np.ones((3, 3, 3)))

    if n_bif_gt == 0:
        return {
            'tp': 0, 'fp': n_bif_pred, 'fn': 0,
            'recall': 0.0, 'precision': 0.0, 'f1': 0.0,
            'n_gt': 0, 'n_pred': n_bif_pred,
        }

    sphere_struct = _get_euclidean_struct(tolerance_radius)

    # Dilatation globale une seule fois (au lieu d'une dilatation par composante) :
    # par symétrie de l'élément structurant sphérique, "comp dilaté touche bif_pred"
    # est équivalent à "comp touche bif_pred dilaté".
    bif_pred_dilated = binary_dilation(bif_pred, structure=sphere_struct)
    bif_gt_dilated = binary_dilation(bif_gt, structure=sphere_struct)

    # TP / FN : bifurcations GT couvertes par au moins une prédiction proche
    tp = 0
    for i in range(1, n_bif_gt + 1):
        comp = (gt_labels == i)
        if np.logical_and(comp, bif_pred_dilated).any():
            tp += 1

    fn = n_bif_gt - tp

    # FP : bifurcations prédites sans aucun GT proche
    fp = 0
    for i in range(1, n_bif_pred + 1):
        comp = (pred_labels == i)
        if not np.logical_and(comp, bif_gt_dilated).any():
            fp += 1

    recall = tp / n_bif_gt if n_bif_gt > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'n_gt': n_bif_gt,
        'n_pred': n_bif_pred,
    }


def find_optimal_bifurcation_threshold(
    sk_pred: np.ndarray,
    sk_gt: np.ndarray,
    mask: np.ndarray = None,
    thresholds: list = [3, 4, 5],
    tolerance_radius: int = 3,
    optimize_on: str = 'f1',
    verbose: bool = True,
) -> dict:
    """
    Trouve le seuil de détection de bifurcations qui maximise la métrique cible.
    """
    # Appliquer le masque une seule fois ici
    if mask is not None:
        mask_bin = mask > 0
        sk_pred = sk_pred & mask_bin
        sk_gt = sk_gt & mask_bin

    # GT : seuil fixe à 3 (détecte toutes les jonctions à ≥ 3 branches)
    # mask=None car déjà appliqué ci-dessus
    bif_gt = detect_bifurcations(sk_gt, mask=None, threshold=3)
    _, n_bif_gt = scipy_label(bif_gt, structure=np.ones((3, 3, 3)))

    if n_bif_gt == 0:
        if verbose:
            print("[BIF OPT] No bifurcations found in GT")
        empty = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                 'n_gt': 0, 'n_pred': 0, 'tp': 0, 'fp': 0, 'fn': 0}
        return {
            'best_threshold': thresholds[0],
            'best_f1': 0.0,
            'best_precision': 0.0,
            'best_recall': 0.0,
            'results': {t: empty.copy() for t in thresholds},
        }

    best_threshold = thresholds[0]
    best_score = -1.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    results = {}

    for t in thresholds:
        # mask=None : déjà appliqué sur sk_pred ci-dessus
        bif_pred = detect_bifurcations(sk_pred, mask=None, threshold=t)
        metrics = compute_bdr(bif_pred, bif_gt, tolerance_radius)

        results[t] = {
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'n_gt': metrics['n_gt'],
            'n_pred': metrics['n_pred'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
        }

        score = metrics[optimize_on]

        if verbose:
            print(
                f"[BIF OPT] threshold={t}: F1={metrics['f1']:.3f}, "
                f"Recall={metrics['recall']:.3f}, Precision={metrics['precision']:.3f}, "
                f"GT={metrics['n_gt']}, Pred={metrics['n_pred']}, "
                f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}"
            )

        if score > best_score:
            best_score = score
            best_threshold = t
            best_precision = metrics['precision']
            best_recall = metrics['recall']
            best_f1 = metrics['f1']

    if verbose:
        print(
            f"[BIF OPT] Best threshold: {best_threshold} "
            f"(F1={best_f1:.3f}, Recall={best_recall:.3f}, Precision={best_precision:.3f})"
        )

    return {
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'results': results,
    }


def bifurcation_detection_rate(
    v_pred: np.ndarray,
    v_gt: np.ndarray,
    mask: np.ndarray = None,
    tolerance_radius: int = 3,
    bifurcation_threshold: int = 3,
    optimize_threshold: bool = False,
    thresholds_to_test: list = [3, 4, 5],
    voxel_spacing_mm: float = None,
    **kwargs
) -> dict:
    """
    Bifurcation Detection Rate avec tolérance spatiale euclidienne.
    """
    v_pred = v_pred.astype(bool)
    v_gt = v_gt.astype(bool)

    # masque appliqué une seule fois ici
    if mask is not None:
        mask_bin = mask > 0
        v_pred = v_pred & mask_bin
        v_gt = v_gt & mask_bin

    # Conversion mm → voxels si spacing fourni
    if voxel_spacing_mm is not None:
        tolerance_radius_vx = max(1, int(round(tolerance_radius / voxel_spacing_mm)))
        logger.info(
            f"[BDR] tolerance {tolerance_radius}mm → {tolerance_radius_vx}vx "
            f"(spacing={voxel_spacing_mm}mm/vx)"
        )
    else:
        tolerance_radius_vx = tolerance_radius

    s_pred_cached = kwargs.get('_s_pred', None)
    s_gt_cached = kwargs.get('_s_gt', None)
    
    sk_pred = _get_skeleton_or_compute(v_pred, s_pred_cached)
    sk_gt = _get_skeleton_or_compute(v_gt, s_gt_cached)

    if optimize_threshold:
        opt_results = find_optimal_bifurcation_threshold(
            sk_pred=sk_pred,
            sk_gt=sk_gt,
            mask=None,
            thresholds=thresholds_to_test,
            tolerance_radius=tolerance_radius_vx,
            verbose=True,
        )
        bifurcation_threshold = opt_results['best_threshold']
        logger.info(f"[BDR] Using optimized threshold: {bifurcation_threshold}")

    bif_gt = detect_bifurcations(sk_gt, mask=None, threshold=bifurcation_threshold)
    bif_pred = detect_bifurcations(sk_pred, mask=None, threshold=bifurcation_threshold)

    logger.debug(
        f"[BDR] sk_gt={sk_gt.sum()} vx | bif_gt={bif_gt.sum()} vx | "
        f"sk_pred={sk_pred.sum()} vx | bif_pred={bif_pred.sum()} vx"
    )

    metrics = compute_bdr(bif_pred, bif_gt, tolerance_radius_vx)

    logger.info(
        f"[BDR] GT={metrics['n_gt']}, Pred={metrics['n_pred']}, "
        f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, "
        f"Recall={metrics['recall']:.3f}, Precision={metrics['precision']:.3f}, "
        f"F1={metrics['f1']:.3f}, "
        f"radius={tolerance_radius_vx}vx (euclidien), threshold={bifurcation_threshold}"
    )

    return {
        'bifurcation_detection_rate': float(metrics['recall']),
        'bifurcation_precision': float(metrics['precision']),
        'bifurcation_f1': float(metrics['f1']),
        'n_bifurcations_gt': metrics['n_gt'],
        'n_bifurcations_pred': metrics['n_pred'],
        'n_bifurcations_detected': metrics['tp'],
        'bifurcation_tp': metrics['tp'],
        'bifurcation_fp': metrics['fp'],
        'bifurcation_fn': metrics['fn'],
        'bifurcation_threshold': bifurcation_threshold,
        'tolerance_radius_vx': tolerance_radius_vx,
    }

# Groupes "gratuits"(dérivés de la même matrice de confusion) : toujours
# calculés ensemble, aucun intérêt à les séparer (coût négligeable).
_CM_DERIVED_METRICS = {"dice", "mcc", "recall", "specificity", "precision", "accuracy"}

# Groupes coûteux, réellement skippables.
_ALL_METRIC_GROUPS = _CM_DERIVED_METRICS | {"roc", "pr", "cldice", "components", "bifurcation"}


def _normalize_metrics_selection(metrics) -> set:
    """
    Traduit le paramètre `metrics` ("all"ou liste de noms) en un set
    normalisé de groupes à calculer. Lève ValueError si un nom est inconnu.
    """
    if metrics is None or metrics == "all":
        return set(_ALL_METRIC_GROUPS)
    selection = set(metrics)
    unknown = selection - _ALL_METRIC_GROUPS
    if unknown:
        raise ValueError(
            f"Métrique(s) inconnue(s) : {unknown}. "
            f"Valeurs valides : {sorted(_ALL_METRIC_GROUPS)}"
        )
    return selection


_EMPTY_CONN_METRICS = {
    "n_components_pred": 0, "n_components_gt": 0, "components_ratio": 0.0,
    "excess_components": 0, "missing_components": 0,
    "pred_comp_sizes_mean": 0.0, "pred_comp_sizes_median": 0.0,
    "gt_comp_sizes_mean": 0.0, "gt_comp_sizes_median": 0.0,
    "pred_small_components": 0, "pred_medium_components": 0, "pred_large_components": 0,
    "gt_small_components": 0, "gt_medium_components": 0, "gt_large_components": 0,
    "largest_component_pred": 0, "largest_component_gt": 0, "largest_ratio": 0.0,
    "largest_component_overlap": 0.0, "fragmentation_ratio": 0.0,
    "skeleton_component_connectivity": 0.0, "largest_gt_recall": 0.0,
    "gt_fragmentation": 0.0,
}

_EMPTY_BIF_METRICS = {
    'bifurcation_detection_rate': 0.0, 'bifurcation_precision': 0.0, 'bifurcation_f1': 0.0,
    'n_bifurcations_gt': 0, 'n_bifurcations_pred': 0, 'n_bifurcations_detected': 0,
    'bifurcation_tp': 0, 'bifurcation_fp': 0, 'bifurcation_fn': 0,
}


# Métriques

def detailed_metrics(y_pred: ndarray, y_true: ndarray, mask: ndarray = None,
                     threshold=None, skip_bifurcation: bool = False,
                     metrics="all") -> dict:
    """
    Calcule les métriques sélectionnées.
    """
    selected = _normalize_metrics_selection(metrics)

    if threshold is None:
        threshold = 0.5
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
        logger.warning(f"No threshold provided for detailed_metrics, defaulting to {threshold}")

    # ── BINARISATION DE LA PRÉDICTION ──
    if isinstance(threshold, tuple):
        if np.array_equal(np.unique(y_pred), [0, 1]) or np.array_equal(np.unique(y_pred), [0.0, 1.0]):
            y_pred_bin = y_pred.astype(bool)
        else:
            low, high = threshold
            y_pred_bin = (y_pred > low) & (y_pred < high)
    else:
        y_pred_bin = (y_pred > threshold).astype(bool) if y_pred.dtype != bool else y_pred.astype(bool)

    # ── BINARISATION DE LA GT ──
    if y_true.dtype == bool:
        y_true_bin = y_true.astype(bool)
    elif y_true.dtype == np.uint8:
        if np.all((y_true == 0) | (y_true == 255)):
            y_true_bin = (y_true > 127).astype(bool)
        else:
            y_true_bin = (y_true > 0).astype(bool)
    else:
        y_true_bin = (y_true > 0.5).astype(bool) if y_true.max() <= 1.0 else (y_true > 127).astype(bool)

    need_skeletons = bool(selected & {"cldice", "components", "bifurcation"})

    if need_skeletons:
        s_pred, s_gt = _compute_skeletons_once(y_pred_bin, y_true_bin)
        if mask is not None:
            mask_bin_full = (mask > 0).astype(bool)
            s_pred_masked = _skeletonize_3d(y_pred_bin & mask_bin_full)
            s_gt_masked = _skeletonize_3d(y_true_bin & mask_bin_full)
        else:
            s_pred_masked = s_pred
            s_gt_masked = s_gt

    # ── MATRICE DE CONFUSION (toujours calculée : quasi gratuite, et
    # gt_vessels/pred_vessels sont utilisés indépendamment de la sélection) ──
    cm = confusion_matrix(y_pred_bin, y_true_bin, mask)
    tp, fp, fn, tn = cm['tp'], cm['fp'], cm['fn'], cm['tn']

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    dice_val = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator > 0:
        mcc_val = (tp * tn - fp * fn) / np.sqrt(float(denominator))
    else:
        mcc_val = 0.0

    # ── ROC / PR : seulement si demandés ──
    if selected & {"roc", "pr"}:
        y_pred_proba = y_pred.ravel() if y_pred.dtype != bool else y_pred.astype(float).ravel()
        y_true_flat = y_true_bin.ravel()

        if mask is not None:
            mask_bool = mask > 0 if mask.dtype != bool else mask
            y_pred_proba = y_pred_proba[mask_bool.ravel()]
            y_true_flat = y_true_flat[mask_bool.ravel()]

        try:
            roc_val = roc_auc_score(y_true_flat, y_pred_proba) if "roc"in selected else 0.0
        except Exception:
            roc_val = 0.0

        try:
            pr_val = average_precision_score(y_true_flat, y_pred_proba) if "pr"in selected else 0.0
        except Exception:
            pr_val = 0.0
    else:
        roc_val = 0.0
        pr_val = 0.0

    # ── clDice : seulement si demandé ──
    if "cldice"in selected:
        cldice_val = cldice(y_pred_bin, y_true_bin, mask, compute_background=False,
                            _s_pred=s_pred_masked, _s_gt=s_gt_masked)
    else:
        cldice_val = 0.0

    # ── Composantes connexes : seulement si demandé ──
    if "components"in selected:
        conn_metrics = connected_components_metrics(y_pred_bin, y_true_bin, mask,
                                               _sk_gt=s_gt_masked)
    else:
        conn_metrics = _EMPTY_CONN_METRICS

    # ── Bifurcations : seulement si demandé (et pas skip_bifurcation) ──
    if "bifurcation"in selected and not skip_bifurcation:
        bif_metrics = bifurcation_detection_rate(y_pred_bin, y_true_bin, mask,
                                        _s_pred=s_pred_masked, _s_gt=s_gt_masked)
    else:
        bif_metrics = _EMPTY_BIF_METRICS

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'gt_vessels': cm['gt_vessels'],
        'pred_vessels': cm['pred_vessels'],
        'dice': round(dice_val, 4),
        'mcc': round(mcc_val, 4),
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4),
        'precision': round(precision, 4),
        'accuracy': round(accuracy, 4),
        'roc': round(roc_val, 4),
        'pr': round(pr_val, 4),
        'cldice': round(cldice_val, 4),
        'n_components_pred': conn_metrics['n_components_pred'],
        'n_components_gt': conn_metrics['n_components_gt'],
        'components_ratio': round(conn_metrics['components_ratio'], 4),
        'excess_components': conn_metrics['excess_components'],
        'missing_components': conn_metrics['missing_components'],
        'pred_small_components': conn_metrics['pred_small_components'],
        'pred_medium_components': conn_metrics['pred_medium_components'],
        'pred_large_components': conn_metrics['pred_large_components'],
        'gt_small_components': conn_metrics['gt_small_components'],
        'gt_medium_components': conn_metrics['gt_medium_components'],
        'gt_large_components': conn_metrics['gt_large_components'],
        'largest_component_pred': conn_metrics['largest_component_pred'],
        'largest_component_gt': conn_metrics['largest_component_gt'],
        'largest_ratio': round(conn_metrics['largest_ratio'], 4),
        'largest_gt_recall': round(conn_metrics['largest_gt_recall'], 4),
        'largest_component_overlap': round(conn_metrics['largest_component_overlap'], 4),
        'fragmentation_ratio': round(conn_metrics['fragmentation_ratio'], 4),
        'skeleton_component_connectivity': round(conn_metrics['skeleton_component_connectivity'], 4),
        'threshold': threshold,
        'bifurcation_detection_rate': bif_metrics['bifurcation_detection_rate'],
        'bifurcation_precision': bif_metrics['bifurcation_precision'],
        'bifurcation_f1': bif_metrics['bifurcation_f1'],
        'n_bifurcations_gt': bif_metrics['n_bifurcations_gt'],
        'n_bifurcations_pred': bif_metrics['n_bifurcations_pred'],
        'n_bifurcations_detected': bif_metrics['n_bifurcations_detected'],
        'bifurcation_tp': bif_metrics['bifurcation_tp'],
        'bifurcation_fp': bif_metrics['bifurcation_fp'],
        'bifurcation_fn': bif_metrics['bifurcation_fn'],
        '_metrics_computed': sorted(selected),
    }


def enhanced_stats(data_enhanced: ndarray) -> dict:
    arr = data_enhanced.ravel()
    unique = np.unique(arr)

    return {
        'min': float(arr.min()),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'vessel_ratio': float(np.mean(arr > 0.5)),
        'is_binary': bool(np.array_equal(unique, [0, 1]) or np.array_equal(unique, [0.0, 1.0])),
    }


def print_detailed_metrics(y_pred: ndarray, y_true: ndarray, mask: ndarray = None,
                            name: str = "", logger=None) -> dict:
    """
    Affiche toutes les métriques de manière lisible.
    """
    metrics = detailed_metrics(y_pred, y_true, mask)

    lines = [
        f"\n{'='*60}",
        f"MÉTRIQUES DÉTAILLÉES - {name.upper() if name else 'RESULTATS'}",
        f"{'='*60}",
        f"MATRICE DE CONFUSION",
        f"TP: {metrics['tp']:>12,d} FP: {metrics['fp']:>12,d}",
        f"FN: {metrics['fn']:>12,d} TN: {metrics['tn']:>12,d}",
        f"GT vessels: {metrics['gt_vessels']:>12,d} Pred vessels: {metrics['pred_vessels']:>12,d}",
        f"",
        f"MÉTRIQUES CLASSIQUES",
        f"Dice: {metrics['dice']:.4f}",
        f"MCC: {metrics['mcc']:.4f}",
        f"Sensitivity: {metrics['sensitivity']:.4f}",
        f"Specificity: {metrics['specificity']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"",
        f"MÉTRIQUE TOPOLOGIQUE (clDice)",
        f"clDice: {metrics['cldice']:.4f}",
        f"",
        f"BIFURCATIONS",
        f"BDR (recall): {metrics['bifurcation_detection_rate']:.4f}",
        f"Precision: {metrics['bifurcation_precision']:.4f}",
        f"F1: {metrics['bifurcation_f1']:.4f}",
        f"GT: {metrics['n_bifurcations_gt']} Pred: {metrics['n_bifurcations_pred']}",
        f"TP: {metrics['bifurcation_tp']} FP: {metrics['bifurcation_fp']} FN: {metrics['bifurcation_fn']}",
        f"",
        f"COMPOSANTES CONNEXES (continuité)",
        f"Composantes GT: {metrics['n_components_gt']}",
        f"Composantes pred: {metrics['n_components_pred']}",
        f"Ratio (pred/gt): {metrics['components_ratio']:.2f}",
        f"Fragmentation: +{metrics['excess_components']} composantes",
        f"Plus grande GT: {metrics['largest_component_gt']:,} voxels",
        f"Plus grande pred: {metrics['largest_component_pred']:,} voxels",
        f"Ratio plus grande: {metrics['largest_ratio']:.2f}",
        f"Recall tronc GT: {metrics['largest_gt_recall']:.4f}",
        f"Overlap tronc: {metrics['largest_component_overlap']:.4f}",
        f"Fragmentation ratio: {metrics['fragmentation_ratio']:.2f}",
        f"Distribution GT: small={metrics['gt_small_components']}, "
        f"medium={metrics['gt_medium_components']}, large={metrics['gt_large_components']}",
        f"Distribution pred: small={metrics['pred_small_components']}, "
        f"medium={metrics['pred_medium_components']}, large={metrics['pred_large_components']}",
        f"{'='*60}"
    ]

    for line in lines:
        if logger:
            logger.info(line)
        else:
            print(line)

    return metrics