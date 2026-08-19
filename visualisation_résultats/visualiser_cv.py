import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys
import os

# Reproductibilité globale
SEED = 42
np.random.seed(SEED)

# Add the project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CONFIGURATION

BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "outputs/benchmark/jerman/bullitt_enhancer_jerman_2026-07-11_14-25-54/results"
IMAGES_DIR = BASE_DIR / "data/bullitt/images"
LABELS_DIR = BASE_DIR / "data/bullitt/labels"

# Dossier de sortie
OUTPUT_DIR = BASE_DIR / "outputs/benchmark/jerman/bullitt_enhancer_jerman_2026-07-11_14-25-54/overview/visualisation_vesselness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Opérateurs
OPERATORS = ['default', 'gaussian', 'farid', 'cubic', 'trigonometric', 'catmull', 'bspline', 'bezier','scharr']

# Patients à traiter (01 à 21)
PATIENTS = [f"{i:02d}"for i in range(1, 34)]

# Paramètres d'affichage
ALPHA_OVERLAY = 0.65
DPI = 200

# Couleurs pour les overlays
COLORS = {
    'tp': [1, 1, 0], # Jaune - True Positive
    'fn': [0, 1, 0], # Vert - False Negative
    'fp': [1, 0, 0], # Rouge - False Positive
}

# FONCTIONS PRINCIPALES

def compute_dice(seg_bin, gt_bin):
    """Calcule le coefficient de Dice entre deux masques binaires"""
    intersection = np.logical_and(seg_bin, gt_bin).sum()
    if intersection == 0:
        return 0.0
    return 2.0 * intersection / (seg_bin.sum() + gt_bin.sum())

def compute_metrics(pred_bin, gt_bin):
    """Calcule TP, FP, FN, TN"""
    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()
    tn = np.logical_and(~pred_bin, ~gt_bin).sum()
    return tp, fp, fn, tn

def format_threshold_info(op_data):
    """
    Formate l'information du seuil pour l'affichage
    """
    threshold = getattr(op_data, 'threshold', None)
    
    if threshold is None:
        return "N/A"
    
    if isinstance(threshold, (tuple, list, np.ndarray)):
        threshold_arr = np.array(threshold).flatten()
        if len(threshold_arr) == 2:
            return f"low={threshold_arr[0]:.3f}, high={threshold_arr[1]:.3f}"
        elif len(threshold_arr) == 1:
            return f"th={threshold_arr[0]:.3f}"
        else:
            return f"multi={threshold_arr}"
    
    if isinstance(threshold, (int, float)):
        return f"th={float(threshold):.3f}"
    
    return str(threshold)

def load_patient_data(patient_id):
    """Charge toutes les données pour un patient"""
    
    # Image originale

    image_file = IMAGES_DIR / f"patient_{patient_id}_images.nii.gz"
    if not image_file.exists():
        raise FileNotFoundError(f"Image non trouvée: {image_file}")
    
    raw_img = nib.load(image_file)
    raw_data = raw_img.get_fdata()
    raw_norm = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min() + 1e-8)
    
    # Ground truth
    label_file = LABELS_DIR / f"patient_{patient_id}_label.nii.gz"
    if not label_file.exists():
        raise FileNotFoundError(f"GT non trouvé: {label_file}")
    
    gt_img = nib.load(label_file)
    gt_data = gt_img.get_fdata() > 0.5
    
    # Résultats
    result_file = RESULTS_DIR / f"results_patient_{patient_id}_images.nii"
    if not result_file.exists():
        raise FileNotFoundError(f"Résultats non trouvés: {result_file}")
    
    with open(result_file, 'rb') as f:
        results = pickle.load(f)
    
    return raw_norm, gt_data, results

def get_best_slice(gt_data):
    """Trouve la coupe avec le plus de voxels GT"""
    gt_per_slice = gt_data.sum(axis=(0, 1))
    margin = max(1, len(gt_per_slice) // 20)
    gt_per_slice[:margin] = 0
    gt_per_slice[-margin:] = 0
    best_z = np.argmax(gt_per_slice) if gt_per_slice.sum() > 0 else gt_data.shape[2] // 2
    return best_z

def process_patient(patient_id):
    """Génère la figure pour un patient"""
    
    print(f"\n Patient {patient_id}")
    
    # Chargement des données
    raw_norm, gt_data, results = load_patient_data(patient_id)
    best_z = get_best_slice(gt_data)
    print(f"Coupe: z={best_z}, GT: {gt_data.sum():,} voxels")
    
    # Création de la figure
    n_rows = len(OPERATORS) + 1
    n_cols = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), dpi=DPI)
    fig.suptitle(f'Patient {patient_id} - Comparaison des 8 opérateurs (coupe z={best_z})',
                 fontsize=18, fontweight='bold')
    
    # LIGNE 0: GROUND TRUTH
    
    # GT - Originale
    axes[0, 0].imshow(raw_norm[:, :, best_z], cmap='gray')
    axes[0, 0].set_title('Image originale', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # GT - Information
    axes[0, 1].text(0.5, 0.5, 'GROUND TRUTH\n(Référence)', ha='center', va='center',
                    transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # GT - Masque
    gt_slice = gt_data[:, :, best_z]
    axes[0, 2].imshow(gt_slice, cmap='gray')
    axes[0, 2].set_title(f'Masque GT\n{gt_slice.sum():,} voxels', fontsize=12)
    axes[0, 2].axis('off')
    
    # GT - Overlay
    display_gt = np.stack([raw_norm[:, :, best_z]] * 3, axis=-1)
    display_gt[gt_slice] = [0, 1, 0] # Vert pour GT
    axes[0, 3].imshow(np.clip(display_gt, 0, 1), origin='lower')
    axes[0, 3].set_title('GT overlay (vert)', fontsize=12)
    axes[0, 3].axis('off')
    
    # LIGNES 1-8: OPÉRATEURS
    
    derivators = results.get('derivator', results)
    
    for idx, op_name in enumerate(OPERATORS):
        row = idx + 1
        
        if op_name not in derivators:
            for col in range(4):
                axes[row, col].text(0.5, 0.5, f'{op_name}\nNon disponible',
                                   ha='center', va='center', transform=axes[row, col].transAxes,
                                   fontsize=12)
                axes[row, col].axis('off')
            continue
        
        op_data = derivators[op_name]
        
        # Extraire l'enhancement (vesselness)
        enh_data = getattr(op_data, 'data_enhanced', getattr(op_data, 'enhanced', None))
        
        # Extraire la segmentation - DÉJÀ BINAIRE !
        seg_data = getattr(op_data, 'data_segmented', getattr(op_data, 'segmented', None))
        
        # CORRECTION CRUCIALE : Utiliser directement la segmentation binaire
        if seg_data is not None:
            # seg_data est déjà binaire (0/1), on le caste en bool
            pred_bin = seg_data.astype(bool)
        else:
            pred_bin = None
        
        # Calculer les métriques
        if pred_bin is not None:
            dice = compute_dice(pred_bin, gt_data)
            tp, fp, fn, tn = compute_metrics(pred_bin, gt_data)
        else:
            dice = 0.0
            tp, fp, fn, tn = 0, 0, 0, 0
        
        # Récupérer clDice
        cldice = getattr(op_data, 'cldice_score', getattr(op_data, 'cldice', 0.0))
        
        # Formater l'information du seuil pour affichage
        threshold_info = format_threshold_info(op_data)
        
        # --- Colonne 0: Image + Nom opérateur ---
        axes[row, 0].imshow(raw_norm[:, :, best_z], cmap='gray')
        axes[row, 0].axis('off')
        axes[row, 0].text(0.5, 0.05, op_name.upper(), transform=axes[row, 0].transAxes,
                          fontsize=14, fontweight='bold', ha='center', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # --- Colonne 1: Vesselness (enhancement) ---
        if enh_data is not None:
            enh_slice = enh_data[:, :, best_z]
            enh_min, enh_max = enh_slice.min(), enh_slice.max()
            if enh_max > enh_min:
                enh_norm = (enh_slice - enh_min) / (enh_max - enh_min + 1e-8)
            else:
                enh_norm = enh_slice
            axes[row, 1].imshow(enh_norm, cmap='hot')
            axes[row, 1].set_title('Vesselness', fontsize=11)
        else:
            axes[row, 1].text(0.5, 0.5, 'Non dispo', ha='center', va='center',
                             transform=axes[row, 1].transAxes, fontsize=12)
        axes[row, 1].axis('off')
        
        # --- Colonne 2: Masque final avec Dice ---
        if pred_bin is not None:
            pred_slice = pred_bin[:, :, best_z]
            axes[row, 2].imshow(pred_slice, cmap='gray')
            
            # Afficher Dice avec code couleur
            dice_color = 'green'if dice > 0.5 else 'orange'if dice > 0.2 else 'red'
            axes[row, 2].set_title(f'Dice = {dice:.4f}', fontsize=14, fontweight='bold', color=dice_color)
            
            # Informations supplémentaires
            info_text = f'voxels: {pred_slice.sum():,}'
            if threshold_info != "N/A":
                info_text += f'\n{threshold_info}'
            
            axes[row, 2].text(0.5, 0.95, info_text, transform=axes[row, 2].transAxes,
                             fontsize=8, ha='center', va='top', color='white',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        else:
            axes[row, 2].text(0.5, 0.5, 'Non dispo', ha='center', va='center',
                             transform=axes[row, 2].transAxes, fontsize=12)
        axes[row, 2].axis('off')
        
        # --- Colonne 3: Overlay avec métriques ---
        display_overlay = np.stack([raw_norm[:, :, best_z]] * 3, axis=-1)
        
        if pred_bin is not None:
            pred_slice = pred_bin[:, :, best_z]
            gt_slice = gt_data[:, :, best_z]
            
            # Créer les masques
            tp_mask = np.logical_and(gt_slice, pred_slice)
            fn_mask = np.logical_and(gt_slice, ~pred_slice)
            fp_mask = np.logical_and(~gt_slice, pred_slice)
            
            # Appliquer les couleurs
            display_overlay[tp_mask] = COLORS['tp'] # Jaune
            display_overlay[fn_mask] = COLORS['fn'] # Vert
            display_overlay[fp_mask] = COLORS['fp'] # Rouge
        
        axes[row, 3].imshow(np.clip(display_overlay, 0, 1), origin='lower')
        
        # Afficher les métriques détaillées
        if pred_bin is not None:
            metric_text = f'TP={tp:,} FP={fp:,}\nFN={fn:,} clDice={cldice:.3f}'
            axes[row, 3].text(0.05, 0.95, metric_text, transform=axes[row, 3].transAxes,
                             fontsize=9, ha='left', va='top', color='white',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        axes[row, 3].set_title('Overlay', fontsize=11)
        axes[row, 3].axis('off')
        
        # Log pour suivi
        print(f"{op_name.upper()}: Dice={dice:.4f}, clDice={cldice:.3f}, "
              f"TP={tp:,}, FP={fp:,}, FN={fn:,}, {threshold_info}")
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='GT seul (FN)'),
        Patch(facecolor='red', alpha=0.7, label='Prédiction seule (FP)'),
        Patch(facecolor='yellow', alpha=0.7, label='Overlap (TP)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.08)
    
    # Sauvegarde
    output_file = OUTPUT_DIR / f"patient_{patient_id}_comparison_operators.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n Sauvegardé: {output_file.name}")
    return True

# BOUCLE SUR TOUS LES PATIENTS

print("\n"+ "="*60)
print("GÉNÉRATION DES FIGURES POUR TOUS LES PATIENTS")
print("="*60)
print(f"Dossier de sortie: {OUTPUT_DIR}")
print("="*60)

success_count = 0
error_count = 0
errors_list = []

for patient_id in PATIENTS:
    try:
        if process_patient(patient_id):
            success_count += 1
    except Exception as e:
        print(f"Patient {patient_id}: {e}")
        errors_list.append((patient_id, str(e)))
        error_count += 1
        import traceback
        traceback.print_exc()

print("\n"+ "="*60)
print("RÉSUMÉ")
print("="*60)
print(f"Patients traités avec succès: {success_count}/20")
print(f"Erreurs: {error_count}")
if errors_list:
    print("\nDétail des erreurs:")
    for pid, err in errors_list:
        print(f"- Patient {pid}: {err[:100]}...")
print(f"\n Dossier de sortie: {OUTPUT_DIR}")
print("="*60)
print("\n TRAITEMENT TERMINÉ")