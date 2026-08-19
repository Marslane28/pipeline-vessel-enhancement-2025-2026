import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk

def save_vessel_masks_as_nifti(patient_volume, output_dir: Path):
    """Sauvegarde chaque masque vasculaire (individuel + fusion) au format NIfTI."""
    patient_dir = output_dir / f"patient_{patient_volume.patient_id:02d}"
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder le CT (optionnel)
    ct_img = sitk.GetImageFromArray(patient_volume.ct_volume)
    ct_img.SetSpacing(patient_volume.spacing)
    sitk.WriteImage(ct_img, str(patient_dir / "ct.nii.gz"))
    
    # Sauvegarder chaque masque
    for name, mask in patient_volume.vessel_masks.items():
        mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_img.SetSpacing(patient_volume.spacing)
        sitk.WriteImage(mask_img, str(patient_dir / f"vessel_{name}.nii.gz"))
    
    # Fusion
    if patient_volume.vessel_mask is not None:
        fused_img = sitk.GetImageFromArray(patient_volume.vessel_mask.astype(np.uint8))
        fused_img.SetSpacing(patient_volume.spacing)
        sitk.WriteImage(fused_img, str(patient_dir / "vessel_fused.nii.gz"))

def save_slice_overviews(patient_volume, output_dir: Path, num_slices=3):
    """Génère des vues axiales (milieu) superposant CT + contours des vaisseaux."""
    vol = patient_volume.ct_volume
    z_mid = vol.shape[0] // 2
    indices = np.linspace(z_mid - num_slices//2, z_mid + num_slices//2, num_slices).astype(int)
    
    patient_dir = output_dir / f"patient_{patient_volume.patient_id:02d}"
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    for i, z in enumerate(indices):
        fig, axes = plt.subplots(1, len(patient_volume.vessel_masks)+1, figsize=(15,4))
        ax_ct = axes[0]
        ax_ct.imshow(vol[z], cmap='gray')
        ax_ct.set_title(f"CT slice {z}")
        
        for (name, mask), ax in zip(patient_volume.vessel_masks.items(), axes[1:]):
            # superposer le contour du masque sur le CT
            ax.imshow(vol[z], cmap='gray')
            ax.imshow(mask[z], alpha=0.4, cmap='Reds')
            ax.set_title(name)
        plt.tight_layout()
        plt.savefig(patient_dir / f"overview_slice_{z}.png", dpi=150)
        plt.close()