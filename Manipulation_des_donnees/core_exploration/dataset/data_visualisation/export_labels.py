import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk
from Manipulation_des_donnees.core_exploration.dataset.dicom_loader import PatientVolume

def save_vessel_masks_as_nifti(pv: PatientVolume, output_dir: Path):
    """Sauvegarde chaque masque vasculaire individuel + fusion + FOIE au format NIfTI."""
    patient_dir = output_dir / f"patient_{pv.patient_id:02d}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    # CT
    if pv.ct_volume is not None:
        ct_img = sitk.GetImageFromArray(pv.ct_volume)
        ct_img.SetSpacing(pv.spacing)
        sitk.WriteImage(ct_img, str(patient_dir / "ct.nii.gz"))

   
    if pv.liver_mask is not None:
        liver_img = sitk.GetImageFromArray(pv.liver_mask.astype(np.uint8))
        liver_img.SetSpacing(pv.spacing)
        sitk.WriteImage(liver_img, str(patient_dir / "liver.nii.gz"))
        print(f"Foie exporté pour patient {pv.patient_id:02d}")
    else:
        print(f"️ Pas de masque foie pour patient {pv.patient_id:02d}")

    # Masques vasculaires individuels
    for name, mask in pv.vessel_masks.items():
        mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_img.SetSpacing(pv.spacing)
        sitk.WriteImage(mask_img, str(patient_dir / f"vessel_{name}.nii.gz"))

    # Masque fusionné
    if pv.vessel_mask is not None:
        fused_img = sitk.GetImageFromArray(pv.vessel_mask.astype(np.uint8))
        fused_img.SetSpacing(pv.spacing)
        sitk.WriteImage(fused_img, str(patient_dir / "vessel_fused.nii.gz"))

def save_slice_overviews(pv: PatientVolume, output_dir: Path, num_slices=3):
    """Génère des vues axiales superposant CT + chaque masque vasculaire."""
    vol = pv.ct_volume
    if vol is None:
        return
    z_mid = vol.shape[0] // 2
    indices = np.linspace(z_mid - num_slices//2, z_mid + num_slices//2, num_slices).astype(int)

    patient_dir = output_dir / f"patient_{pv.patient_id:02d}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    for z in indices:
        fig, axes = plt.subplots(1, len(pv.vessel_masks) + 2, figsize=(4*(len(pv.vessel_masks)+2), 4))
        # CT seul
        axes[0].imshow(vol[z], cmap='gray')
        axes[0].set_title(f"CT slice {z}")
        axes[0].axis('off')
        
        # Masque foie (AJOUT)
        if pv.liver_mask is not None:
            axes[1].imshow(vol[z], cmap='gray')
            axes[1].imshow(pv.liver_mask[z], alpha=0.4, cmap='Greens')
            axes[1].set_title("Liver")
            axes[1].axis('off')
            start_idx = 2
        else:
            start_idx = 1
        
        # Masques vasculaires
        for idx, (name, mask) in enumerate(pv.vessel_masks.items(), start=start_idx):
            axes[idx].imshow(vol[z], cmap='gray')
            axes[idx].imshow(mask[z], alpha=0.4, cmap='Reds')
            axes[idx].set_title(name)
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(patient_dir / f"overview_slice_{z}.png", dpi=150)
        plt.close()