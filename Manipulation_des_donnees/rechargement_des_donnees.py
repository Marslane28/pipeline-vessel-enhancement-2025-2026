import os
import shutil
from pathlib import Path

def prepare_data(
    source_dir="données_pretraitées",
    target_dir="data/3d-échantillonnées",
    use_copy=False
):
    source = Path(source_dir).resolve()
    images_dir = (Path(target_dir) / "images").resolve()
    labels_dir = (Path(target_dir) / "labels").resolve()
    masks_dir = (Path(target_dir) / "masks").resolve()

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for f in images_dir.glob("*.nii*"): f.unlink()
    for f in labels_dir.glob("*.nii*"): f.unlink()
    for f in masks_dir.glob("*.nii*"): f.unlink()
    patients = sorted(source.glob("patient_*"))
    print(f"{len(patients)} patients trouvés")

    success = 0
    for i, patient in enumerate(patients, start=1):
        idx = f"{i:02d}"
        name = patient.name

        src_img = (patient / f"patient_{idx}_ct_isotropic.nii.gz").resolve()
        src_lbl = (patient / f"patient_{idx}_vessels_gt_isotropic_in_liver.nii.gz").resolve()
        src_mask = (patient / f"patient_{idx}_liver_mask_isotropic.nii.gz").resolve()

        dst_img = images_dir / f"patient_{idx}_images.nii.gz"
        dst_lbl = labels_dir / f"patient_{idx}_label.nii.gz"
        dst_mask = masks_dir / f"patient_{idx}_liver.nii.gz"

        if not src_img.exists() or not src_lbl.exists():
            print(f"{name} - image ou label manquant, ignoré")
            continue

        # Le masque est optionnel on avertit mais on ne bloque pas
        #Vascusynth n'a pas de masque, on peut l'ignorer
        has_mask = src_mask.exists()
        if not has_mask:
            print(f"{name} - masque foie absent (évaluation sans masque)")

        try:
            if use_copy:
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)
                if has_mask:
                    shutil.copy2(src_mask, dst_mask)
            else:
                dst_img.unlink(missing_ok=True)
                dst_lbl.unlink(missing_ok=True)
                dst_mask.unlink(missing_ok=True)
                os.symlink(src_img, dst_img)
                os.symlink(src_lbl, dst_lbl)
                if has_mask:
                    os.symlink(src_mask, dst_mask)

            success += 1
            mask_status = "mask"if has_mask else "no mask"
            print(f"{name} → patient_{idx} [{mask_status}]")

        except Exception as e:
            print(f"{name} {e}")

    print(f"\nSuccès : {success}/{len(patients)}")

if __name__ == "__main__":
    prepare_data()