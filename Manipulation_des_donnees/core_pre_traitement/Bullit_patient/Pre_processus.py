import numpy as np
import SimpleITK as sitk


class TissueProcessor:

    @staticmethod
    def finest_spacing(image: sitk.Image) -> tuple[float, float, float]:
        finest = min(image.GetSpacing())
        return (finest, finest, finest)

    @staticmethod
    def compute_bbox(mask_image: sitk.Image, margin_px: int = 15):

        arr = sitk.GetArrayFromImage(mask_image)  # ordre (z, y, x)
        coords = np.argwhere(arr > 0)
        if coords.size == 0:
            raise ValueError("Masque vide : impossible de calculer la bounding box.")

        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        shape = arr.shape  # (z, y, x)

        zmin = max(int(zmin) - margin_px, 0)
        ymin = max(int(ymin) - margin_px, 0)
        xmin = max(int(xmin) - margin_px, 0)
        zmax = min(int(zmax) + margin_px, shape[0] - 1)
        ymax = min(int(ymax) + margin_px, shape[1] - 1)
        xmax = min(int(xmax) + margin_px, shape[2] - 1)

        return (xmin, ymin, zmin), (xmax, ymax, zmax)

    @staticmethod
    def crop_to_bbox(image: sitk.Image, bbox_min: tuple, bbox_max: tuple) -> sitk.Image:
        """Recadre `image` (CT/MRA, label ou masque) sur la bbox (x, y, z) donnée."""
        size = [int(bbox_max[i] - bbox_min[i] + 1) for i in range(3)]
        index = [int(v) for v in bbox_min]
        return sitk.RegionOfInterest(image, size=size, index=index)


    
    @staticmethod
    def normalize_minmax(image: sitk.Image) -> sitk.Image:
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if (mx - mn) < 1e-8:
            norm = np.zeros_like(arr)
        else:
            norm = (arr - mn) / (mx - mn)
        out = sitk.GetImageFromArray(norm)
        out.CopyInformation(image)
        return out
    


    @staticmethod
    def apply_mask(image: sitk.Image, mask_image: sitk.Image) -> sitk.Image:
        arr = sitk.GetArrayFromImage(image)
        marr = sitk.GetArrayFromImage(mask_image) > 0
        masked = arr * marr
        out = sitk.GetImageFromArray(masked.astype(arr.dtype))
        out.CopyInformation(image)
        return out