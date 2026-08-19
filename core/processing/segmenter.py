import numpy as np
from numpy import ndarray
from typing import Literal, Optional, Tuple
from sklearn.metrics import precision_recall_curve

from core.utils.helpers import normalize_data

import logging

logger = logging.getLogger(__name__)


class Segmenter:

    def __init__(self):
        self.selector = {
            'thresholding': self.thresholding,
        }

    # Sélection de la fonction de segmentation

    def select_segmentation_function(
        self,
        method: Literal['thresholding'],
    ):
        if method not in self.selector:
            raise ValueError(f"Unknown segmentation method: {method}")
        return self.selector[method]

    # Pré-calcul GT/mask à réutiliser sur un grid search

    @staticmethod
    def precompute_gt_cache(
        ground_truth: ndarray,
        mask: Optional[ndarray] = None,
    ) -> dict:
        gt_binary = (ground_truth > 0).astype(np.uint8)

        if mask is not None:
            mask_binary = (mask > 0).astype(bool)
            gt_m = gt_binary[mask_binary]
        else:
            mask_binary = None
            gt_m = gt_binary.ravel()

        return {"gt_m": gt_m, "mask_binary": mask_binary}

    # Single threshold segmentation principale

    def thresholding(
        self,
        data: ndarray,
        threshold: Optional[float] = None,
        ground_truth: Optional[ndarray] = None,
        mask: Optional[ndarray] = None,
        gt_cache: Optional[dict] = None,
    ) -> Tuple[ndarray, float]:
        # --- Normalisation unique -------------------------------------------
        data_normalized = normalize_data(data)

        return self._segment_single(
            data_normalized=data_normalized,
            threshold=threshold,
            ground_truth=ground_truth,
            mask=mask,
            gt_cache=gt_cache,
        )

    # Mode simple (seuil unique)

    def _segment_single(
        self,
        data_normalized: ndarray,
        threshold: Optional[float],
        ground_truth: Optional[ndarray],
        mask: Optional[ndarray],
        gt_cache: Optional[dict] = None,
    ) -> Tuple[ndarray, float]:
        """
        Segmentation avec un seuil unique.
        """
        # Optimisation du seuil
        if threshold is not None:
            final_threshold = float(threshold)
        elif ground_truth is not None:
            final_threshold = self._find_best_threshold_f1(
                data_normalized, ground_truth, mask, gt_cache=gt_cache,
            )
        else:
            final_threshold = 0.5

        # Application du seuil
        data_segmented = (data_normalized > final_threshold).astype(np.uint8)

        if mask is not None:
            mask_binary = (
                gt_cache["mask_binary"]
                if gt_cache is not None and gt_cache.get("mask_binary") is not None
                else (mask > 0)
            )
            data_segmented = data_segmented * mask_binary.astype(np.uint8)

        return data_segmented, final_threshold

    # Single threshold helpers

    def _find_best_threshold_f1(
        self,
        data_normalized: ndarray,
        ground_truth: ndarray,
        mask: Optional[ndarray],
        gt_cache: Optional[dict] = None,
    ) -> float:
        """Seuil simple qui maximise le F1."""
        if gt_cache is not None:
            gt_m = gt_cache["gt_m"]
            mask_binary = gt_cache.get("mask_binary")
        else:
            gt_binary = (ground_truth > 0).astype(np.uint8)
            if mask is not None:
                mask_binary = (mask > 0).astype(bool)
                gt_m = gt_binary[mask_binary]
            else:
                mask_binary = None
                gt_m = gt_binary.ravel()

        if mask_binary is not None:
            data_m = data_normalized[mask_binary]
        else:
            data_m = data_normalized.ravel()

        if len(np.unique(gt_m)) < 2:
            return 0.5

        precision, recall, thresholds = precision_recall_curve(gt_m, data_m)
        f1_scores = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-8)
        )

        if len(f1_scores) > 0:
            return float(thresholds[np.argmax(f1_scores)])
        return 0.5