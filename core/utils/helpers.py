
from time import perf_counter
import numpy as np
from typing import Optional, Tuple



def compute_time(function, *args, **kargs):
    start = perf_counter()
    function(*args, **kargs)
    end = perf_counter()
    return end - start


def normalize_data(data: np.ndarray):
    data_min, data_max = np.min(data), np.max(data)
    if data_max > data_min:  # !div0
        data = (data - data_min) / (data_max - data_min)
    else:
        data = np.zeros_like(data)  # uniform image
    
    return data


def crop_data(data: np.ndarray, target_shape: Tuple[Optional[int], ...]) -> np.ndarray:
    ndim = data.ndim
    if len(target_shape) != ndim:
        raise ValueError(f"Croping failed. data and target_shape must have the same dimension. Current dimensions: {ndim} and {len(target_shape)}.")

    target_shape = [
        data.shape[i] if (target_shape[i] is None or target_shape[i] > data.shape[i]) else target_shape[i]
        for i in range(ndim)
    ]

    slices = tuple(
        slice((s - t) // 2, (s - t) // 2 + t)
        for s, t in zip(data.shape, target_shape)
    )

    return data[slices]


def create_error_map(ground_truth: np.ndarray, segmented: np.ndarray) -> np.ndarray:

    if ground_truth.ndim == 2:
        height, width = ground_truth.shape
        error_map = np.zeros((height, width, 3), dtype=np.uint8)

        true_positive =  (ground_truth == 1) & (segmented == 1)
        true_negative =  (ground_truth == 0) & (segmented == 0)
        false_negative = (ground_truth == 1) & (segmented == 0)
        false_positive = (ground_truth == 0) & (segmented == 1)

        error_map[true_positive | true_negative] = ground_truth[true_positive | true_negative][:, None] * 255
        error_map[false_negative] = [255, 100, 100]
        error_map[false_positive] = [100, 100, 255]

        return error_map

    elif ground_truth.ndim == 3:
        depth, height, width = ground_truth.shape
        error_map = np.zeros((depth, height, width, 3), dtype=np.uint8)

        true_positive = (ground_truth == 1) & (segmented == 1)
        true_negative = (ground_truth == 0) & (segmented == 0)
        false_negative = (ground_truth == 1) & (segmented == 0)
        false_positive = (ground_truth == 0) & (segmented == 1)

        error_map[true_positive | true_negative] = ground_truth[true_positive | true_negative][..., None] * 255
        error_map[false_negative] = [255, 100, 100]
        error_map[false_positive] = [100, 100, 255]

        return error_map
