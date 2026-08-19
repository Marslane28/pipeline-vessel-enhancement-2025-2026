from functools import lru_cache


@lru_cache(maxsize=1)
def is_gpu_available_available():
    import cupy as cp
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def gpu_hessian_matrix_eigvals(H_elements_gpu: tuple):
    import cupy as cp
    from cupy.linalg import eigh

    hessian_elements_gpu = tuple(cp.asarray(e) for e in H_elements_gpu)

    if len(hessian_elements_gpu) == 3:  # 2D
        Hxx, Hxy, Hyy = hessian_elements_gpu  # shapes: (H, W)

        H = cp.stack([
            cp.stack([Hxx, Hxy], axis=-1),
            cp.stack([Hxy, Hyy], axis=-1)
        ], axis=-2)  # shape: (H, W, 2, 2)

    elif len(hessian_elements_gpu) == 6:  # 3D
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = hessian_elements_gpu  # shapes: (D, H, W)

        H = cp.stack([
            cp.stack([Hxx, Hxy, Hxz], axis=-1),
            cp.stack([Hxy, Hyy, Hyz], axis=-1),
            cp.stack([Hxz, Hyz, Hzz], axis=-1)
        ], axis=-2)  # shape: (D, H, W, 3, 3)

    else:
        raise ValueError("Hessian must have 3 elements (2D) or 6 elements (3D)")

    eigvals = cp.linalg.eigh(H)[0]  # shape: (D, H, W, 3) or (H, W, 2)

    return cp.sort(eigvals, axis=-1)