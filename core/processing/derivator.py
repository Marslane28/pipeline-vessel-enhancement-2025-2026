from typing import Literal, Callable, Tuple, Optional
from numpy import ndarray
import numpy as np
import time


from core.utils.gpu import is_gpu_available_available

class Derivator:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and is_gpu_available_available()
        self.selector = {
            'default': self.default,
            'gaussian': self.gaussian,
            'farid': self.farid,
            'cubic': self.cubic,
            'trigonometric': self.trigonometric,
            'catmull': self.catmull,
            'bspline': self.bspline,
            'bezier': self.bezier,
            'scharr': self.scharr,
        }
        # Dernier temps de calcul du Hessien (secondes), rempli par compute_hessian
        # et par default/gaussian. None tant qu'aucun appel n'a été fait.
        self.last_hessian_time: Optional[float] = None
    def select_hessian_function(
        self,
        method: Literal['default', 'gaussian', 'farid', 'cubic', 'trigonometric', 'catmull', 'bspline', 'bezier', 'scharr']
    ) -> Callable[..., list[np.ndarray]]:
        if method not in self.selector:
            raise ValueError(f"Unknown differentiation method: {method}. Valid methods: {list(self.selector.keys())}")
        return self.selector[method]

    def compute_hessian(
        self,
        data: ndarray,
        filters: list[list],
        sigma: float = 1.0,
        mode: str = 'reflect',
        smoothing_filters: Optional[list[list]] = None,
    ) -> Tuple[np.ndarray]:

        if self.use_gpu:
            import cupy as cp
            from cupyx.scipy.ndimage import convolve1d as gpu_convolve1d
            from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
            xp = cp
            convolve1d = gpu_convolve1d
            gaussian_filter = gpu_gaussian_filter
        else:
            from scipy.ndimage import gaussian_filter as cpu_gaussian_filter
            from scipy.ndimage import convolve1d as cpu_convolve1d
            xp = np
            convolve1d = cpu_convolve1d
            gaussian_filter = cpu_gaussian_filter

        p = xp.array(filters[0], dtype=xp.float32)
        d1 = xp.array(filters[1], dtype=xp.float32)
        d2 = xp.array(filters[2], dtype=xp.float32)

        if smoothing_filters is not None:
            p1 = xp.array(smoothing_filters[0], dtype=xp.float32)
            p2 = xp.array(smoothing_filters[1], dtype=xp.float32)
        else:
            p1 = p
            p2 = p

        data = xp.array(data, dtype=xp.float32)

        t0 = time.perf_counter()

        data = gaussian_filter(data, sigma=sigma)

        if data.ndim == 2:
            Hxx = convolve1d(convolve1d(data, d2, axis=1, mode=mode), p2, axis=0, mode=mode)
            Hyy = convolve1d(convolve1d(data, p2, axis=1, mode=mode), d2, axis=0, mode=mode)
            Hxy = convolve1d(convolve1d(data, d1, axis=1, mode=mode), d1, axis=0, mode=mode)

            if self.use_gpu:
                xp.cuda.Stream.null.synchronize()
            self.last_hessian_time = time.perf_counter() - t0

            if self.use_gpu:
                return (xp.asnumpy(Hxx), xp.asnumpy(Hxy), xp.asnumpy(Hyy))
            else:
                return (Hxx, Hxy, Hyy)

        elif data.ndim == 3:
            Hxx = convolve1d(convolve1d(convolve1d(data, d2, axis=0, mode=mode), p2, axis=1, mode=mode), p2, axis=2, mode=mode)
            Hyy = convolve1d(convolve1d(convolve1d(data, p2, axis=0, mode=mode), d2, axis=1, mode=mode), p2, axis=2, mode=mode)
            Hzz = convolve1d(convolve1d(convolve1d(data, p2, axis=0, mode=mode), p2, axis=1, mode=mode), d2, axis=2, mode=mode)
            Hxy = convolve1d(convolve1d(convolve1d(data, d1, axis=0, mode=mode), d1, axis=1, mode=mode), p1, axis=2, mode=mode)
            Hyz = convolve1d(convolve1d(convolve1d(data, p1, axis=0, mode=mode), d1, axis=1, mode=mode), d1, axis=2, mode=mode)
            Hzx = convolve1d(convolve1d(convolve1d(data, d1, axis=0, mode=mode), p1, axis=1, mode=mode), d1, axis=2, mode=mode)

            if self.use_gpu:
                xp.cuda.Stream.null.synchronize()
            self.last_hessian_time = time.perf_counter() - t0

            if self.use_gpu:
                return (xp.asnumpy(Hxx), xp.asnumpy(Hxy), xp.asnumpy(Hzx),
                        xp.asnumpy(Hyy), xp.asnumpy(Hyz), xp.asnumpy(Hzz))
            else:
                return (Hxx, Hxy, Hzx, Hyy, Hyz, Hzz)

    def default(self, data: np.ndarray, sigma: int, mode: str = 'reflect', cval: float = 0.0, order: str = 'rc', **unused) -> list[np.ndarray]:
        from skimage.feature import hessian_matrix as cpu_hessian_matrix
        hessian_matrix = cpu_hessian_matrix
        t0 = time.perf_counter()
        result = hessian_matrix(
            image=data, sigma=sigma, mode=mode, cval=cval, order=order,
            use_gaussian_derivatives=False
        )
        self.last_hessian_time = time.perf_counter() - t0
        return result

    def gaussian(self, data: np.ndarray, sigma: int, mode: str = 'reflect', cval: float = 0.0, order: str = 'rc', **unused) -> list[np.ndarray]:
        from skimage.feature import hessian_matrix as cpu_hessian_matrix
        hessian_matrix = cpu_hessian_matrix
        t0 = time.perf_counter()
        result = hessian_matrix(
            image=data, sigma=sigma, mode=mode, cval=cval, order=order,
            use_gaussian_derivatives=True
        )
        self.last_hessian_time = time.perf_counter() - t0
        return result

    def farid(self, data: ndarray, sigma: float = 1.0, mode: str = 'reflect', order: int = 7, **unused):
        if order == 5:
            p  = [0.030320, 0.249724, 0.439911, 0.249724, 0.030320]
            d1 = [-0.104550, -0.292315, 0.0, 0.292315, 0.104550]
            d2 = [0.232905, 0.002668, -0.471147, 0.002668, 0.232905]
        elif order == 7:
            p  = [0.0047, 0.0693, 0.2454, 0.3611, 0.2454, 0.0693, 0.0047]
            d1 = [0.0187, 0.1253, 0.1930, 0.0, -0.1930, -0.1253, -0.0187]
            d2 = [0.0553, 0.1378, -0.0566, -0.2731, -0.0566, 0.1378, 0.0553]
        return self.compute_hessian(data=data, filters=[p, d1, d2], sigma=sigma, mode=mode)

    def cubic(self, data: ndarray, sigma: float = 1.0, mode: str = 'reflect', **unused):
        
        p  = [0.0039, -0.0703, 0.2461, 0.6406, 0.2461, -0.0703, 0.0039]
        d1 = [-0.0026, 0.0938, -0.6797, 0.0, 0.6797, -0.0938, 0.0026]
        d2 = [-0.0312, 0.3125, 0.0312, -0.6250, 0.0312, 0.3125, -0.0312]
        return self.compute_hessian(data=data, filters=[p, d1, d2], sigma=sigma, mode=mode)

    def trigonometric(self, data: ndarray, sigma: float = 1.0, mode: str = 'reflect', **unused):
        p  = [0.0043, -0.0745, 0.2457, 0.6490, 0.2457, -0.0745, 0.0043]
        d1 = [-0.0061, 0.1966, -1.3282, 0.0, 1.3282, -0.1966, 0.0061]
        d2 = [-0.1272, 1.2203, 0.1272, -2.4405, 0.1272, 1.2203, -0.1272]
        return self.compute_hessian(data=data, filters=[p, d1, d2], sigma=sigma, mode=mode)

    def catmull(self, data: ndarray, sigma: float = 1.0, mode: str = 'reflect', **unused):
        p  = [0.0039, -0.0703, 0.2461, 0.6406, 0.2461, -0.0703, 0.0039]
        d1 = [-0.0078, 0.1562, -0.7891, 0.0, 0.7891, -0.1562, 0.0078]
        d2 = [-0.0312, 0.3125, 0.0312, -0.6250, 0.0312, 0.3125, -0.0312]
        return self.compute_hessian(data=data, filters=[p, d1, d2], sigma=sigma, mode=mode)

    def bspline(self, data: ndarray, sigma: float = 1.0, mode: str = 'reflect', **unused):
        
        p  = [0.0004, 0.0200, 0.2496, 0.4601, 0.2496, 0.0200, 0.0004]
        d1 = [-0.0026, -0.0729, -0.3464, 0.0, 0.3464, 0.0729, 0.0026]
        d2 = [0.0104, 0.2292, -0.0104, -0.4583, -0.0104, 0.2292, 0.0104]
        return self.compute_hessian(data=data, filters=[p, d1, d2], sigma=sigma, mode=mode)

    def bezier(self, data: ndarray, sigma: float = 1.0, mode: str = 'reflect', **unused):
        p  = [0.0156, 0.0938, 0.2344, 0.3125, 0.2344, 0.0938, 0.0156]
        d1 = [0.0938, 0.3750, 0.4688, 0.0, -0.4688, -0.3750, -0.0938]
        d2 = [0.3750, 0.7500, -0.3750, -1.5000, -0.3750, 0.7500, 0.3750]
        return self.compute_hessian(data=data, filters=[p, d1, d2], sigma=sigma, mode=mode)

    def scharr(self, data: ndarray, sigma: float = 1.0, mode: str = 'reflect', **unused):
        i1 = [0.01504, 0.23301, 0.50390, 0.23301, 0.01504]
        i2 = [0.01554, 0.23204, 0.50484, 0.23204, 0.01554]
        d1 = [0.06368, 0.37263, 0.0, -0.37263, -0.06368]
        d2 = [0.20786, 0.16854, -0.75282, 0.16854, 0.20786]
        return self.compute_hessian(
            data=data, filters=[i2, d1, d2], sigma=sigma, mode=mode,
            smoothing_filters=[i1, i2],
        )