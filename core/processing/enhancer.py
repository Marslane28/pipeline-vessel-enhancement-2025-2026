import numpy as np
from numpy import ndarray
import gc
import time
from skimage.filters import frangi as frangi_skimage
from typing import Callable, Optional, Sequence, Literal
from core.processing.derivator import Derivator
from core.utils.gpu import is_gpu_available_available


class Enhancer:

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and is_gpu_available_available()
        self.selector = {
            'frangi': self.frangi,
            'jerman': self.jerman,
            'mfat': self.mfat,
        }
        self.apply_selector = {
            'frangi': self.frangi_apply,
            'jerman': self.jerman_apply,
            'mfat': self.mfat_apply,
        }
        self._s3d_call_count = 0
        self._last_total_hessian_time = 0.0

    def select_enhancement_function(
        self, method: Literal['frangi', 'jerman', 'mfat']
    ) -> Callable[..., ndarray]:
        if method not in self.selector:
            raise ValueError(
                f"Unknown enhancement method: {method}. "
                f"Valid methods : {list(self.selector.keys())}"
            )
        return self.selector[method]

    def select_apply_function(
        self, method: Literal['frangi', 'jerman', 'mfat']
    ) -> Callable[..., ndarray]:
        """Retourne la fonction d'application rapide pour le grid search."""
        if method not in self.apply_selector:
            raise ValueError(
                f"Unknown enhancement method: {method}. "
                f"Valid methods : {list(self.apply_selector.keys())}"
            )
        return self.apply_selector[method]

    def _check_gpu_consistency(self, hessian_function: Optional[Callable]) -> None:
        if hessian_function and hasattr(hessian_function.__self__, 'use_gpu'):
            if hessian_function.__self__.use_gpu != self.use_gpu:
                raise ValueError(
                    f'GPU must be used for both Derivator and Enhancer or none. '
                    f'Current values: Derivator({hessian_function.__self__.use_gpu}), '
                    f'Enhancer({self.use_gpu})'
                )

    def _prepare_common(
        self,
        image: ndarray,
        hessian_function: Optional[Callable],
        black_ridges: Optional[bool],
        negate_image: bool = True,
    ):
        self._check_gpu_consistency(hessian_function)

        if self.use_gpu:
            import cupy as cp
            from core.utils.gpu import gpu_hessian_matrix_eigvals
            xp = cp
            eigvals_function = gpu_hessian_matrix_eigvals
        else:
            from skimage.feature import hessian_matrix_eigvals as cpu_hessian_eigvals
            xp = np
            eigvals_function = cpu_hessian_eigvals

        from skimage.feature import hessian_matrix as cpu_hessian_matrix
        hessian_function = hessian_function if hessian_function else cpu_hessian_matrix

        if negate_image and not black_ridges:
            image = -image

        image = xp.asarray(image)
        image = image.astype(xp.float32, copy=False)

        return xp, eigvals_function, hessian_function, image

    def _iter_scale_eigvals(
        self,
        image: ndarray,
        hessian_function: Callable,
        hessian_params: dict,
        scales: Sequence[float],
        eigvals_function: Callable,
    ):
        derivator_obj = None
        if hasattr(hessian_function, '__self__'):
            derivator_obj = hessian_function.__self__
        total_hessian_time = 0.0

        for scale in scales:
            hessian = hessian_function(image, sigma=scale, **hessian_params)
            if derivator_obj is not None and hasattr(derivator_obj, 'last_hessian_time') and derivator_obj.last_hessian_time is not None:
                total_hessian_time += derivator_obj.last_hessian_time
            eigvals = eigvals_function(hessian)
            yield scale, hessian, eigvals

        if derivator_obj is not None and hasattr(derivator_obj, 'last_hessian_time'):
            derivator_obj.last_hessian_time = total_hessian_time
        self._last_total_hessian_time = total_hessian_time

    # Pre-computation pour grid search optimization

    def precompute_eigenvalues(
        self,
        image: ndarray,
        method: Literal['frangi', 'jerman', 'mfat'],
        scales: Sequence[float],
        black_ridges: bool,
        hessian_params: dict,
        hessian_function: Optional[Callable] = None,
        mask: Optional[ndarray] = None,
    ) -> tuple[list, tuple, any, Optional[ndarray]]:
        """
        Calcule les eigenvalues triés une fois pour toutes.
        """
        xp, eigvals_function, hessian_function, image = self._prepare_common(
            image, hessian_function, black_ridges,
            negate_image=(method == 'frangi')
        )
        
        cached = []
        for scale, hessian, eigvals in self._iter_scale_eigvals(
            image, hessian_function, hessian_params, scales, eigvals_function
        ):
            # Tri par valeur absolue (commun aux trois filtres)
            sorted_eigvals = xp.take_along_axis(
                eigvals, xp.abs(eigvals).argsort(0), axis=0
            )
            cached.append((scale, sorted_eigvals))
            del hessian, eigvals
        mask_arr = xp.asarray(mask, dtype=bool) if mask is not None else None
        
        return cached, image.shape, xp, mask_arr

# Pré-calcul des rapports indépendants de α, β et γ.

    def precompute_frangi_ratios(
        self,
        cached_eigvals: list,
        image_shape: tuple,
        xp: any,
    ) -> list[tuple]:

        is_2d = len(image_shape) == 2
        ratios = []

        for _, eigvals in cached_eigvals:
            if is_2d:
                lambda1 = eigvals[0]
                lambda2 = xp.maximum(eigvals[1], 1e-10)
                r_a = xp.inf
                r_b = xp.abs(lambda1) / lambda2
            else:
                lambda1, lambda2, lambda3 = eigvals[0], eigvals[1], eigvals[2]
                lambda2_pos = xp.maximum(lambda2, 1e-10)
                lambda3_pos = xp.maximum(lambda3, 1e-10)
                r_a = lambda2_pos / lambda3_pos
                r_b = xp.abs(lambda1) / xp.sqrt(lambda2_pos * lambda3_pos)

            s = xp.sqrt((eigvals**2).sum(axis=0))
            ratios.append((r_a, r_b, s))

        return ratios

    # Frangi - version complète

    def frangi(
        self,
        image: ndarray,
        hessian_function: Callable[..., list[ndarray]] = None,
        hessian_params: dict = {'mode': 'reflect', 'cval': 0},
        scales: Optional[Sequence[int]] = range(0, 10, 2),
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: Optional[float] = None,
        black_ridges: Optional[bool] = True,
        skimage: bool = False,
        **kwargs,
    ) -> tuple[ndarray, float]:

        if skimage:
            self._check_gpu_consistency(hessian_function)
            if self.use_gpu:
                raise ValueError('Skimage function can only be used for non GPU processing.')
            return frangi_skimage(
                image, sigmas=scales, alpha=alpha, beta=beta,
                gamma=gamma, black_ridges=black_ridges
            )

        xp, eigvals_function, hessian_function, image = self._prepare_common(
            image, hessian_function, black_ridges, negate_image=True
        )

        filtered_image = xp.zeros_like(image)

        for scale, hessian, eigvals in self._iter_scale_eigvals(
            image, hessian_function, hessian_params, scales, eigvals_function
        ):
            eigvals = xp.take_along_axis(eigvals, xp.abs(eigvals).argsort(0), axis=0)

            if image.ndim == 2:
                lambda1 = eigvals[0]
                lambda2 = xp.maximum(eigvals[1], 1e-10)
                r_a = xp.inf
                r_b = xp.abs(lambda1) / lambda2
            else:
                lambda1, lambda2, lambda3 = eigvals[0], eigvals[1], eigvals[2]
                lambda2_pos = xp.maximum(lambda2, 1e-10)
                lambda3_pos = xp.maximum(lambda3, 1e-10)
                r_a = lambda2_pos / lambda3_pos
                r_b = xp.abs(lambda1) / xp.sqrt(lambda2_pos * lambda3_pos)

            s = xp.sqrt((eigvals**2).sum(axis=0))

            if gamma is None:
                gamma = s.max() / 2 if s.max() != 0 else 1

            vesselness = 1.0 - xp.exp(-(r_a**2) / (2 * alpha**2))
            vesselness *= xp.exp(-(r_b**2) / (2 * beta**2))
            vesselness *= (1.0 - xp.exp(-(s**2) / (2 * gamma**2)))

            filtered_image = xp.maximum(filtered_image, vesselness)

            del hessian, eigvals, vesselness, s
            gc.collect()

        total_hessian_time = self._last_total_hessian_time

        return xp.asnumpy(filtered_image) if self.use_gpu else filtered_image, total_hessian_time

    # Frangi - Apply only (for grid search)

    def frangi_apply(
        self,
        cached_eigvals: list,
        image_shape: tuple,
        xp: any,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: Optional[float] = None,
        cached_ratios: Optional[list] = None,
        **kwargs
    ) -> ndarray:
        """
        Applique UNIQUEMENT la formule Frangi (pas de Hessien).
        Utilisé pour le grid search après pré-calcul des eigenvalues.

        """
        filtered = xp.zeros(image_shape, dtype=xp.float32)

        ratios = (
            cached_ratios
            if cached_ratios is not None
            else self.precompute_frangi_ratios(cached_eigvals, image_shape, xp)
        )
        local_gamma = gamma

        for r_a, r_b, s in ratios:
            if local_gamma is None:
                local_gamma = s.max() / 2 if s.max() != 0 else 1

            vesselness = 1.0 - xp.exp(-(r_a**2) / (2 * alpha**2))
            vesselness *= xp.exp(-(r_b**2) / (2 * beta**2))
            vesselness *= (1.0 - xp.exp(-(s**2) / (2 * local_gamma**2)))

            filtered = xp.maximum(filtered, vesselness)

            del vesselness

        return xp.asnumpy(filtered) if self.use_gpu else filtered

    # =========================================================================
    # Jerman - "Beyond Frangi: an improved multiscale vesselness filter"
    # (Jerman et al., IEEE TMI 2016 / SPIE 2015)
    # lien d'inspiration : https://github.com/JonasLamy/LiverVesselness
    # =========================================================================

    def jerman(
        self,
        image: ndarray,
        hessian_function: Callable[..., list[ndarray]] = None,
        hessian_params: dict = {'mode': 'reflect', 'cval': 0},
        scales: Optional[Sequence[int]] = range(1, 10, 2),
        tau: float = 0.75,
        black_ridges: Optional[bool] = True,
        mask: Optional[ndarray] = None,
        **kwargs,
    ) -> tuple[ndarray, float]:
        """
        Vesselness de Jerman:

            F = 0 si lambda2 <= 0 ou lambda_rho <= 0
            F = 1 si lambda2 >= lambda_rho/2 > 0
            F = lambda2^2 (lambda_rho - lambda2) (3/(lambda2+lambda_rho))^3 sinon

        avec lambda_rho, régularisation de lambda3 :
            lambda_rho = lambda3 si lambda3 > tau * max_x lambda3(x)
            lambda_rho = tau * max_x lambda3(x) si 0 < lambda3 <= tau * max_x lambda3(x)
            lambda_rho = 0 sinon

        Tri des valeurs propres : par VALEUR ABSOLUE (|lambda1| <= |lambda2|
        <= |lambda3|)
        """
        xp, eigvals_function, hessian_function, image = self._prepare_common(
            image, hessian_function, black_ridges, negate_image=False
        )

        mask_arr = xp.asarray(mask, dtype=bool) if mask is not None else None

        filtered_image = xp.zeros_like(image)

        for scale, hessian, eigvals in self._iter_scale_eigvals(
            image, hessian_function, hessian_params, scales, eigvals_function
        ):
            # Tri par VALEUR ABSOLUE.
            eigvals_sorted = xp.take_along_axis(
                eigvals, xp.abs(eigvals).argsort(0), axis=0
            )

            if image.ndim == 2:
                lambda2_raw = eigvals_sorted[1]
                lambda3_raw = eigvals_sorted[1]
            else:
                lambda2_raw = eigvals_sorted[1]
                lambda3_raw = eigvals_sorted[2]

            # --- convention de signe ---
            if not black_ridges:
                lambda2 = -lambda2_raw
                lambda3 = -lambda3_raw
            else:
                lambda2 = lambda2_raw
                lambda3 = lambda3_raw

            # --- lambda_rho : régularisation de lambda3 par le MAXIMUM
            # GLOBAL, restreint au masque si fourni (seul usage du masque
            # dans tout le filtre).
            if mask_arr is not None:
                l3_region = lambda3[mask_arr]
            else:
                l3_region = lambda3.ravel()
            lambda3_max = l3_region.max() if l3_region.size > 0 else xp.array(0.0, dtype=lambda3.dtype)
            lambda3_max = xp.maximum(lambda3_max, xp.array(1e-10, dtype=lambda3.dtype))

            floor_val = tau * lambda3_max
            zero = xp.array(0.0, dtype=lambda3.dtype)
            lambda_rho = xp.where(
                lambda3 <= 0, zero,
                xp.where(lambda3 <= floor_val, floor_val, lambda3)
            )

            denom = (lambda2 + lambda_rho) ** 3
            lambda_scale = xp.max(xp.abs(lambda3))
            safe_floor = 1e-6 * lambda_scale
            denom_safe = xp.where(xp.abs(denom) < safe_floor, safe_floor, denom)
            response = (lambda2 ** 2) * (lambda_rho - lambda2) * 27.0 / denom_safe
            saturated = (lambda2 >= lambda_rho / 2.0) & (lambda_rho > 0)
            response = xp.where(saturated, xp.array(1.0, dtype=response.dtype), response)

            invalid = (lambda2 <= 0) | (lambda_rho <= 0)
            response = xp.where(invalid, zero, response)
            response = xp.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
            filtered_image = xp.maximum(filtered_image, response)

            del hessian, eigvals, eigvals_sorted, lambda2, lambda3, lambda_rho, response
            gc.collect()

        total_hessian_time = self._last_total_hessian_time

        return xp.asnumpy(filtered_image) if self.use_gpu else filtered_image, total_hessian_time

    # Jerman - Uniquement (for grid search)

    def jerman_apply(
        self,
        cached_eigvals: list,
        image_shape: tuple,
        xp: any,
        tau: float = 0.75,
        black_ridges: bool = True,
        mask: Optional[ndarray] = None,
        **kwargs
    ) -> ndarray:
        """
        Applique UNIQUEMENT la formule Jerman (pas de Hessien).
        Utilisé pour le grid search après pré-calcul des eigenvalues.
        """
        filtered = xp.zeros(image_shape, dtype=xp.float32)
        is_2d = len(image_shape) == 2
        mask_arr = xp.asarray(mask, dtype=bool) if mask is not None else None
        
        for _, eigvals in cached_eigvals:
            # eigvals déjà triés par valeur absolue
            if is_2d:
                lambda2_raw = eigvals[1]
                lambda3_raw = eigvals[1]
            else:
                lambda2_raw = eigvals[1]
                lambda3_raw = eigvals[2]
            
            # Convention de signe
            if not black_ridges:
                lambda2 = -lambda2_raw
                lambda3 = -lambda3_raw
            else:
                lambda2 = lambda2_raw
                lambda3 = lambda3_raw
            
            # lambda_rho
            if mask_arr is not None:
                l3_region = lambda3[mask_arr]
            else:
                l3_region = lambda3.ravel()
            lambda3_max = l3_region.max() if l3_region.size > 0 else xp.array(0.0, dtype=lambda3.dtype)
            lambda3_max = xp.maximum(lambda3_max, xp.array(1e-10, dtype=lambda3.dtype))
            
            floor_val = tau * lambda3_max
            zero = xp.array(0.0, dtype=lambda3.dtype)
            lambda_rho = xp.where(
                lambda3 <= 0, zero,
                xp.where(lambda3 <= floor_val, floor_val, lambda3)
            )
            
            # Formule Jerman
            denom = (lambda2 + lambda_rho) ** 3
            lambda_scale = xp.max(xp.abs(lambda3))
            safe_floor = 1e-6 * lambda_scale
            denom_safe = xp.where(xp.abs(denom) < safe_floor, safe_floor, denom)
            response = (lambda2 ** 2) * (lambda_rho - lambda2) * 27.0 / denom_safe
            saturated = (lambda2 >= lambda_rho / 2.0) & (lambda_rho > 0)
            response = xp.where(saturated, xp.array(1.0, dtype=response.dtype), response)
            invalid = (lambda2 <= 0) | (lambda_rho <= 0)
            response = xp.where(invalid, zero, response)
            response = xp.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
            
            filtered = xp.maximum(filtered, response)
            del response, lambda2, lambda3, lambda_rho
        
        return xp.asnumpy(filtered) if self.use_gpu else filtered

    # =========================================================================
    # MFAT / PFAT - "2D and 3D Vascular Structures Enhancement via Multiscale
    # Fractional Anisotropy Tensor"(Alhasson, Alharbi & Obara, ECCVW 2018)
    # lien: https://github.com/Haifafh/MFAT/tree/master/3D/lib
    # =========================================================================

    def mfat(
        self,
        image: ndarray,
        hessian_function: Callable[..., list[ndarray]] = None,
        hessian_params: dict = {'mode': 'reflect', 'cval': 0},
        scales: Optional[Sequence[float]] = (0.5, 1.0, 1.5, 2.0),
        tau: float = 0.02,
        tau2: float = 0.2,
        step_size: float = 0.18,
        variant: Literal['eigen', 'proba'] = 'eigen',
        black_ridges: Optional[bool] = True,
        **kwargs,
    ) -> tuple[ndarray, float]:
        """
        Portage EXACT du code MATLAB de référence de Haifa Alhasson
        (github.com/Haifafh/MFAT).
        """
        if not scales:
            raise ValueError("`scales` cannot be empty.")

        xp, eigvals_function, hessian_function, image = self._prepare_common(
            image, hessian_function, black_ridges, negate_image=False
        )

        is_2d = image.ndim == 2
        vesselness = None # accumulateur multi-échelle : l'ordre des scales compte ici

        for scale, hessian, eigvals in self._iter_scale_eigvals(
            image, hessian_function, hessian_params, scales, eigvals_function
        ):
            eigvals_sorted = xp.take_along_axis(eigvals, xp.abs(eigvals).argsort(0), axis=0)

            lambda2 = eigvals_sorted[1]
            lambda3_base = lambda2 if is_2d else eigvals_sorted[2]

            # --- régularisation de lambda3_base : lambda_rho (tau) et lambda_nu (tau2) ---
            l3_region = lambda3_base.ravel()
            lambda3_min = (
                l3_region.min() if l3_region.size > 0 else xp.array(0.0, dtype=lambda3_base.dtype)
            )

            floor_rho = tau * lambda3_min
            floor_nu = tau2 * lambda3_min

            lambda_rho = xp.where(
                (lambda3_base < 0) & (lambda3_base >= floor_rho), floor_rho, lambda3_base
            )
            lambda_nu = xp.where(
                (lambda3_base < 0) & (lambda3_base >= floor_nu), floor_nu, lambda3_base
            )

            # --- F AT_λ (Eq. 7) ou F AT_p (Eq. 9) ---
            with np.errstate(divide='ignore', invalid='ignore'):
                if variant == 'eigen':
                    lambda_md = (xp.abs(lambda2) + xp.abs(lambda_rho) + xp.abs(lambda_nu)) / 3.0
                    num = (
                        (xp.abs(lambda2) - lambda_md) ** 2
                        + (xp.abs(lambda_rho) - lambda_md) ** 2
                        + (xp.abs(lambda_nu) - lambda_md) ** 2
                    )
                    denom = xp.abs(lambda2) ** 2 + xp.abs(lambda_rho) ** 2 + xp.abs(lambda_nu) ** 2
                    fat = xp.sqrt(num / denom + 1e-10) # 1e-10 pour éviter NaN si denom=0
                else: # 'proba'
                    trace = xp.abs(lambda2) + xp.abs(lambda_rho) + xp.abs(lambda_nu)
                    p2 = xp.abs(lambda2 / trace)
                    p_rho = xp.abs(lambda_rho / trace)
                    p_nu = xp.abs(lambda_nu / trace)
                    lambda_md = 1.0 / 3.0
                    num = (p2 - lambda_md) ** 2 + (p_rho - lambda_md) ** 2 + (p_nu - lambda_md) ** 2
                    denom = p2 ** 2 + p_rho ** 2 + p_nu ** 2
                    fat = xp.sqrt(num) / xp.sqrt(denom)

                fat = xp.sqrt(1.5) * fat
                response = 1.0 - fat 

            # --- restrictions (Eq. 10 pour la 3D / bloc équivalent 2D) ---
            x = lambda_rho - lambda2
            x_extreme_region = x.ravel()

            if is_2d:
                x_min = x_extreme_region.min()
                x_max = x_extreme_region.max()
                response = xp.where(x == x_min, xp.array(1.0, dtype=response.dtype), response)
                response = xp.where(x < x_max, xp.array(0.0, dtype=response.dtype), response)
                response = xp.where(lambda2 > x, xp.array(0.0, dtype=response.dtype), response)
                response = xp.where(lambda_rho > x, xp.array(0.0, dtype=response.dtype), response)
                response = xp.where(lambda2 >= 0, xp.array(0.0, dtype=response.dtype), response)
                response = xp.where(lambda_rho >= 0, xp.array(0.0, dtype=response.dtype), response)
                response = xp.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                x_max = x_extreme_region.max()
                if variant == 'eigen':
                    saturate = x == x_max
                else:
                    saturate = x > x_max
                response = xp.where(saturate, xp.array(1.0, dtype=response.dtype), response)
                response = xp.where(lambda_rho > x, xp.array(0.0, dtype=response.dtype), response)
                response = xp.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
                response = xp.where(lambda2 >= 0, xp.array(0.0, dtype=response.dtype), response)
                response = xp.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
                response = xp.where(lambda_rho >= 0, xp.array(0.0, dtype=response.dtype), response)
                response = xp.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)

            # --- accumulation multi-échelle (Eq. 11-12) ---
            if vesselness is None:
                vesselness = response
            else:
                if is_2d and variant == 'proba':
                    vesselness = vesselness + step_size * (response - step_size)
                elif is_2d:
                    vesselness = vesselness + step_size * xp.tanh(response - step_size) # pas d'abs en 2D
                else:
                    vesselness = vesselness + step_size * xp.tanh(xp.abs(response) - step_size)
                vesselness = xp.maximum(vesselness, response)
            vesselness = xp.clip(vesselness, 0.0, 1.0)

            del hessian, eigvals, eigvals_sorted, lambda2, lambda3_base, lambda_rho, lambda_nu, response, x
            gc.collect()

        total_hessian_time = self._last_total_hessian_time

        # normalisation finale : diffère entre 2D et 3D
        v_max = vesselness.max()
        v_max_safe = v_max if v_max != 0 else xp.array(1.0, dtype=vesselness.dtype)
        filtered_image = vesselness / v_max_safe

        if is_2d:
            filtered_image = xp.where(
                filtered_image < 1e-2, xp.array(0.0, dtype=filtered_image.dtype), filtered_image
            )
        else:
            f_min, f_max = filtered_image.min(), filtered_image.max()
            f_range = f_max - f_min
            f_range_safe = f_range if f_range != 0 else xp.array(1.0, dtype=filtered_image.dtype)
            filtered_image = (filtered_image - f_min) / f_range_safe

        return xp.asnumpy(filtered_image) if self.use_gpu else filtered_image, total_hessian_time

    # MFAT - Uniquement  (for grid search)

    def mfat_apply(
        self,
        cached_eigvals: list,
        image_shape: tuple,
        xp: any,
        tau: float = 0.02,
        tau2: float = 0.2,
        step_size: float = 0.18,
        variant: Literal['eigen', 'proba'] = 'eigen',
        **kwargs
    ) -> ndarray:
        """
        Applique UNIQUEMENT la formule MFAT (pas de Hessien).
        Utilisé pour le grid search après pré-calcul des eigenvalues.
        """
        is_2d = len(image_shape) == 2
        vesselness = None
        
        for _, eigvals in cached_eigvals:
            # eigvals déjà triés par valeur absolue
            lambda2 = eigvals[1]
            lambda3_base = lambda2 if is_2d else eigvals[2]
            
            # Régularisation
            l3_region = lambda3_base.ravel()
            lambda3_min = l3_region.min() if l3_region.size > 0 else xp.array(0.0, dtype=lambda3_base.dtype)
            
            floor_rho = tau * lambda3_min
            floor_nu = tau2 * lambda3_min
            
            lambda_rho = xp.where(
                (lambda3_base < 0) & (lambda3_base >= floor_rho), floor_rho, lambda3_base
            )
            lambda_nu = xp.where(
                (lambda3_base < 0) & (lambda3_base >= floor_nu), floor_nu, lambda3_base
            )
            
            # Calcul F AT
            with np.errstate(divide='ignore', invalid='ignore'):
                if variant == 'eigen':
                    lambda_md = (xp.abs(lambda2) + xp.abs(lambda_rho) + xp.abs(lambda_nu)) / 3.0
                    num = ((xp.abs(lambda2) - lambda_md) ** 2 +
                           (xp.abs(lambda_rho) - lambda_md) ** 2 +
                           (xp.abs(lambda_nu) - lambda_md) ** 2)
                    denom = xp.abs(lambda2) ** 2 + xp.abs(lambda_rho) ** 2 + xp.abs(lambda_nu) ** 2
                    fat = xp.sqrt(num / denom + 1e-10)
                else: # proba
                    trace = xp.abs(lambda2) + xp.abs(lambda_rho) + xp.abs(lambda_nu)
                    p2 = xp.abs(lambda2 / trace)
                    p_rho = xp.abs(lambda_rho / trace)
                    p_nu = xp.abs(lambda_nu / trace)
                    lambda_md = 1.0 / 3.0
                    num = (p2 - lambda_md) ** 2 + (p_rho - lambda_md) ** 2 + (p_nu - lambda_md) ** 2
                    denom = p2 ** 2 + p_rho ** 2 + p_nu ** 2
                    fat = xp.sqrt(num) / xp.sqrt(denom)
                
                fat = xp.sqrt(1.5) * fat
                response = 1.0 - fat
            
            # Restrictions
            x = lambda_rho - lambda2
            if is_2d:
                x_min = x.min()
                x_max = x.max()
                response = xp.where(x == x_min, xp.array(1.0, dtype=response.dtype), response)
                response = xp.where(x < x_max, xp.array(0.0, dtype=response.dtype), response)
                response = xp.where(lambda2 > x, xp.array(0.0, dtype=response.dtype), response)
                response = xp.where(lambda_rho > x, xp.array(0.0, dtype=response.dtype), response)
                response = xp.where(lambda2 >= 0, xp.array(0.0, dtype=response.dtype), response)
                response = xp.where(lambda_rho >= 0, xp.array(0.0, dtype=response.dtype), response)
            else:
                x_max = x.ravel().max()
                saturate = x == x_max if variant == 'eigen'else x > x_max
                response = xp.where(saturate, xp.array(1.0, dtype=response.dtype), response)
                response = xp.where(lambda_rho > x, xp.array(0.0, dtype=response.dtype), response)
                response = xp.where(lambda2 >= 0, xp.array(0.0, dtype=response.dtype), response)
                response = xp.where(lambda_rho >= 0, xp.array(0.0, dtype=response.dtype), response)
            
            response = xp.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Accumulation
            if vesselness is None:
                vesselness = response
            else:
                if is_2d and variant == 'proba':
                    vesselness = vesselness + step_size * (response - step_size)
                elif is_2d:
                    vesselness = vesselness + step_size * xp.tanh(response - step_size)
                else:
                    vesselness = vesselness + step_size * xp.tanh(xp.abs(response) - step_size)
                vesselness = xp.maximum(vesselness, response)
            vesselness = xp.clip(vesselness, 0.0, 1.0)
            
            del response, lambda2, lambda3_base, lambda_rho, lambda_nu, x
        
        # Normalisation
        v_max = vesselness.max()
        v_max_safe = v_max if v_max != 0 else xp.array(1.0, dtype=vesselness.dtype)
        filtered = vesselness / v_max_safe
        
        if is_2d:
            filtered = xp.where(filtered < 1e-2, xp.array(0.0, dtype=filtered.dtype), filtered)
        else:
            f_min, f_max = filtered.min(), filtered.max()
            f_range = f_max - f_min
            f_range_safe = f_range if f_range != 0 else xp.array(1.0, dtype=filtered.dtype)
            filtered = (filtered - f_min) / f_range_safe
        
        return xp.asnumpy(filtered) if self.use_gpu else filtered