import SimpleITK as sitk


class Resampler:


    def __init__(self, target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.target_spacing = target_spacing

    # Helpers

    def _compute_new_size(self, image: sitk.Image, output_spacing: tuple) -> list[int]:
        original_spacing = image.GetSpacing()
        original_size    = image.GetSize()
        return [
            int(round(original_size[i] * (original_spacing[i] / output_spacing[i])))
            for i in range(3)
        ]

    def _base_resampler(self, image: sitk.Image,
                        interpolator, output_spacing: tuple) -> sitk.ResampleImageFilter:
        new_size = self._compute_new_size(image, output_spacing)
        r = sitk.ResampleImageFilter()
        r.SetSize(new_size)
        r.SetOutputSpacing(output_spacing)
        r.SetOutputOrigin(image.GetOrigin())
        r.SetOutputDirection(image.GetDirection())
        r.SetInterpolator(interpolator)
        r.SetDefaultPixelValue(0)
        return r

    # API publique

    def resample_image(self, image: sitk.Image,
                       output_spacing: tuple = None) -> sitk.Image:

        output_spacing = output_spacing or self.target_spacing
        r = self._base_resampler(image, sitk.sitkBSpline, output_spacing)
        return r.Execute(image)

    def resample_mask(self, mask: sitk.Image,
                      reference_image: sitk.Image = None,
                      output_spacing: tuple = None) -> sitk.Image:

        if reference_image is not None:
            r = sitk.ResampleImageFilter()
            r.SetReferenceImage(reference_image)
            r.SetInterpolator(sitk.sitkNearestNeighbor)
            r.SetDefaultPixelValue(0)
            return r.Execute(mask)

        output_spacing = output_spacing or self.target_spacing
        r = self._base_resampler(mask, sitk.sitkNearestNeighbor, output_spacing)
        return r.Execute(mask)