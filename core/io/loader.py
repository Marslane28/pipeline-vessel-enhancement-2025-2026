import pickle
import numpy as np
import tifffile as tiff
import nibabel as nib
import SimpleITK as sitk
import json
from skimage import io, color
from pathlib import Path

from configs.args import INPUT_DIR
from core.config.benchmark import BenchmarkResults
from core.config.base import Config, ConfigBase
from core.io.logger import setup_logger, close_logger
from logging import Logger
from core.utils.helpers import normalize_data, crop_data

class Loader:
    
    def __init__(self, input_dir: str = "raw", logger : Logger = setup_logger()) -> None:
        self.logger = logger
        self.input_dir = Path(f"{INPUT_DIR}/{input_dir}")
        self.input_dir.mkdir(parents=True, exist_ok=True)


    def load_results(self, filename: str) -> BenchmarkResults:
        with open(filename, 'rb') as f:
            results: BenchmarkResults = pickle.load(f)
        return results
    
    def load_config(self, filename: str, config_instance: Config) -> ConfigBase:
        path = Path(self.input_dir) / filename
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        config_instance.from_dict(data)
        
        return config_instance

    
    def load_data(self, filename: str, normalize: bool = True, crop: bool = False, target_shape: tuple = (64, 64, 64)) -> np.ndarray:
        path = Path(self.input_dir) / filename
        suffixes = path.suffixes
        if not suffixes:
            raise ValueError(f"File {filename} has no extension or is invalid")

        # .JPG, .PNG
        if suffixes[-1] in ['.jpg', '.png', '.jp2']:
            data = io.imread(path).astype(np.float32)
            if data.ndim == 3:
                data = color.rgb2gray(data)

        # .TIF, .TIFF
        elif suffixes[-1] in ['.tif', '.tiff']:
            data = tiff.imread(path).astype(np.float32)
            if data.ndim == 3:  # 3D
                data = np.transpose(data, (2, 1, 0))  # (z, y, x) -> (x, y, z)
            else:  # 2D
                data = np.transpose(data, (1, 0))  # (y, x) -> (x, y)
                
        # .NII, .NII.GZ
        elif suffixes[-2:] == ['.nii', '.gz'] or suffixes[-1] == '.nii':
            data = np.squeeze(nib.load(path).get_fdata().astype(np.float32))

        # .MHD, .RAW
        elif suffixes[-1] == '.mhd':
            sitk_image = sitk.ReadImage(path)
            data = sitk.GetArrayFromImage(sitk_image).astype(np.float32)  # shape: (z, y, x)
            data = np.transpose(data, (2, 1, 0))  # -> (x, y, z)
            
        # .NPY
        elif suffixes[-1] == '.npy':
            data = np.load(path).astype(np.float32)
        
    
        else:
            raise ValueError(f'Unsupported file type {suffixes[-1]}. Only the following types are supported: .jpg, .tif, .tiff, .nii(.gz), .mhd, .gif or .npy')

        # Crop data
        if crop:
            data = crop_data(data, target_shape)
        
        # Normalize data [0, 1]
        if normalize:
            data = normalize_data(data)

        self.logger.info(f'[LOAD] Data {path.name} loaded | shape={data.shape}.')
        return data
        

    def get_metadata(self, filename: str) -> dict:
        path = Path(self.input_dir) / filename
        suffixes = path.suffixes

        # .JPG, .PNG
        if suffixes[-1] in ['.jpg', '.png', '.jp2']:
            name = path.stem
            extension = suffixes[-1]
            shape = io.imread(path).shape
            
            metadata = {
                'name': name,
                'extension': extension,
                'shape': shape,
            }

        # .TIF, .TIFF
        elif suffixes[-1] in ['.tif', '.tiff']:
            with tiff.TiffFile(path) as tif:
                metadata = {}
                for tag in tif.pages[0].tags.values():
                    metadata[tag.name] = tag.value

        # .NII, .NII.GZ
        elif suffixes[-1] == '.nii' or (suffixes[-2:] == ['.nii', '.gz']):
            img = nib.load(path)
            header = img.header
            affine = img.affine
            shape = img.shape
            
            metadata = {
                'shape': shape,
                'datatype': str(header.get_data_dtype()),
                'voxel_size': header.get_zooms(),
                'affine': affine.tolist(), # list pou r la lisibilité
            }
            
        # .MHD, .RAW
        elif suffixes[-1] == '.mhd':
            data = sitk.ReadImage(path)
            size = data.GetSize()  # (x, y, z)
            spacing = data.GetSpacing()
            origin = data.GetOrigin()
            direction = data.GetDirection()
            pixel_type = data.GetPixelIDTypeAsString()

            metadata = {
                'size': size,
                'spacing': spacing,
                'origin': origin,
                'direction': direction,
                'pixel_type': pixel_type,
            }
            
        # .NPY
        elif suffixes[-1] == '.npy':
            array = np.load(path)
            metadata = {
                'shape': array.shape,
                'dtype': str(array.dtype),
            }


        else:
            raise ValueError(f'Unsupported file type {suffixes[-1]}. Only the following types are supported: .jpg, .tif, .tiff, .nii(.gz), .mhd, .gif or .npy')

        self.logger.info(f'[METADATA] "{filename}":' + "\n    ".join([f"  - {k}: {v}" for k, v in metadata.items()]))            
        
        return metadata
