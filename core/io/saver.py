import pickle
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation
from datetime import datetime
from typing import Literal, Optional
import json

from core.config.figure import FigureData
from core.config.benchmark import BenchmarkResults
from core.config.base import ConfigBase
from core.io.logger import setup_logger, close_logger
from logging import Logger
from configs.args import OUTPUT_DIR

class Saver:
    def __init__(self, 
            experiment_name: str = "default", 
            output_dir: str | Path = "results", 
            logger: Logger = setup_logger(),
            print_mode: bool = False,
            dataset: str = None,
            enhancer: str = None
        ):
        self.logger = logger
        self.print_mode = print_mode
        if dataset and enhancer:
            full_name = f"{dataset}_enhancer_{enhancer}_{self._get_timestamp()}"
        elif dataset:
            full_name = f"{dataset}_{self._get_timestamp()}"
        else:
            full_name = f"{experiment_name}_{self._get_timestamp()}"
        self.output_dir = Path(f'{OUTPUT_DIR}/{output_dir}/{full_name}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_timestamp(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return timestamp
    
    
    def save_results(self, results: BenchmarkResults, filename: Optional[str], dirname: str = None):
        output_dir = self.output_dir / dirname if dirname is not None else self.output_dir
        path = output_dir / f'results_{filename}'
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(results, f)
        
        if self.print_mode:
            self.logger.info(f'[SAVE] Results saved as {path.stem}.')
        
        return path
    
    
    def save_text(self, content: str, filename: str, dirname: Optional[str] = None):
        output_dir = self.output_dir / dirname if dirname is not None else self.output_dir
        path = output_dir / f'text_{filename}.txt'
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)


    def save_data(self, data: np.ndarray, filename: str, dirname: Optional[str] = None, extension: Literal['.nii', '.npz', '.npy', ] = '.npz'):
        output_dir = self.output_dir / dirname if dirname is not None else self.output_dir
        path = output_dir / f'data_{filename}'
        path.parent.mkdir(parents=True, exist_ok=True)

        if extension == '.nii':
            data_nii = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4))
            nib.save(data_nii, path)
        elif extension in ['.npy', '.npz']:
            if extension == '.npy':
                np.save(path, data)
            else:
                np.savez(path, data=data)
        else:
            raise ValueError('Extension extension invalid.')


    def save_plot(self, fig: plt.Figure, filename: str, dirname: Optional[str] = None, dpi: int = 150):
        output_dir = self.output_dir / dirname if dirname is not None else self.output_dir
        path = output_dir / f'plot_{filename}'
        path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(path, dpi=dpi, bbox_inches='tight')


    def save_anim(self, anim: FuncAnimation, filename: str, dirname: Optional[str] = None, extension: Literal['.mp4', '.mov', '.avi', '.gif'] = '.gif', fps: int = 30, dpi: int = 150):
        output_dir = self.output_dir / dirname if dirname is not None else self.output_dir
        path = output_dir / f'anim_{filename}{extension}'
        path.parent.mkdir(parents=True, exist_ok=True)

        anim.save(str(path), fps=fps, dpi=dpi)
        

    def save_figure(self, figure: FigureData, dirname: Optional[str] = None):
        match figure.mode:
            case 'plot': self.save_plot(figure.figure, figure.name, dirname)
            case 'text': self.save_text(figure.figure, figure.name, dirname)
            case 'anim':
                pass
                 #self.save_anim(figure.figure, figure.name, dirname)
            case 'data': self.save_data(figure.figure, figure.name, dirname)         
        
        if self.print_mode:
            self.logger.info(f'[SAVE] {figure.mode.capitalize()} saved as {figure.name}.')

    def save_config(self, config: ConfigBase, filename, dirname: Optional[str] = None):
        output_dir = self.output_dir / dirname if dirname is not None else self.output_dir
        path = output_dir / f'config_{filename}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=4, ensure_ascii=False)
            print(f"Configuration sauvegardée dans {path}")
