
import matplotlib.pyplot as plt
from typing import Union

from core.utils.viewer import Viewer
from core.config.figure import FigureData, FigureMode


class AnalyticsBase():
    
    def __init__(self):
        self.viewer = Viewer()
        

    def _create_figure(self, figure: Union[plt.Figure, str], name: str, mode: FigureMode) -> FigureData:
        return FigureData(name=name, figure=figure, mode=mode)
    
    
    