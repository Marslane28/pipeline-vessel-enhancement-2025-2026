from typing import Literal, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass

FigureMode = Literal['text', 'plot', 'anim', 'data']

@dataclass
class FigureData:
    figure: Union[plt.Figure, str] # figure or text
    mode: FigureMode # ['text', 'plot', 'anim', 'data']
    name: str 
    
    