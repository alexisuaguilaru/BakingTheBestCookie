import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

def CreateCanvas(
        NumRows: int = 1,
        NumColumns: int = 1,
        FigureSize: tuple[float,float] = (6,5)      
    ) -> tuple[Figure,Axes]:

    Fig , Axes = plt.subplots(
        NumRows,
        NumColumns,
        subplot_kw = {'frame_on':False},
        figsize = FigureSize,
    )

    return Fig , Axes

def SetLabels(
        Axes: Axes,
        LabelX: str = None,
        LabelY: str = None,
        Title: str = None,
        FontSizeLabels: float = 12,
        FontSizeTitle: float = 14,
        FontSizeTicks: float = 10,
    ) -> None:

    if LabelX: Axes.set_xlabel(
        LabelX,
        size = FontSizeLabels,
    )
        
    if LabelY: Axes.set_ylabel(
        LabelY,
        size = FontSizeLabels,
    )
        
    if Title: Axes.set_title(
        Title,
        size = FontSizeTitle,
    )
        
    Axes.tick_params(
        'both',
        labelsize = FontSizeTicks,
    )