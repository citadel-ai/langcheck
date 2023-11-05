import re
from enum import Enum
from typing import Union

from plotly.graph_objects import Figure


class Axis(Enum):
    vertical = 0
    horizontal = 1


def _plot_threshold(fig: Figure, threshold_op: str,
                    threshold: Union[float, int], direction: Axis):
    '''Draw dash line on target figure by giving threshold and axis
    '''
    threshold_text = f"{threshold_op} {threshold}"
    if direction == Axis.horizontal:  # Draw a horizontal line
        fig.add_hline(y=threshold,
                      line_width=3,
                      line_dash='dash',
                      annotation_text=threshold_text,
                      annotation_position='right')
    elif direction == Axis.vertical:  # Draw a vertical line
        fig.add_vline(x=threshold,
                      line_width=3,
                      line_dash='dash',
                      annotation_text=threshold_text,
                      annotation_position='top')
    return
