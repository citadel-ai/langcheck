import re
from enum import Enum
from typing import Union

from plotly.graph_objects import Figure


class Axis(Enum):
    vertical = 0
    horizontal = 1


def _plot_threshold(fig: Figure, threshold_op: str,
                    threshold: Union[float, int], direction: Axis):
    '''Draw a dashed line on the target figure at the specified threshold value
    along either the horizontal or vertical axis.

    Args:
        fig: Plotly figure to draw the line on
        threshold_op: A string representing the threshold operator, e.g. '<'
        threshold: Threshold value
        direction: Direction (horizontal or vertical) that the dashed line
            should be drawn
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
