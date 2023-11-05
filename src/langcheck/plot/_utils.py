import re
from enum import Enum

from plotly.graph_objects import Figure


class Axis(Enum):
    vertical = 0
    horizontal = 1


def _plot_threshold(fig: Figure, threshold_text: str, direction: Axis):
    # Analyize threshold condition
    pattern = r'(<|<=|>|>=|==)\s([0-9]\.[0-9]+)'
    match = re.search(pattern, threshold_text)
    if match:
        operator, threshold = match.groups()
    else:
        raise ValueError('Threshold not Found!')
    operator, threshold = operator[0], float(threshold)
    if direction == Axis.horizontal:  # Paint in row
        fig.add_hline(y=threshold,
                      line_width=3,
                      line_dash='dash',
                      annotation_text=threshold_text,
                      annotation_position='left')
    elif direction == Axis.vertical:
        fig.add_vline(x=threshold,
                      line_width=3,
                      line_dash='dash',
                      annotation_text=threshold_text,
                      annotation_position='top')
    return
