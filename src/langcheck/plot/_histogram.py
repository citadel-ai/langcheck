import math

import plotly.express as px
from dash import Dash, Input, Output, dcc, html

from langcheck.metrics.metric_value import MetricValue, MetricValueWithThreshold
from langcheck.plot._css import GLOBAL_CSS
from langcheck.plot._utils import Axis, _plot_threshold


def histogram(metric_value: MetricValue, jupyter_mode: str = 'inline') -> None:
    '''Shows an interactive histogram of all data points in
    :class:`~langcheck.metrics.metric_value.MetricValue`. When run in a
    notebook, this usually displays the chart inline in the cell output.

    Args:
        metric_value: The :class:`~langcheck.metrics.metric_value.MetricValue`
            to plot.
        other_metric_value: If provided, another
            :class:`~langcheck.metrics.metric_value.MetricValue` to plot on the
            same chart.
        jupyter_mode: Defaults to 'inline', which displays the chart in the
            cell output. For Colab, set this to 'external' instead. See the
            Dash documentation for more info:
            https://dash.plotly.com/workspaces/using-dash-in-jupyter-and-workspaces#display-modes
    '''
    # Rename some MetricValue fields for display
    df = metric_value.to_df()
    df.rename(columns={'metric_value': metric_value.metric_name}, inplace=True)

    # Define layout of the Dash app (histogram + input for number of bins)
    app = Dash(__name__)
    app.layout = html.Div([
        html.Div([
            html.Label('Number of bins: '),
            dcc.Slider(id='num_bins',
                       min=1,
                       max=50,
                       step=1,
                       value=10,
                       marks={
                           1: '1',
                           10: '10',
                           20: '20',
                           30: '30',
                           40: '40',
                           50: '50'
                       },
                       tooltip={
                           "placement": "bottom",
                           "always_visible": True
                       })
        ]),
        dcc.Graph(
            id='histogram',
            config={
                'displaylogo': False,
                'modeBarButtonsToRemove': ['select', 'lasso2d', 'resetScale']
            })
    ],
                          style=GLOBAL_CSS)

    # This function gets called whenever the user changes the num_bins value
    @app.callback(
        Output('histogram', 'figure'),
        Input('num_bins', 'value'),
    )
    def update_figure(num_bins):
        # Plot the histogram
        fig = px.histogram(df, x=metric_value.metric_name)
        if isinstance(metric_value, MetricValueWithThreshold):
            _plot_threshold(fig, metric_value.threshold_op,
                            metric_value.threshold, Axis.vertical)
        # Manually set the number of bins in the histogram. We can't use the
        # nbins parameter of px.histogram() since it's just a suggested number
        # of bins. See: https://community.plotly.com/t/histogram-bin-size-with-plotly-express/38927/5  # NOQA: E501
        start = math.floor(df[metric_value.metric_name].min())
        end = math.ceil(df[metric_value.metric_name].max())
        step_size = (end - start) / int(num_bins)
        fig.update_traces(xbins={'start': start, 'end': end, 'size': step_size})
        # Explicitly set the default axis ranges (with a little padding) so that
        # the range plotted would not be influenced by threshold settings
        fig.update_xaxes(range=[
            min(-0.1, math.floor(df[metric_value.metric_name].min())),
            max(1.1, math.ceil(df[metric_value.metric_name].max()))
        ])
        # If the user manually zoomed in, keep that zoom level even when
        # update_figure() re-runs
        fig.update_layout(uirevision='constant')

        # Disable drag-to-zoom by default (the user can still enable it in the
        # modebar)
        fig.update_layout(dragmode=False)

        return fig

    # Display the Dash app inline in the notebook
    app.run(jupyter_mode=jupyter_mode)  # type: ignore
