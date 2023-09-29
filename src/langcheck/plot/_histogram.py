import math

import plotly.express as px
from dash import Dash, Input, Output, dcc, html

from langcheck.eval.eval_value import EvalValue
from langcheck.plot._css import GLOBAL_CSS


def histogram(eval_value: EvalValue, jupyter_mode: str = 'inline') -> None:
    '''Shows an interactive histogram of all data points in
    :class:`~langcheck.eval.eval_value.EvalValue`. When run in a notebook, this
    usually displays the chart inline in the cell output.

    Args:
        eval_value: The :class:`~langcheck.eval.eval_value.EvalValue` to plot.
        other_eval_value: If provided, another
            :class:`~langcheck.eval.eval_value.EvalValue` to plot on the same
            chart.
        jupyter_mode: Defaults to 'inline', which displays the chart in the
            cell output. For Colab, set this to 'external' instead. See the
            Dash documentation for more info:
            https://dash.plotly.com/workspaces/using-dash-in-jupyter-and-workspaces#display-modes
    '''
    # Rename some EvalValue fields for display
    df = eval_value.to_df()
    df.rename(columns={'metric_value': eval_value.metric_name}, inplace=True)

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
        fig = px.histogram(df, x=eval_value.metric_name)

        # Manually set the number of bins in the histogram. We can't use the
        # nbins parameter of px.histogram() since it's just a suggested number
        # of bins. See: https://community.plotly.com/t/histogram-bin-size-with-plotly-express/38927/5  # NOQA E501
        start = math.floor(df[eval_value.metric_name].min())
        end = math.ceil(df[eval_value.metric_name].max())
        step_size = (end - start) / int(num_bins)
        fig.update_traces(xbins={'start': start, 'end': end, 'size': step_size})

        # If the user manually zoomed in, keep that zoom level even when
        # update_figure() re-runs
        fig.update_layout(uirevision='constant')

        # Disable drag-to-zoom by default (the user can still enable it in the
        # modebar)
        fig.update_layout(dragmode=False)

        return fig

    # Display the Dash app inline in the notebook
    app.run(jupyter_mode=jupyter_mode)  # type: ignore
