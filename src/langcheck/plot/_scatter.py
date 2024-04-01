import math
import textwrap
from copy import deepcopy
from typing import Optional, Union

import plotly.express as px
from dash import Dash, Input, Output, dcc, html
from pandas.core.indexes.base import Index

from langcheck.metrics.metric_value import MetricValue, MetricValueWithThreshold
from langcheck.plot._css import GLOBAL_CSS, INPUT_CSS, NUM_RESULTS_CSS
from langcheck.plot._utils import Axis, _plot_threshold


def scatter(metric_value: MetricValue,
            other_metric_value: Optional[MetricValue] = None,
            jupyter_mode: str = 'inline') -> None:
    '''Shows an interactive scatter plot of all data points in an
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
    if metric_value.is_pairwise or (other_metric_value is not None and
                                    other_metric_value.is_pairwise):
        raise NotImplementedError(
            'Scatter plots for pairwise MetricValues are not supported yet')

    if other_metric_value is None:
        _scatter_one_metric_value(metric_value, jupyter_mode)
    else:
        _scatter_two_metric_values(metric_value, other_metric_value,
                                   jupyter_mode)


def _format_text_for_hover(text: str):
    '''Helper function to format a string so that it displays nicely on hover in
    the scatter plot.
    '''
    # First, split the text by newline characters. This is recommended in
    # https://docs.python.org/3/library/textwrap.html#textwrap.TextWrapper.replace_whitespace
    paragraphs = text.split('\n')
    # Then, split the paragraphs into separate lines with a max width of 70
    # chars (default)
    lines = [line for p in paragraphs for line in textwrap.wrap(p)]
    # Only show a max of 5 lines. If there are more than 5, add '...' to
    # indicate that the text has been cut off
    if len(lines) > 5:
        lines = lines[:5] + ['...']
    return '<br>'.join(lines)


def _scatter_one_metric_value(metric_value: MetricValue,
                              jupyter_mode: str) -> None:
    '''Shows an interactive scatter plot of all data points in one
    :class:`~langcheck.metrics.metric_value.MetricValue`.
    '''
    # Rename some MetricValue fields for display
    df = metric_value.to_df()
    df.rename(columns={'metric_value': metric_value.metric_name}, inplace=True)
    df['prompt'] = df['prompt'].fillna('None').apply(_format_text_for_hover)
    df['reference_output'] = df['reference_output'].fillna('None').apply(
        _format_text_for_hover)
    df['source'] = df['source'].fillna('None').apply(_format_text_for_hover)
    df['explanation'] = df['explanation'].fillna('None').apply(
        _format_text_for_hover)
    df['generated_output'] = df['generated_output'].fillna('None').apply(
        _format_text_for_hover)

    # Define layout of the Dash app (chart + search boxes)
    app = Dash(__name__)
    app.layout = html.Div([
        html.Div([
            html.Label('Filter generated_outputs: '),
            dcc.Input(id='filter_generated_outputs',
                      type='text',
                      placeholder='Type to search...',
                      style=INPUT_CSS),
        ]),
        html.Div([
            html.Label('Filter reference_outputs: '),
            dcc.Input(id='filter_reference_outputs',
                      type='text',
                      placeholder='Type to search...',
                      style=INPUT_CSS),
        ]),
        html.Div([
            html.Label('Filter prompts: '),
            dcc.Input(id='filter_prompts',
                      type='text',
                      placeholder='Type to search...',
                      style=INPUT_CSS),
        ]),
        html.Div([
            html.Label('Filter sources: '),
            dcc.Input(id='filter_sources',
                      type='text',
                      placeholder='Type to search...',
                      style=INPUT_CSS),
        ]),
        html.Div([html.Span(id='num_results_message', style=NUM_RESULTS_CSS)]),
        dcc.Graph(
            id='scatter_plot',
            config={
                'displaylogo': False,
                'modeBarButtonsToRemove': ['select', 'lasso2d', 'resetScale']
            })
    ],
                          style=GLOBAL_CSS)

    # This function gets called whenever the user types in the search boxes
    @app.callback(Output('scatter_plot', 'figure'),
                  Output('num_results_message', 'children'),
                  Input('filter_generated_outputs', 'value'),
                  Input('filter_reference_outputs', 'value'),
                  Input('filter_prompts', 'value'),
                  Input('filter_sources', 'value'))
    def update_figure(filter_generated_outputs, filter_reference_outputs,
                      filter_prompts, filter_sources):
        # Filter data points based on search boxes, case-insensitive
        filtered_df = df.copy()
        if filter_generated_outputs:
            filtered_df = filtered_df[
                filtered_df['generated_output'].str.lower().str.contains(
                    filter_generated_outputs.lower())]
        if filter_reference_outputs:
            filtered_df = filtered_df[
                filtered_df['reference_output'].str.lower().str.contains(
                    filter_reference_outputs.lower())]
        if filter_prompts:
            filtered_df = filtered_df[
                filtered_df['prompt'].str.lower().str.contains(
                    filter_prompts.lower())]
        if filter_sources:
            filtered_df = filtered_df[
                filtered_df['source'].str.lower().str.contains(
                    filter_sources.lower())]

        # Configure the actual scatter plot
        fig = px.scatter(filtered_df,
                         x=filtered_df.index,
                         y=metric_value.metric_name,
                         hover_data=filtered_df.columns)
        if isinstance(metric_value, MetricValueWithThreshold):
            _plot_threshold(fig, metric_value.threshold_op,
                            metric_value.threshold, Axis.horizontal)
        # Explicitly set the default axis ranges (with a little padding) so that
        # the plot doesn't change when the user types in the search boxes
        fig.update_xaxes(range=[-0.1, len(df)])
        fig.update_yaxes(range=[
            min(-0.1, math.floor(df[metric_value.metric_name].min())),
            max(1.1, math.ceil(df[metric_value.metric_name].max()))
        ])

        # However, if the user manually zoomed in, keep that zoom level even
        # when update_figure() re-runs
        fig.update_layout(uirevision='constant')

        # Disable drag-to-zoom by default (the user can still enable it
        # in the modebar)
        fig.update_layout(dragmode=False)

        # Display a message about how many data points are hidden
        num_results_message = (
            f'Showing {len(filtered_df)} of {len(df)} data points.')

        return fig, num_results_message

    # Display the Dash app inline in the notebook
    # TODO: This doesn't seem to display inline if you click "Run All" in VSCode
    # instead of running the cell directly
    app.run(jupyter_mode=jupyter_mode)  # type: ignore


def _scatter_two_metric_values(metric_value: MetricValue,
                               other_metric_value: MetricValue,
                               jupyter_mode: str) -> None:
    '''Shows an interactive scatter plot of all data points in two
    :class:`~langcheck.metrics.metric_value.MetricValue`.
    '''
    # Validate that the two MetricValues have the same data points
    if metric_value.generated_outputs != other_metric_value.generated_outputs:
        raise ValueError(
            'Both MetricValues must have the same generated_outputs')
    if metric_value.prompts != other_metric_value.prompts:
        raise ValueError('Both MetricValues must have the same prompts')
    if metric_value.reference_outputs != other_metric_value.reference_outputs:
        raise ValueError(
            'Both MetricValues must have the same reference_outputs')
    if metric_value.language != other_metric_value.language:
        raise ValueError('Both MetricValues must have the same language')

    # Append "(other)" to the metric name of the second MetricValue if
    # necessary. (It's possible to plot two MetricValues from the same metric,
    # e.g. if you compute semantic_similarity() with a local model and an OpenAI
    # model)
    if metric_value.metric_name == other_metric_value.metric_name:
        other_metric_value = deepcopy(other_metric_value)
        other_metric_value.metric_name += ' (other)'

    # Rename some MetricValue fields for display
    df = metric_value.to_df()
    df.rename(columns={'metric_value': metric_value.metric_name}, inplace=True)
    df[other_metric_value.metric_name] = other_metric_value.to_df(
    )['metric_value']
    df['prompt'] = df['prompt'].fillna('None').apply(_format_text_for_hover)
    df['reference_output'] = df['reference_output'].fillna('None').apply(
        _format_text_for_hover)
    df['source'] = df['source'].fillna('None').apply(_format_text_for_hover)
    df['explanation'] = df['explanation'].fillna('None').apply(
        _format_text_for_hover)
    df['generated_output'] = df['generated_output'].fillna('None').apply(
        _format_text_for_hover)

    # Define layout of the Dash app (chart + search boxes)
    app = Dash(__name__)
    app.layout = html.Div([
        html.Div([
            html.Label('Filter generated_outputs: '),
            dcc.Input(id='filter_generated_outputs',
                      type='text',
                      placeholder='Type to search...',
                      style=INPUT_CSS),
        ]),
        html.Div([
            html.Label('Filter reference_outputs: '),
            dcc.Input(id='filter_reference_outputs',
                      type='text',
                      placeholder='Type to search...',
                      style=INPUT_CSS),
        ]),
        html.Div([
            html.Label('Filter prompts: '),
            dcc.Input(id='filter_prompts',
                      type='text',
                      placeholder='Type to search...',
                      style=INPUT_CSS),
        ]),
        html.Div([
            html.Label('Filter sources: '),
            dcc.Input(id='filter_sources',
                      type='text',
                      placeholder='Type to search...',
                      style=INPUT_CSS),
        ]),
        html.Div([html.Span(id='num_results_message', style=NUM_RESULTS_CSS)]),
        dcc.Graph(
            id='scatter_plot',
            config={
                'displaylogo': False,
                'modeBarButtonsToRemove': ['select', 'lasso2d', 'resetScale']
            })
    ],
                          style=GLOBAL_CSS)

    # This function gets called whenever the user types in the search boxes
    @app.callback(Output('scatter_plot', 'figure'),
                  Output('num_results_message', 'children'),
                  Input('filter_generated_outputs', 'value'),
                  Input('filter_reference_outputs', 'value'),
                  Input('filter_prompts', 'value'),
                  Input('filter_sources', 'value'))
    def update_figure(filter_generated_outputs, filter_reference_outputs,
                      filter_prompts, filter_sources):
        # Filter data points based on search boxes, case-insensitive
        filtered_df = df.copy()
        if filter_generated_outputs:
            filtered_df = filtered_df[
                filtered_df['generated_output'].str.lower().str.contains(
                    filter_generated_outputs.lower())]
        if filter_reference_outputs:
            filtered_df = filtered_df[
                filtered_df['reference_output'].str.lower().str.contains(
                    filter_reference_outputs.lower())]
        if filter_prompts:
            filtered_df = filtered_df[
                filtered_df['prompt'].str.lower().str.contains(
                    filter_prompts.lower())]
        if filter_sources:
            filtered_df = filtered_df[
                filtered_df['source'].str.lower().str.contains(
                    filter_sources.lower())]

        # Configure the actual scatter plot
        # (We need to explicitly add the index column into hover_data here.
        # Unfortunately it's not possible to make "index" show up at the top of
        # the tooltip like _scatter_one_metric_value() since Plotly always
        # displays the x and y values at the top.)
        hover_data: dict[str, Union[bool, Index]] = {
            col: True for col in filtered_df.columns
        }
        hover_data['index'] = filtered_df.index
        fig = px.scatter(filtered_df,
                         x=metric_value.metric_name,
                         y=other_metric_value.metric_name,
                         hover_data=hover_data)
        # Draw threshold if any of metric_value is MetricValueWithThreshold
        if isinstance(metric_value, MetricValueWithThreshold):
            _plot_threshold(fig, metric_value.threshold_op,
                            metric_value.threshold, Axis.vertical)
        if isinstance(other_metric_value, MetricValueWithThreshold):
            _plot_threshold(fig, other_metric_value.threshold_op,
                            other_metric_value.threshold, Axis.horizontal)

        # Explicitly set the default axis ranges (with a little padding) so that
        # the plot doesn't change when the user types in the search boxes
        fig.update_xaxes(range=[
            min(-0.1, math.floor(df[metric_value.metric_name].min())),
            max(1.1, math.ceil(df[metric_value.metric_name].max()))
        ])
        fig.update_yaxes(range=[
            min(-0.1, math.floor(df[other_metric_value.metric_name].min())),
            max(1.1, math.ceil(df[other_metric_value.metric_name].max()))
        ])

        # However, if the user manually zoomed in, keep that zoom level even
        # when update_figure() re-runs
        fig.update_layout(uirevision='constant')

        # Disable drag-to-zoom by default (the user can still enable it in the
        # modebar)
        fig.update_layout(dragmode=False)

        # Display a message about how many data points are hidden
        num_results_message = (
            f'Showing {len(filtered_df)} of {len(df)} data points.')

        return fig, num_results_message

    # Display the Dash app inline in the notebook
    # TODO: This doesn't seem to display inline if you click "Run All" in VSCode
    # instead of running the cell directly
    app.run(jupyter_mode=jupyter_mode)  # type: ignore
