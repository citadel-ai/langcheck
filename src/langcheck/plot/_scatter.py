import math
from copy import deepcopy
from typing import Optional

import plotly.express as px
from dash import Dash, Input, Output, dcc, html

from langcheck.eval.eval_value import EvalValue
from langcheck.plot._css import GLOBAL_CSS, INPUT_CSS, NUM_RESULTS_CSS


def scatter(eval_value: EvalValue,
            other_eval_value: Optional[EvalValue] = None,
            jupyter_mode: str = 'inline') -> None:
    '''Shows an interactive scatter plot of all data points in an
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
    if other_eval_value is None:
        _scatter_one_eval_value(eval_value, jupyter_mode)
    else:
        _scatter_two_eval_values(eval_value, other_eval_value, jupyter_mode)


def _scatter_one_eval_value(eval_value: EvalValue, jupyter_mode: str) -> None:
    '''Shows an interactive scatter plot of all data points in one
    :class:`~langcheck.eval.eval_value.EvalValue`.
    '''
    # Rename some EvalValue fields for display
    df = eval_value.to_df()
    df.rename(columns={'metric_value': eval_value.metric_name}, inplace=True)
    df['prompt'] = df['prompt'].fillna('None')
    df['reference_output'] = df['reference_output'].fillna('None')
    df['source'] = df['source'].fillna('None')

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
                         y=eval_value.metric_name,
                         hover_data=filtered_df.columns)

        # Explicitly set the default axis ranges (with a little padding) so that
        # the plot doesn't change when the user types in the search boxes
        fig.update_xaxes(range=[-0.1, len(df)])
        fig.update_yaxes(range=[
            min(-0.1, math.floor(df[eval_value.metric_name].min())),
            max(1.1, math.ceil(df[eval_value.metric_name].max()))
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


def _scatter_two_eval_values(eval_value: EvalValue, other_eval_value: EvalValue,
                             jupyter_mode: str) -> None:
    '''Shows an interactive scatter plot of all data points in two
    :class:`~langcheck.eval.eval_value.EvalValue`.
    '''
    # Validate that the two EvalValues have the same data points
    if eval_value.generated_outputs != other_eval_value.generated_outputs:
        raise ValueError('Both EvalValues must have the same generated_outputs')
    if eval_value.prompts != other_eval_value.prompts:
        raise ValueError('Both EvalValues must have the same prompts')
    if eval_value.reference_outputs != other_eval_value.reference_outputs:
        raise ValueError('Both EvalValues must have the same reference_outputs')
    if eval_value.language != other_eval_value.language:
        raise ValueError('Both EvalValues must have the same language')

    # Append "(other)" to the metric name of the second EvalValue if necessary.
    # (It's possible to plot two EvalValues from the same metric, e.g. if you
    # compute semantic_sim() with a local model and an OpenAI model)
    if eval_value.metric_name == other_eval_value.metric_name:
        other_eval_value = deepcopy(other_eval_value)
        other_eval_value.metric_name += ' (other)'

    # Rename some EvalValue fields for display
    df = eval_value.to_df()
    df.rename(columns={'metric_value': eval_value.metric_name}, inplace=True)
    df[other_eval_value.metric_name] = other_eval_value.to_df()['metric_value']
    df['prompt'] = df['prompt'].fillna('None')
    df['reference_output'] = df['reference_output'].fillna('None')
    df['source'] = df['source'].fillna('None')

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
        # the tooltip like _scatter_one_eval_value() since Plotly always
        # displays the x and y values at the top.)
        hover_data = {col: True for col in filtered_df.columns}
        hover_data['index'] = filtered_df.index
        fig = px.scatter(filtered_df,
                         x=eval_value.metric_name,
                         y=other_eval_value.metric_name,
                         hover_data=hover_data)

        # Explicitly set the default axis ranges (with a little padding) so that
        # the plot doesn't change when the user types in the search boxes
        fig.update_xaxes(range=[
            min(-0.1, math.floor(df[eval_value.metric_name].min())),
            max(1.1, math.ceil(df[eval_value.metric_name].max()))
        ])
        fig.update_yaxes(range=[
            min(-0.1, math.floor(df[other_eval_value.metric_name].min())),
            max(1.1, math.ceil(df[other_eval_value.metric_name].max()))
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
