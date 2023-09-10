import math
from typing import Optional

import plotly.express as px
from dash import Dash, Input, Output, dcc, html

from langcheck.eval import EvalValue


def scatter(eval_value: EvalValue,
            other_eval_value: Optional[EvalValue] = None) -> None:
    '''Shows an interactive scatter plot of all data points in an EvalValue.
    Intended to be used in a Jupyter notebook.

    Args:
        eval_value: The EvalValue to plot.
        other_eval_value: If provided, another EvalValue to plot on the same
            chart.
    '''
    if other_eval_value is None:
        _scatter_one_eval_value(eval_value)
    else:
        _scatter_two_eval_values(eval_value, other_eval_value)


def _scatter_one_eval_value(eval_value: EvalValue) -> None:
    '''Shows an interactive scatter plot of all data points in one EvalValue.
    Intended to be used in a Jupyter notebook.
    '''
    # Rename some EvalValue fields for display
    df = eval_value.to_df()
    df.rename(columns={'metric_value': eval_value.metric_name}, inplace=True)
    df['prompt'] = df['prompt'].fillna('None')

    # Define layout of the Dash app (chart + search boxes)
    app = Dash(__name__)
    app.layout = html.Div([
        html.Div([
            html.Label('Filter generated_outputs: ',
                       style={'background-color': 'white'}),
            dcc.Input(id='filter_generated_outputs',
                      type='text',
                      placeholder='Type to search...'),
        ]),
        html.Div([
            html.Label('Filter prompts: ', style={'background-color':
                                                  'white'}),
            dcc.Input(id='filter_prompts',
                      type='text',
                      placeholder='Type to search...'),
        ]),
        html.Div([
            html.Span(id='num_results_message',
                      style={
                          'background-color': 'white',
                          'font-style': 'italic'
                      })
        ]),
        dcc.Graph(id='scatter_plot',
                  config={
                      'displaylogo': False,
                      'modeBarButtonsToRemove':
                      ['select', 'lasso2d', 'resetScale']
                  })
    ])

    # This function gets called whenever the user types in the search boxes
    @app.callback(Output('scatter_plot', 'figure'),
                  Output('num_results_message', 'children'),
                  Input('filter_generated_outputs', 'value'),
                  Input('filter_prompts', 'value'))
    def update_figure(filter_generated_outputs, filter_prompts):
        # Filter data points based on search boxes, case-insensitive
        filtered_df = df.copy()
        if filter_generated_outputs:
            filtered_df = filtered_df[
                filtered_df['generated_output'].str.lower().str.contains(
                    filter_generated_outputs.lower())]
        if filter_prompts:
            filtered_df = filtered_df[
                filtered_df['prompt'].str.lower().str.contains(
                    filter_prompts.lower())]

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

        # Disable drag-to-zoom by default (the user can still enable it in the modebar)
        fig.update_layout(dragmode=False)

        # Display a message about how many data points are hidden
        num_results_message = f'Showing {len(filtered_df)} of {len(df)} data points.'

        return fig, num_results_message

    # Display the Dash app inline in the notebook
    # TODO: This doesn't seem to display inline if you click "Run All" in VSCode
    # instead of running the cell directly
    app.run(jupyter_mode='inline')


def _scatter_two_eval_values(eval_value: EvalValue,
                             other_eval_value: EvalValue) -> None:
    '''Shows an interactive scatter plot of all data points in two EvalValues.
    Intended to be used in a Jupyter notebook.
    '''
    # Validate that the two EvalValues have the same data points
    if eval_value.generated_outputs != other_eval_value.generated_outputs:
        raise ValueError(
            'Both EvalValues must have the same generated_outputs')
    if eval_value.prompts != other_eval_value.prompts:
        raise ValueError('Both EvalValues must have the same prompts')
    if eval_value.reference_outputs != other_eval_value.reference_outputs:
        raise ValueError(
            'Both EvalValues must have the same reference_outputs')
    if eval_value.language != other_eval_value.language:
        raise ValueError('Both EvalValues must have the same language')
    if eval_value.metric_name == other_eval_value.metric_name:
        raise ValueError('Both EvalValues must have different metric_names')

    # Rename some EvalValue fields for display
    df = eval_value.to_df()
    df.rename(columns={'metric_value': eval_value.metric_name}, inplace=True)
    df[other_eval_value.metric_name] = other_eval_value.to_df()['metric_value']
    df['prompt'] = df['prompt'].fillna('None')

    # Define layout of the Dash app (chart + search boxes)
    app = Dash(__name__)
    app.layout = html.Div([
        html.Div([
            html.Label('Filter generated_outputs: ',
                       style={'background-color': 'white'}),
            dcc.Input(id='filter_generated_outputs',
                      type='text',
                      placeholder='Type to search...'),
        ]),
        html.Div([
            html.Label('Filter prompts: ', style={'background-color':
                                                  'white'}),
            dcc.Input(id='filter_prompts',
                      type='text',
                      placeholder='Type to search...'),
        ]),
        html.Div([
            html.Span(id='num_results_message',
                      style={
                          'background-color': 'white',
                          'font-style': 'italic'
                      })
        ]),
        dcc.Graph(id='scatter_plot',
                  config={
                      'displaylogo': False,
                      'modeBarButtonsToRemove':
                      ['select', 'lasso2d', 'resetScale']
                  })
    ])

    # This function gets called whenever the user types in the search boxes
    @app.callback(Output('scatter_plot', 'figure'),
                  Output('num_results_message', 'children'),
                  Input('filter_generated_outputs', 'value'),
                  Input('filter_prompts', 'value'))
    def update_figure(filter_generated_outputs, filter_prompts):
        # Filter data points based on search boxes, case-insensitive
        filtered_df = df.copy()
        if filter_generated_outputs:
            filtered_df = filtered_df[
                filtered_df['generated_output'].str.lower().str.contains(
                    filter_generated_outputs.lower())]
        if filter_prompts:
            filtered_df = filtered_df[
                filtered_df['prompt'].str.lower().str.contains(
                    filter_prompts.lower())]

        # Configure the actual scatter plot
        hover_data = {'index': filtered_df.index}  # #xplicitly add the index
        hover_data.update({col: True for col in filtered_df.columns})
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

        # Disable drag-to-zoom by default (the user can still enable it in the modebar)
        fig.update_layout(dragmode=False)

        # Display a message about how many data points are hidden
        num_results_message = f'Showing {len(filtered_df)} of {len(df)} data points.'

        return fig, num_results_message

    # Display the Dash app inline in the notebook
    # TODO: This doesn't seem to display inline if you click "Run All" in VSCode
    # instead of running the cell directly
    app.run(jupyter_mode='inline')
