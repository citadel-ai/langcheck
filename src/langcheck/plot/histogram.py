import plotly.express as px
from dash import Dash, Input, Output, dcc, html

from langcheck.eval import EvalValue


def histogram(eval_value: EvalValue) -> None:
    '''Shows an interactive histogram of all data points in EvalValue. Intended
    to be used in a Jupyter notebook.
    '''
    # Rename some EvalValue fields for display
    df = eval_value.to_df()
    df.rename(columns={'metric_value': eval_value.metric_name}, inplace=True)

    # Define layout of the Dash app (histogram + input for number of bins)
    app = Dash(__name__)
    app.layout = html.Div([
        html.Div([
            html.Label('Number of bins: ', style={'background-color':
                                                  'white'}),
            dcc.Input(id='num_bins', type='number', value=10),
        ]),
        dcc.Graph(id='histogram',
                  config={
                      'displaylogo': False,
                      'modeBarButtonsToRemove':
                      ['select', 'lasso2d', 'resetScale']
                  })
    ])

    # This function gets called whenever the user changes the num_bins value
    @app.callback(
        Output('histogram', 'figure'),
        Input('num_bins', 'value'),
    )
    def update_figure(num_bins):
        # Configure the actual histogram plot
        fig = px.histogram(df, nbins=int(num_bins), x=eval_value.metric_name)

        # If the user manually zoomed in, keep that zoom level even when
        # update_figure() re-runs
        fig.update_layout(uirevision='constant')

        # Disable drag-to-zoom by default (the user can still enable it in the modebar)
        fig.update_layout(dragmode=False)

        return fig

    # Display the Dash app inline in the notebook
    app.run(jupyter_mode='inline')
