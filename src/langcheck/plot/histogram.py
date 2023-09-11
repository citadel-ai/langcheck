import plotly.express as px
from dash import Dash, Input, Output, dcc, html
from plotly.graph_objects import Figure, Histogram

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
            html.Label('Max number of bins: ',
                       style={'background-color': 'white'}),
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
        # TODO: nbins is just a suggestion and not enforced. Figure out how to
        # force it to use the exact number of bins the user selected.
        fig = px.histogram(df, nbins=int(num_bins), x=eval_value.metric_name)

        # If the user manually zoomed in, keep that zoom level even when
        # update_figure() re-runs
        fig.update_layout(uirevision='constant')

        # Disable drag-to-zoom by default (the user can still enable it in the modebar)
        fig.update_layout(dragmode=False)

        return fig

    # Display the Dash app inline in the notebook
    app.run(jupyter_mode='inline')
