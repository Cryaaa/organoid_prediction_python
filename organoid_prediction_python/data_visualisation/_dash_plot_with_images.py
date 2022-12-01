import pandas as pd
from jupyter_dash import JupyterDash
import io
import base64
import pickle
import seaborn as sns
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go

from PIL import Image
import numpy as np

def get_dash_app_3D_scatter_hover_images(
    dataframe:pd.DataFrame,
    plot_keys:list, 
    hue:str,
    images:np.ndarray
):

    # Helper functions
    def np_image_to_base64(im_matrix):
        im = Image.fromarray(im_matrix)
        buffer = io.BytesIO()
        im.save(buffer, format="jpeg")
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
        im_url = "data:image/jpeg;base64, " + encoded_image
        return im_url

    # Color for each digit
    color_map = list(sns.color_palette("tab10").as_hex())
    labels = dataframe[hue]
    mapping = {value:integer for integer,value in enumerate(dataframe[hue].unique())}
    colors = [color_map[mapping[label]] for label in labels]
    x,y,z = [dataframe[key].to_numpy() for key in plot_keys]

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=6,
            color=colors,
        )
    )])

    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

    app = JupyterDash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
        ],
    )

    @app.callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("graph-5", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        im_matrix = images[num]
        im_url = np_image_to_base64(im_matrix)
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "350px", 'display': 'block', 'margin': '0 auto'},
                ),
                html.P("", style={'font-weight': 'bold'})
            ])
        ]

        return True, bbox, children

    return app
