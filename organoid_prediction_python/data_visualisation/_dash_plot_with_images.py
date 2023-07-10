import pandas as pd
from jupyter_dash import JupyterDash
import io
import base64
import seaborn as sns
from dash import dcc, html, Input, Output, no_update
import plotly.graph_objects as go

from PIL import Image
import numpy as np
from matplotlib.colors import to_hex

# code taken and modified from: https://dash.plotly.com/dash-core-components/tooltip?_gl=1*9tyg7p*_ga*NDYwMzcxMTAxLjE2Njk3MzgyODM.*_ga_6G7EE0JNSC*MTY3MzI2ODgyOS45LjEuMTY3MzI2OTA0Ni4wLjAuMA..
# TODO put license here
def get_dash_app_3D_scatter_hover_images(
    dataframe:pd.DataFrame,
    plot_keys:list, 
    hue:str,
    images:np.ndarray,
    additional_info: str = "",
):
    """
    The get_dash_app_3D_scatter_hover_images() function creates a Dash app that displays a 3D 
    scatter plot with hover information for each data point. The hover information consists of 
    an image and a label associated with the data point. The image is retrieved from an array 
    of images passed to the function.
    
    Parameters
    ----------
    dataframe: pd.DataFrame
        A Pandas DataFrame containing the data to be plotted.
    plot_keys: list 
        A list of column names in the dataframe that represent the x, y, and z coordinates of 
        the data points.
    hue: str
        A string representing the column name in the dataframe that contains the labels 
        associated with the data points.
    images: 
        A numpy array containing the images to be displayed in the hover information.

    Returns:
        app: a Dash app object representing the 3D scatter plot with hover information.
    """
    # Create a color map for each categorical value and assigns a color to each data 
    # point based on its category. It then extracts the x, y, and z data from the 
    # input DataFrame, and uses them to create a 3D scatter plot using the 
    # plotly.graph_objects library.
    
    labels = dataframe[hue].to_numpy()
    if labels.dtype.name == 'object':
        color_map = list(sns.color_palette("tab10").as_hex())
        mapping = {value:integer for integer,value in enumerate(np.unique(labels))}
        colors = [color_map[mapping[label]] for label in labels]
    else:
        color_map = sns.color_palette("rocket",as_cmap=True)
        scaled = np.array((labels - labels.min()) / (labels.max()-labels.min()))
        colors = [to_hex(color_map(val)) for val in scaled]
    
    add_info = ["" for i in range(len(dataframe))]
    if additional_info != "":
        add_info = dataframe[additional_info].to_numpy()

    
    x,y,z = [dataframe[key].to_numpy() for key in plot_keys]

    # Make the plot. 
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        opacity=0.7,
        marker=dict(
            size=5,
            color=colors,
        )
    )])

    # The plot's hover information is set to "none" and its hover template is set 
    # to None to prevent default hover information from being displayed. The plot's 
    # layout is set to fixed dimensions of 1500x800 pixels.
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

    fig.update_layout(
        autosize=False,
        width=1500,
        height=800,)


    # Definition of a JupyterDash application and creates a layout 
    # consisting of a dcc.Graph component for the 3D scatter plot and a dcc.Tooltip 
    # component for the hover information.
    app = JupyterDash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
        ],
    )

    # Definition of a callback function that listens for hover events on the 3D scatter 
    # plot and returns the appropriate hover information. When a data point is hovered 
    # over, the callback extracts the point's index and image from the input images array, 
    # converts the image to a base64 encoded string using the np_image_to_base64 helper 
    # function, and returns a html.Div containing the image and the category label of 
    # the hovered data point.
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
                    src=im_url, style={"width": "100%"},
                ),
                html.P(hue + ": " + str(labels[num]), style={'font-weight': 'bold'}),
                html.P(additional_info + ": " + str(add_info[num]), style={'font-weight': 'bold'})
            ], style={'width': '200px', 'white-space': 'normal'})
        ]

        return True, bbox, children

    return app

def get_dash_app_2D_scatter_hover_images(
    dataframe:pd.DataFrame,
    plot_keys:list, 
    hue:str,
    images:np.ndarray,
    additional_info: str = "",
):
    """
    The get_dash_app_3D_scatter_hover_images() function creates a Dash app that displays a 3D 
    scatter plot with hover information for each data point. The hover information consists of 
    an image and a label associated with the data point. The image is retrieved from an array 
    of images passed to the function.
    
    Parameters
    ----------
    dataframe: pd.DataFrame
        A Pandas DataFrame containing the data to be plotted.
    plot_keys: list 
        A list of column names in the dataframe that represent the x, y, and z coordinates of 
        the data points.
    hue: str
        A string representing the column name in the dataframe that contains the labels 
        associated with the data points.
    images: 
        A numpy array containing the images to be displayed in the hover information.

    Returns:
        app: a Dash app object representing the 3D scatter plot with hover information.
    """
    # Create a color map for each categorical value and assigns a color to each data 
    # point based on its category. It then extracts the x, y, and z data from the 
    # input DataFrame, and uses them to create a 3D scatter plot using the 
    # plotly.graph_objects library.

    labels = dataframe[hue].to_numpy()
    if labels.dtype.name == 'object':
        color_map = list(sns.color_palette("tab10").as_hex())
        mapping = {value:integer for integer,value in enumerate(np.unique(labels))}
        colors = [color_map[mapping[label]] for label in labels]
    else:
        color_map = sns.color_palette("rocket",as_cmap=True)
        scaled = np.array((labels - labels.min()) / (labels.max()-labels.min()))
        colors = [to_hex(color_map(val)) for val in scaled]
    
    add_info = ["" for i in range(len(dataframe))]
    if additional_info != "":
        add_info = dataframe[additional_info].to_numpy()
    
    x,y = [dataframe[key].to_numpy() for key in plot_keys]

    # Make the plot. 
    fig = go.Figure(data=[go.Scatter(
        x=x,
        y=y,
        mode='markers',
        opacity=0.7,
        marker=dict(
            size=5,
            color=colors,
        )
    )])

    # The plot's hover information is set to "none" and its hover template is set 
    # to None to prevent default hover information from being displayed. The plot's 
    # layout is set to fixed dimensions of 1500x800 pixels.
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

    fig.update_layout(
        autosize=False,
        width=1500,
        height=800,)


    # Definition of a JupyterDash application and creates a layout 
    # consisting of a dcc.Graph component for the 3D scatter plot and a dcc.Tooltip 
    # component for the hover information.
    app = JupyterDash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
        ],
    )

    # Definition of a callback function that listens for hover events on the 3D scatter 
    # plot and returns the appropriate hover information. When a data point is hovered 
    # over, the callback extracts the point's index and image from the input images array, 
    # converts the image to a base64 encoded string using the np_image_to_base64 helper 
    # function, and returns a html.Div containing the image and the category label of 
    # the hovered data point.
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
                    src=im_url, style={"width": "100%"},
                ),
                html.P(hue + ": " + str(labels[num]), style={'font-weight': 'bold'}),
                html.P(additional_info + ": " + str(add_info[num]), style={'font-weight': 'bold'})
            ], style={'width': '200px', 'white-space': 'normal'})
        ]

        return True, bbox, children

    return app

# Definition of a nested helper function np_image_to_base64 that converts numpy 
# arrays of images into base64 encoded strings for display in HTML.
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url
