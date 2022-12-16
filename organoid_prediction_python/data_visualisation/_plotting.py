import plotly.express as px
from seaborn import color_palette
from pandas import DataFrame
import numpy as np

def plotly3d(df:DataFrame, hue:str, xyz_columns:list,alpha = 0.7):
    color_map = color_palette("tab10").as_hex()
    labels = df[hue].to_numpy()
    mapping = {
        value:color for value,color 
        in zip(np.unique(labels),color_map)
    }

    fig = px.scatter_3d(
        df, 
        x=xyz_columns[0], 
        y=xyz_columns[1], 
        z=xyz_columns[2],
        color_discrete_map=mapping,
        color=hue,
        opacity=alpha
    )
    
    fig.update_layout(
        autosize=False,
        width=1500,
        height=800,)

    fig.update_traces(marker=dict(size=5))

    fig.show()