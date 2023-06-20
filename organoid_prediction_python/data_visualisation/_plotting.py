import plotly.express as px
from seaborn import color_palette, stripplot
from matplotlib import colormaps, collections
from pandas import DataFrame
import numpy as np

def plotly3d(df:DataFrame, hue:str, xyz_columns:list,alpha = 0.7):
    """
    This function plots a 3D scatter plot using Plotly library with 
    the given dataset and columns.

    Parameters:
    -----------
    - df: pd.DataFrame
        The DataFrame containing the data to be plotted.
    hue: str
        The name of the column in df to be used for coloring the 
        data points.
    xyz_columns: list
        A list of 3 strings representing the names of the columns 
        in df to be used for the x, y, and z axes of the 3D plot, 
        respectively.
    alpha: float
        A float value representing the opacity of the data points. 
        Default is 0.7.

    Returns:
        None: The function only generates the plot and displays it.
    """
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

def raincloud_plot(df,x,y,ax,jitter = 0.3,palette = 'flare', quantiles = [0.25,0.75], size_cloud = 3, size_quant = 5):


    cmap = colormaps[palette]
    stripplot(
        x = x, y = y, data = df, palette = palette,
        size = size_cloud, jitter =jitter-0.05, zorder = 1,ax=ax,**{"alpha":0.7},
    )
    
    data_manual_x = [df[df[x] == feat][y] for feat in df[x].unique()]
    num_hues = len(df[x].unique())
    positions_violin = np.arange(num_hues)-jitter
    ax.plot(
        positions_violin,
        [data_manual_x[i].mean() for i in range(num_hues)],
        c="white",
        marker ="o",
        zorder=3,
        linewidth=0,
        markersize=size_quant/2
    )
    
    lines_quant = [
        [
            [positions_violin[i],data_manual_x[i].quantile(quantiles[0])],
            [positions_violin[i],data_manual_x[i].quantile(quantiles[1])]
        ] for i in range(num_hues)
    ]

    collection_quant=collections.LineCollection(
        lines_quant,
        linewidths = size_quant,
        color=np.clip(
            np.array([cmap(np.linspace(0,1,3)[i]) for i in range(num_hues)])-np.array([0.1,0.1,0.1,0])*3,
            0,
            1
        ),
        zorder=2,
    )
    collection_quant.set_capstyle("round")
    ax.add_collection(collection_quant)
    
    vp = ax.violinplot(data_manual_x,positions=positions_violin, 
               showmeans=False, showextrema=False, showmedians=False,vert=True
    )
    for idx, b in enumerate(vp['bodies']):
        lim = idx - jitter
        # Get the center of the plot
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        
        # Modify it so we only see the upper half of the violin plot
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], lim-1, lim)
        
        # Change to the desired color
        b.set_color(cmap(np.linspace(0,1,3)[idx]))
        b.set_alpha(0.6)
        b.set_zorder(1)