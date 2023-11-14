import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import to_hex
import numpy as np
import re
def _try_dropping(
    dataframe: pd.DataFrame,
    key_list:list=[
        "ObjectNumber","Plate.1",
        "Run.1","ID.1","Axes",
        "Morph","Morph_Class",
        "Plate","Run","ID",
    ],
):
    df_copy = dataframe.copy() 
    for key in key_list:
        try:
            df_copy.drop(key,axis = 1,inplace=True)
        except KeyError:
            pass
    return df_copy

def display_hex_colors(hex_colors):
    def apply_formatting(col, hex_colors):
        for hex_color in hex_colors:
            if col.name == hex_color:
                return [f'background-color: {hex_color}' for c in col.values]
    df = pd.DataFrame(hex_colors).T
    df.columns = hex_colors
    df.iloc[0,0:len(hex_colors)] = ""
    display(df.style.apply(lambda x: apply_formatting(x, hex_colors)))

    
def heatmap_coloring_func(value,data_bounds = (-1,0,1),cmap = "bwr"):
    col_map = colormaps[cmap]
    hex_color = to_hex(col_map(np.interp(value,data_bounds,(0,0.5,1))))
    return f"background: {hex_color};"

def natural_key(string_):
    """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]