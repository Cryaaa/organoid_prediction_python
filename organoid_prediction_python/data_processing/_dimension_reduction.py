import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from .._utils import _try_dropping

def umap_with_indices_and_ground_truth(
    dataframe: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    gt_keys:list = ["Axes","Morph","Morph_Class"],
    standardscale = True,
    n_neighbors=10,
    n_components=2,
    random_state = None,
):
    index = dataframe.index
    data = _try_dropping(dataframe)
    if standardscale:
        data = StandardScaler().fit_transform(data)

    transformer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        random_state=random_state,
    )
    embedding = transformer.fit_transform(data)
    umap_df = pd.DataFrame(
        embedding,
        index=index,
        columns=[f"UMAP_{i+1}" for i in range(n_components)],
    )

    return pd.concat([umap_df,ground_truth_df[gt_keys]],axis=1)


    
