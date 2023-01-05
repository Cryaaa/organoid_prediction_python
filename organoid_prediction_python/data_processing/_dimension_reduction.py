import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .._utils import _try_dropping

def umap_with_indices_and_ground_truth(
    dataframe: pd.DataFrame,
    ground_truth_df: pd.DataFrame = None,
    gt_keys: list = ["Axes","Morph","Morph_Class"],
    standardscale: bool = True,
    n_neighbors: int = 10,
    n_components: int = 2,
    random_state: int = None,
):
    """
    Generates a UMAP from a dataframe and returns it as a dataframe
    containing the grounbd truth specified by ground_truth_df. Both 
    dataframes need to have matching indices.

    Parameters
    ----------

    dataframe: pd.DataFrame
        Input dataframe from which the UMAP is generated.
    ground_truth_df: pd.Dataframe
        Dataframe containing the ground truth that will be added to the
        output dataframe
    gt_keys: list
        List of strings specifying which columns from the groound_truth_df
        will be added to the output dataframe.
    standardscale: bool
        If True standardscaling / z-normalisation will be performed on the 
        data prior to UMAP generation
    n_neighbors: int
        Parameter specifying the number 0of neighbors to regard during 
        UMAP generation (see UMAP documentation)
    n_components: int
        Number of dimensions the UMAP will have.
    random_state: int
        Random seed which can be used to get the same result twice
    """
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
    if ground_truth_df is None:
        return umap_df
    return pd.concat([umap_df,ground_truth_df[gt_keys]],axis=1).dropna()    

# TODO docstring
def PCA_with_indices_and_ground_truth(
    dataframe: pd.DataFrame,
    ground_truth_df: pd.DataFrame = None,
    gt_keys:list = ["Axes","Morph","Morph_Class"],
    n_components=2,
    remove_unclassified = True,
):
    
    if ground_truth_df is not None:
        if remove_unclassified:
            dataframe = dataframe.loc[(ground_truth_df["Morph"]!="unclassified")&(ground_truth_df["Axes"]!="unclassified")]
        ground_truth_df = ground_truth_df.loc[(ground_truth_df["Morph"]!="unclassified")&(ground_truth_df["Axes"]!="unclassified")]
        

    
    index = dataframe.index
    data = _try_dropping(dataframe)

    transformer = PCA(n_components=n_components,)
    embedding = transformer.fit_transform(data)
    pca_df = pd.DataFrame(
        embedding,
        index=index,
        columns=[f"PC_{i+1}" for i in range(n_components)],
    )

    if ground_truth_df is None:
        return pca_df
    return pd.concat([pca_df,ground_truth_df[gt_keys]],axis=1).dropna()

    
