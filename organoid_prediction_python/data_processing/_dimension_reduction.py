import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA
from .._utils import _try_dropping, heatmap_coloring_func
from functools import partial

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
    containing the ground truth specified by ground_truth_df. Both 
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

    Returns:
        pd.DataFrame: A new DataFrame containing the UMAP features and ground 
        truth labels (if provided). If ground_truth_df is None, returns only 
        the UMAP features.
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
   
def PCA_with_indices_and_ground_truth(
    dataframe: pd.DataFrame,
    ground_truth_df: pd.DataFrame = None,
    gt_keys:list = ["Axes","Morph","Morph_Class"],
    n_components=2,
    remove_unclassified = True,
    standardscale = False,
    return_transformer = False,
    randomstate = None,
):
    """
    This function performs PCA on the input DataFrame and returns a new DataFrame 
    with the specified number of components. If ground truth labels are provided, 
    they will be included in the output DataFrame. The function also provides an 
    option to remove unclassified samples from the input DataFrame and the ground 
    truth DataFrame.

    Parameters
    ----------
    dataframe: (pd.DataFrame)
        The input DataFrame to perform PCA on.
    ground_truth_df: pd.DataFrame, optional
        A DataFrame containing the ground truth labels for the input DataFrame. 
        Defaults to None.
    gt_keys: list, optional
        A list of strings representing the columns of ground_truth_df to include 
        in the output. Defaults to ["Axes","Morph","Morph_Class"].
    n_components: int, optional
        The number of components to extract from the input DataFrame. 
        Defaults to 2.
    remove_unclassified: bool, optional
        A boolean value indicating whether to remove unclassified samples from 
        both the input DataFrame and the ground truth DataFrame. Defaults to True.

    Returns:
        pd.DataFrame: A new DataFrame containing the PCA components and ground 
        truth labels (if provided). If ground_truth_df is None, returns only 
        the PCA components DataFrame.
    """
    if ground_truth_df is not None:
        if remove_unclassified:
            dataframe = dataframe.loc[(ground_truth_df["Morph"]!="unclassified")&(ground_truth_df["Axes"]!="unclassified")]
            ground_truth_df = ground_truth_df.loc[(ground_truth_df["Morph"]!="unclassified")&(ground_truth_df["Axes"]!="unclassified")]
        

    
    index = dataframe.index
    data = _try_dropping(dataframe)
    if standardscale:
        data = StandardScaler().fit_transform(data)
    transformer = PCA(n_components=n_components,random_state=randomstate)
    embedding = transformer.fit_transform(data)
    pca_df = pd.DataFrame(
        embedding,
        index=index,
        columns=[f"PC_{i+1}" for i in range(n_components)],
    )

    if ground_truth_df is None:
        if return_transformer:
            return pca_df, transformer
        return pca_df
    if return_transformer:
        return pd.concat([pca_df,ground_truth_df[gt_keys]],axis=1).dropna(), transformer
    return pd.concat([pca_df,ground_truth_df[gt_keys]],axis=1).dropna()

def sparse_PCA_with_indices_and_ground_truth(
    dataframe: pd.DataFrame,
    ground_truth_df: pd.DataFrame = None,
    gt_keys:list = ["Axes","Morph","Morph_Class"],
    n_components: int = 2,
    remove_unclassified: bool = True,
    standardscale: bool = False,
    return_transformer: bool = False,
    sparseness: float = 0.7,
    randomstate = None,
):
    """
    Perform sparse PCA on the input DataFrame, with optional ground truth labels.
    
    Parameters
    ----------
    dataframe: pd.DataFrame
        The input DataFrame to perform PCA on.
    ground_truth_df: pd.DataFrame, optional
        A DataFrame containing the ground truth labels for the input DataFrame. 
        Defaults to None.
    gt_keys: list, optional
        A list of strings representing the columns of ground_truth_df to include 
        in the output. Defaults to ["Axes","Morph","Morph_Class"].
    n_components: int, optional
        The number of components to extract from the input DataFrame. 
        Defaults to 2.
    remove_unclassified: bool, optional
        A boolean value indicating whether to remove unclassified samples from 
        both the input DataFrame and the ground truth DataFrame. Defaults to True.
    standardscale: bool, optional
        A boolean value indicating whether to standard scale the input DataFrame. 
        Defaults to False.
    return_transformer: bool, optional
        A boolean value indicating whether to return the transformer object. 
        Defaults to False.
    sparseness: float, optional
        The sparseness parameter for the SparsePCA algorithm. Defaults to 0.7.
    randomstate: int, optional
        The random state for the SparsePCA algorithm. Defaults to None.
    
    Returns
    -------
    pd.DataFrame
        A new DataFrame containing the PCA components and ground truth labels (if provided). 
        If ground_truth_df is None, returns only the PCA components DataFrame.
    """
    if ground_truth_df is not None:
        if remove_unclassified:
            dataframe = dataframe.loc[(ground_truth_df["Morph"]!="unclassified")&(ground_truth_df["Axes"]!="unclassified")]
            ground_truth_df = ground_truth_df.loc[(ground_truth_df["Morph"]!="unclassified")&(ground_truth_df["Axes"]!="unclassified")]
        
    index = dataframe.index
    data = _try_dropping(dataframe)
    if standardscale:
        data = StandardScaler().fit_transform(data)
    transformer = SparsePCA(n_components=n_components,alpha=sparseness, n_jobs=4, max_iter=5000,random_state=randomstate)
    embedding = transformer.fit_transform(data)
    pca_df = pd.DataFrame(
        embedding,
        index=index,
        columns=[f"PC_{i+1}" for i in range(n_components)],
    )

    if ground_truth_df is None:
        if return_transformer:
            return pca_df, transformer
        return pca_df
    if return_transformer:
        return pd.concat([pca_df,ground_truth_df[gt_keys]],axis=1).dropna(), transformer
    return pd.concat([pca_df,ground_truth_df[gt_keys]],axis=1).dropna()


def transformer_loading_dataframe(transformer,input_dataframe,n_components=2, loading_bounds = (-1,0,1),cmap="vlag"):
    """
    Generate a heatmap of the loading values of the input_dataframe onto the transformer components.
    
    Parameters
    ----------
    transformer: sklearn.decomposition.SparsePCA
        The transformer object.
    input_dataframe: pd.DataFrame
        The input DataFrame.
    n_components: int, optional
        The number of components to extract from the input DataFrame. 
        Defaults to 2.
    loading_bounds: tuple of floats, optional
        The bounds for the heatmap. Defaults to (-1,0,1).
    cmap: str, optional
        The colormap to use. Defaults to "vlag".
    
    Returns
    -------
    pd.DataFrame
        The styled DataFrame.
    """
    transformerloadings = pd.DataFrame(
        transformer.components_.T, 
        columns=[f'PC_{i+1}' for i in range(n_components)], 
        index=input_dataframe.columns
    )
    out = transformerloadings.style.applymap(
        partial(heatmap_coloring_func,**{"cmap":cmap}),data_bounds = loading_bounds
    )
    return out
