import numpy as np
import pandas as pd
from .._utils import _try_dropping

def correlation_filter(dataframe:pd.DataFrame, threshold:float = 0.95) -> pd.DataFrame:
    cor_matrix = dataframe.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    uncorrelating = dataframe.drop(to_drop, axis=1)

    return uncorrelating

def split_by_cellprofiler_category(dataframe: pd.DataFrame) -> dict:
    output = {}
    prefixes = ["AreaShape","Granularity","Intensity","Texture","RadialDistribution"]
    for prefix in prefixes:
        keys = [key for key in dataframe.keys() if key.startswith(prefix)]
        output[prefix] = dataframe[keys]
    return output
    
def standardscale_per_plate(dataframe: pd.DataFrame, grouping_keys: list =["Run", "Plate"],) -> pd.DataFrame:
    data = _try_dropping(dataframe)
    grouped = data.groupby(grouping_keys)
    transformed: pd.DataFrame =  grouped.transform(
            lambda x: (x - x.mean()) / x.std()
        )
    transformed.dropna(axis=1,inplace = True)

    return transformed

def reform_cellprofiler_table(
    dataframe: pd.DataFrame,
    cellprofiler_useless_columns:list = [
        "ImageNumber","Metadata_FileLocation","Metadata_Frame",
        "Metadata_Series","Metadata_Channel",'AreaShape_NormalizedMoment_0_0', 
        'AreaShape_NormalizedMoment_0_1', 'AreaShape_NormalizedMoment_1_0',
    ]
) -> None:
    useless_keys = []
    for key in dataframe.keys():
        if key.startswith("FileName") or key.startswith("PathName") or key in cellprofiler_useless_columns:
            useless_keys.append(key)

    mapper = {
        "Metadata_Plate":"Plate",
        "Metadata_Well":"ID",
        "Metadata_Run":"Run"
    }
    dataframe.rename(columns=mapper,inplace=True)
    dataframe.drop(useless_keys,axis = 1 , inplace=True)
