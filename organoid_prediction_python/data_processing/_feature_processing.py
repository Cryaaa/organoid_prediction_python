import numpy as np
import pandas as pd
from .._utils import _try_dropping
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler, RobustScaler

def correlation_filter(dataframe:pd.DataFrame, threshold:float = 0.95) -> pd.DataFrame:
    """
    Feature filter which removes features correlating above threshold (measured with 
    Pearson correlation).

    Parameters
    ----------
    dataframe: pandas DataFrame
        Input dataframe with features to be filtered
    threshold: str
        threshold above which features are determined as highly correlated. Must be
        in range 0-1.
    """
    cor_matrix = dataframe.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= threshold)]
    uncorrelating = dataframe.drop(to_drop, axis=1)

    return uncorrelating

def fraction_measurement(
    df1:pd.DataFrame,
    df2:pd.DataFrame,
    key, 
    name,
) -> pd.Series:
    """
    Measures the fraction between two columns of a dataframe. key determines the column 
    and the result is a pandas series with rows equivalent to measurement(df1)/measurement(df2)

    Parameters
    ----------
    df1: pandas DataFrame
        Input dataframe 1 (will be in the numerator of fraction)
    df1: pandas DataFrame
        Input dataframe 2 (will be in the denominator of fraction)
    key: str
        Will determine the column for measurements
    suffix: str
        Suffix appended to the key for naming the pandas Series        
    """
    mapp={key:f"{key}_2"}
    renamed_df2 = df2.rename(columns=mapp)
    concat= pd.concat([df1,renamed_df2],axis=1)
    concat.dropna(inplace=True)
    index = concat.index

    series1 = concat[key].to_numpy()
    series2 = concat[mapp[key]].to_numpy()
    difference = [el1/el2 for el1,el2 in zip(series1,series2)]

    return pd.Series(data=difference,index=index,name=name)

# TODO docstring
def differential_standard_scaling(
    dataframe, 
    columns_regular, 
    columns_by_index, 
    columns_unscaled,
    columns_group_scaling,
    by_index_grouping = ["Run","Plate"],
    
):

    def group_scaling(df):
        matrix = df.to_numpy()
        mean = np.nanmean(matrix)
        stdd = np.nanstd(matrix)
        return (matrix-mean)/stdd
        
    df_no_scaling = dataframe[columns_unscaled]
    df_reg_scaling = dataframe[columns_regular]
    df_index_scaling = dataframe[columns_by_index]
    df_group_scaling = dataframe[columns_group_scaling]

    df_reg_scaling = pd.DataFrame(
        StandardScaler().fit_transform(df_reg_scaling), 
        columns=columns_regular,
        index = df_reg_scaling.index
    )
    
    df_index_scaling = standardscale_per_plate(
        df_index_scaling,
        grouping_keys=by_index_grouping
    )
    
    df_group_scaling = pd.DataFrame(
        group_scaling(df_group_scaling),
        columns=columns_group_scaling,
        index=df_group_scaling.index
    )

    return pd.concat([df_no_scaling,df_reg_scaling,df_index_scaling,df_group_scaling],axis=1)

def standardscale_per_plate(
    dataframe: pd.DataFrame, 
    grouping_keys: list =["Run", "Plate"],
) -> pd.DataFrame:
    """
    Function that performs standard-scaling / z-normalisation in subgroups like 
    plates of samples. The grouping_keys specify the index levels which will be 
    grouped before being z-normalised / standard-scaled. 

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input DataFrame that will be processed. Must be a multi-index DataFrame
    grouping_keys: list
        list of strings that are part of the multiindex which will be used for
        grouping
    """
    data = _try_dropping(dataframe)
    grouped = data.groupby(grouping_keys)
    transformed: pd.DataFrame =  grouped.transform(
            lambda x: (x - x.mean()) / x.std()
        )
    transformed.dropna(axis=1,inplace = True, thresh=1)

    return transformed

def percentile_scale_per_plate(
    dataframe: pd.DataFrame,
    percentile = 1, 
    grouping_keys: list =["Run", "Plate"],
) -> pd.DataFrame:
    """
    Function that performs standard-scaling / z-normalisation in subgroups like 
    plates of samples. The grouping_keys specify the index levels which will be 
    grouped before being z-normalised / standard-scaled. 

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input DataFrame that will be processed. Must be a multi-index DataFrame
    grouping_keys: list
        list of strings that are part of the multiindex which will be used for
        grouping
    """
    data = _try_dropping(dataframe)
    grouped = data.groupby(grouping_keys)
    scaler = RobustScaler(with_centering=False,quantile_range=(percentile,100-percentile))
    transformed: pd.DataFrame =  grouped.transform(
            lambda x: scaler.fit_transform(x.to_numpy().reshape(-1, 1)).reshape(1, -1)[0]
        )
    transformed.dropna(axis=1,inplace = True, thresh=1)

    return transformed

def split_by_cellprofiler_category(
    dataframe: pd.DataFrame, 
    annotation_keys:list = ["Axes","Morph","Morph_Class","Run.1"],
    bra_prefix:str = "BRA_"
) -> dict:
    """
    Function that given a dataframe containing properties from cellprofiler
    returns a list of dataframes split by cellprofiler categories. Any columns 
    that do not match the cellprofiler categories are dropped. 

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe containing cellprofiler data
    annotation_keys: list
        list of strings that match the columns in the dataframe containing
        manual annotations
    bra_prefix: str
        String specifying a prefix which might be peresent in front of the 
        cellprofiler column names
    """
    output = {}
    annotation_keys_present = [key for key in dataframe.keys() if key in annotation_keys]
    if len(annotation_keys_present) > 0:
        annotation_df = dataframe[annotation_keys_present]
    
    prefixes = ["AreaShape","Granularity","Intensity","Texture","RadialDistribution"]
    for prefix in prefixes:
        keys = [key for key in dataframe.keys() if key.startswith(prefix) or key.startswith(f"{bra_prefix}{prefix}")]

        if len(annotation_keys_present) > 0:
            df_sub = pd.concat([dataframe[keys],annotation_df],axis=1)
        else:
            df_sub = dataframe[keys]

        output[prefix] = df_sub
    return output


# TODO docstring or decide if I need to keep this
def reform_cellprofiler_table(
    dataframe: pd.DataFrame,
    cellprofiler_useless_columns:list = [
        "ImageNumber","Metadata_FileLocation","Metadata_Frame",
        "Metadata_Series","Metadata_Channel",'AreaShape_NormalizedMoment_0_0', 
        'AreaShape_NormalizedMoment_0_1', 'AreaShape_NormalizedMoment_1_0',
    ],
    spit_out = False,
) -> None:
    """
    
    """
    useless_keys = []
    for key in dataframe.keys():
        if key.startswith("FileName") or key.startswith("PathName") or key in cellprofiler_useless_columns:
            useless_keys.append(key)

    mapper = {
        "Metadata_Plate":"Plate",
        "Metadata_ID":"ID",
        "Metadata_Well":"ID",
        "Metadata_Run":"Run",
    }
    dataframe.rename(columns=mapper,inplace=True)
    dataframe.drop(useless_keys,axis = 1 , inplace=True)

    if spit_out:
        return dataframe

# TODO docstring or decide if to keep this
def distance_series(
    dataframe1: pd.DataFrame,
    dataframe2: pd.DataFrame, 
    column_name = "Distance_BRA_Center",
) -> pd.Series:
    """
    
    """
    location_keys = [
        'Location_Center_X',
        'Location_Center_Y'
    ]
    rename_mapping = {key:f"{key}_2" for key in location_keys}
    subset_df1: pd.DataFrame = dataframe1[location_keys]
    subset_df2: pd.DataFrame = dataframe2[location_keys]
    subset_df2.rename(columns=rename_mapping,inplace=True)
    
    df_both = pd.concat([subset_df1,subset_df2], axis=1)
    df_both.dropna(inplace=True)
    
    index = df_both.index
    
    all_location_columns = [df_both[key].to_numpy() for key in df_both.keys()]
    xy_xy_zipped = zip(*all_location_columns)
    distances = [euclidean([x,y], [x2,y2]) for x,y,x2,y2 in xy_xy_zipped]
                 
    return pd.Series(data=distances,index=index,name=column_name)

# TODO docstring and decide if to keep this
def correlation_filter_per_category_and_source(dataframe, correlation_threshold = 0.95):
    BF_keys = [key for key in dataframe.keys() if not key.startswith("BRA")]
    BRA_keys = [key for key in dataframe.keys() if key.startswith("BRA") and "Fraction" not in key]
    Frac_keys = [key for key in dataframe.keys() if key.startswith("BRA") and "Fraction" in key]
    source_keys = [BF_keys,BRA_keys,Frac_keys]

    splits = [split_by_cellprofiler_category(dataframe[keys]) for keys in source_keys]
    to_concat = []
    for split in splits:
        for k, df in split.items():
            to_concat.append(correlation_filter(df,correlation_threshold))

    return pd.concat(to_concat,axis = 1)
