import re
import pandas as pd
import numpy as np

def extract_sample_identifiers(
    filenames: list,
    regex: list = r".*FK223_run_(?P<Run>[A-Z]+)_PLATE_(?P<Plate>[0-9]+)_ID_(?P<ID>[A-Z][0-9]+)_.*",
    regex_column_names:list = ["Run","Plate","ID"], return_original_name = False
) -> pd.DataFrame:
    """
    Function which based on a filelist and regular expression returns a dataframe with
    columns for the names extracted with the regular expression

    Parameters
    ----------
    filenames: str
        list containing the filenames
    regex: str
        regular expression with which to extract information from the filenames
    regex_column_names: list(str)
        A list of the names for the different columns extracted with the regular
        expression. MUST match the names used in the regular expression
    """
    p = re.compile(regex)
    raw_columns = {name : [] for name in regex_column_names}
    raw_columns["Filename"] = []
    if return_original_name:
        raw_columns["Original Filename"] =[]
    for filename in filenames:
        m = p.search(filename)
        name_frags = [f"{k}_{v}_" for k,v in m.groupdict().items()]
        name = ""
        for frag in name_frags:
            name += frag
        name = name[:-1]
        raw_columns["Filename"].append(name)
        if return_original_name:
            raw_columns["Original Filename"].append(filename)
        for  regex_name in regex_column_names:
            raw_columns[regex_name].append(m.group(regex_name))
        
    if "Plate" in regex_column_names:
        raw_columns["Plate"] = np.array(raw_columns["Plate"]).astype(int)
        
    return pd.DataFrame(raw_columns)

def stack_time_data(dataframe, hours = ["48","72","96"]):
    """
    Stack dataframes with different timepoints into one dataframe.
    
    Parameters
    ----------
    dataframe: pd.DataFrame
        The dataframe to stack.
    hours: list of str
        The hours to stack. The column names of the dataframe should start with the hour.
        For example, if hours=["48","72","96"], the columns should be ["48_col1","48_col2",...,"96_colN"].
    
    Returns
    -------
    pd.DataFrame
        The stacked dataframe.
    """
    time_keys = [[key for key in dataframe.keys() if key[1:].startswith(str(hour))]+["Run","Plate","ID"] for hour in hours]
    dataframes_separated = [dataframe[key_list].rename(columns={key:key[5:] for key in key_list if key not in ["Run","Plate","ID"]}) for key_list in time_keys]
    
    out = []
    for frame,hour in zip(dataframes_separated,hours):
        frame["Hour"] = np.full(len(frame),hour).astype(int)
        out.append(frame)
        
    return pd.concat(out,axis=0,ignore_index=True)

