import re
import pandas as pd
import numpy as np

def extract_sample_identifiers(
    filenames: list,
    regex: list = r".*FK223_run_(?P<Run>[A-Z]+)_PLATE_(?P<Plate>[0-9]+)_ID_(?P<ID>[A-Z][0-9]+)_.*",
    regex_column_names:list = ["Run","Plate","ID"],
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
    for filename in filenames:
        m = p.search(filename)
        name_frags = [f"{k}_{v}_" for k,v in m.groupdict().items()]
        name = ""
        for frag in name_frags:
            name += frag
        name = name[:-1]
        raw_columns["Filename"].append(name)
        for  regex_name in regex_column_names:
            raw_columns[regex_name].append(m.group(regex_name))
        
    if "Plate" in regex_column_names:
        raw_columns["Plate"] = np.array(raw_columns["Plate"]).astype(int)
        
    return pd.DataFrame(raw_columns)