import pandas as pd
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
