import numpy as np
import pandas as pd

def correlation_filter(dataframe:pd.DataFrame, threshold:float = 0.95) -> pd.DataFrame:
    cor_matrix = dataframe.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    uncorrelating = dataframe.drop(to_drop, axis=1)

    return uncorrelating