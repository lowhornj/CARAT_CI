import numpy as np
from scipy import stats
import polars as pl

def ranged_scaler(x,a=0,b=1):
    col_name = x.name
    x = x.to_numpy()
    x_prime = a + (((x - np.nanmin(x)) * (b-a)) / (np.nanmax(x) - np.nanmin(x)))
    x_prime = pl.from_numpy(x_prime,schema=[col_name],orient='col')
    return x_prime
