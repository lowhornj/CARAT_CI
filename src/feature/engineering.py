import polars as pl
import numpy as np
from scipy import stats
import polars as pl
import math
from scipy.stats import t
from collections import defaultdict
import statsmodels.api as sm
from datetime import datetime
import pandas as pd

def optimal_lag(series,lags=5):
    """
    Finds the optimal lag for a time series based on autocorrelation.

    Args:
        series (pd.Series): The time series data.

    Returns:
        int: The optimal lag.
    """
    autocorrelation_values = sm.tsa.stattools.acf(series)
    significant_lags = np.where(np.abs(autocorrelation_values) > 1.96/np.sqrt(len(series)))[0]

    if len(significant_lags) > 1:
      return significant_lags[1] # Return the first significant lag (excluding lag 0)
    else:
      return np.array([0])

def find_optimal_lags_for_dataframe(df):
    """
    Finds the optimal lag for each column in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary where keys are column names and values are optimal lags.
    """
    optimal_lags = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            optimal_lags.append( optimal_lag(df[col]).item())
    return optimal_lags

from collections import Counter

def most_frequent(list_):
  """
    Finds the most frequent element in a list.

    Args:
      list_: The input list.

    Returns:
      The most frequent element in the list.
  """
  if not list_:
    return None
  count = Counter(list_)
  return count.most_common(1)[0][0]


def adf_test_polars(series: pl.Series, lags: int = 1, alpha=0.05):
    """
    Perform Augmented Dickey-Fuller (ADF) test in Polars.
    
    Args:
        series (pl.Series): Time series column.
        lags (int): Number of lagged differences to include.
        alpha (float): Significance level (default=0.05).
    
    Returns:
        (bool, dict): (Stationary or not, test details)
    """
    # Ensure the series is numeric
    if not series.dtype in [pl.Float32, pl.Float64]:
        return True, {"ADF Statistic": None, "p-value": None, "Critical Values": None, "Skipped": True}

    series = series.drop_nulls()  # Ensure no NaNs
    if series.len() < lags + 1:  # Not enough data
        return True, {"ADF Statistic": None, "p-value": None, "Critical Values": None}

    df = pl.DataFrame({"y": series}).with_columns(
        pl.col("y").shift(1).alias("y_lag"),
        (pl.col("y") - pl.col("y").shift(1)).alias("diff_y")
    ).drop_nulls()

    y_t = df["diff_y"]
    x_t = df["y_lag"]

    # Estimate γ in ΔY_t = γ * Y_{t-1} + error
    gamma_hat = (x_t * y_t).sum() / (x_t * x_t).sum()
    residuals = y_t - gamma_hat * x_t
    s_sq = (residuals * residuals).sum() / (len(residuals) - 1)
    
    # Compute ADF test statistic
    gamma_se = math.sqrt(s_sq / (x_t * x_t).sum())
    adf_statistic = gamma_hat / gamma_se

    # Compute p-value
    dfree = len(y_t) - lags - 1
    p_value = 2 * (1 - t.cdf(abs(adf_statistic), dfree))

    # Critical values (from Dickey-Fuller table)
    critical_values = {"1%": -3.43, "5%": -2.86, "10%": -2.57}

    # Check stationarity
    is_stationary = adf_statistic < critical_values["5%"]

    return is_stationary, {"ADF Statistic": adf_statistic, "p-value": p_value, "Critical Values": critical_values}


def make_stationary(df: pl.DataFrame, max_diff=5):
    """
    Applies differencing iteratively on each column in a Polars DataFrame until it's stationary.
    
    Args:
        df (pl.DataFrame): Input Polars DataFrame.
        max_diff (int): Max number of differencing attempts.
    
    Returns:
        pl.DataFrame: Differenced stationary DataFrame.
    """
    df_stationary = df.clone()  # Copy original DataFrame
    diff_counts = {col: 0 for col in df.columns}

    for col in df.columns:
        # Check if column is float, convert if necessary
        if df[col].dtype not in [pl.Float32, pl.Float64]:
            print(f"⚠️ Skipping column {col} (not a float column)")
            continue

        is_stationary, test_result = adf_test_polars(df[col])

        # Apply differencing until stationary or max_diff is reached
        while not is_stationary and diff_counts[col] < max_diff:
            df_stationary = df_stationary.with_columns(
                (pl.col(col) - pl.col(col).shift(1)).alias(col)
            ).drop_nulls()  # Remove NaNs after differencing
            diff_counts[col] += 1
            is_stationary, test_result = adf_test_polars(df_stationary[col])

        print(f"Column: {col} | Final ADF: {test_result['ADF Statistic']:.4f} | p-value: {test_result['p-value']:.4f} | Diffs: {diff_counts[col]}")

    return df_stationary


def ranged_scaler(x, a=-1, b=1):
    col_name = x.name
    x = x.to_numpy()
    x_prime = a + (((x - np.nanmin(x)) * (b-a))/ (np.nanmax(x) - np.nanmin(x)))
    x_prime = pl.from_numpy(x_prime, schema=[col_name], orient="col")
    return x_prime
def yeo_johnson_pt(x):
    col_name = x.name
    x = x.to_numpy()
    x_prime, lmbda = stats.yeojohnson(x)
    x_prime = pl.from_numpy(x_prime, schema=[col_name], orient="col")
    return x_prime

def drop_near_zero_variance(data, variance_threshold = 0.0001):
    variance_table = data.var()
    for var in variance_table.columns: 
        if variance_table[var].dtype.is_float() | variance_table[var].dtype.is_integer():
            if variance_table[var][0] <= variance_threshold:
                data = data.drop(var)
    return data

class feature_engineering:
    def __init__(self):
        super().__init__(self)
    @classmethod
    def fill_missing_time(self, enb_kpis,dateto,datefrom,cell_level):
        external_dates = pl.datetime_range(datetime.strptime(datefrom, '%Y%m%d'),datetime.strptime(dateto, '%Y%m%d'), "1h", eager=True)
        enb_kpis = enb_kpis.sort("Date_Time_Fin")
        enb_kpis=enb_kpis.set_sorted("Date_Time_Fin")
        if cell_level:
            if 'cell_carrier' not in enb_kpis.columns:
                cells = enb_kpis['EnodeB_Fin'].unique().to_list()
                out_kpi = pl.DataFrame()
                for cell in cells:
                    temp = enb_kpis.filter(pl.col('EnodeB_Fin') == cell)
                    temp = temp.sort("Date_Time_Fin")
                    temp=temp.set_sorted("Date_Time_Fin")
                    temp = temp.upsample(time_column="Date_Time_Fin", every='1h')
                    temp = temp.join(
                        pl.DataFrame({"Date_Time_Fin": external_dates}),
                        on="Date_Time_Fin",
                        how="outer"
                    )
                    temp = temp.with_columns([
                    pl.col("EnodeB_Fin").fill_null(strategy="forward")])
                    temp = temp.with_columns([
                    pl.col("EnodeB_Fin").fill_null(strategy="backward")])
                    out_kpi = out_kpi.vstack(temp)

            else:
                cells = enb_kpis['cell_carrier'].unique().to_list()
                out_kpi = pl.DataFrame()
                for cell in cells:
                    temp = enb_kpis.filter(pl.col('cell_carrier') == cell)
                    temp = temp.sort("Date_Time_Fin")
                    temp=temp.set_sorted("Date_Time_Fin")
                    temp = temp.upsample(time_column="Date_Time_Fin", every='1h')
                    temp = temp.join(
                        pl.DataFrame({"Date_Time_Fin": external_dates}),
                        on="Date_Time_Fin",
                        how="outer"
                    )
                    temp = temp.with_columns([
                    pl.col("cell_carrier").fill_null(strategy="forward"),
                    pl.col("EnodeB_Fin").fill_null(strategy="forward")
                        ])
                    temp = temp.with_columns([
                    pl.col("cell_carrier").fill_null(strategy="backward"),
                    pl.col("EnodeB_Fin").fill_null(strategy="backward")
                        ])
                    out_kpi = out_kpi.vstack(temp)


        else:
            enb_kpis = enb_kpis.upsample(time_column="Date_Time_Fin", every='1h')
            enb_kpis = enb_kpis.join(
                        pl.DataFrame({"Date_Time_Fin": external_dates}),
                        on="Date_Time_Fin",
                        how="outer"
                    )
            out_kpi = enb_kpis.with_columns([
            pl.col("EnodeB_Fin").fill_null(strategy="forward"),
                ])
            out_kpi = enb_kpis.with_columns([
            pl.col("EnodeB_Fin").fill_null(strategy="backward")
                ])
            
        return out_kpi
    @classmethod
    def scaler_variance_adjustment(self,data, scaler = ranged_scaler, cell_level=True , variance_threshold = 0.001):
        out_kpi = pl.DataFrame()
        
        if cell_level:
            cells = data['cell_carrier'].unique().to_list()
            omit_cells = []
            master_col_list = defaultdict(lambda:0)
            # Fill data with 97th percentile and scale afterwards
            for cell in cells:
                temp = data.filter(pl.col('cell_carrier') == cell)
                quants = temp.quantile(0.97, "nearest")
                if scaler is not None:
                    for col in temp.columns:
                        if temp[col].dtype.is_float() | temp[col].dtype.is_integer():
                            quant_val = quants[col][0]
                            temp = temp.with_columns(
                                pl.when(pl.col(col) > quant_val)
                                .then(quant_val)
                                .otherwise(pl.col(col))
                                .alias(col)
                            )
                            temp = temp.with_columns( scaler(temp[col]))
                    out_kpi = out_kpi.vstack(temp)
                else:
                    out_kpi = out_kpi.vstack(temp)
            out_kpi= out_kpi.fill_nan(0) 
            
            # idenfity columns with near zero variance and remove them
            for cell in cells:
                temp = out_kpi.filter(pl.col('cell_carrier') == cell)
                variances = temp.var()
                var_list = []
                zero_count = 0
                for col in variances.columns:
                    if variances[col].dtype.is_float() | variances[col].dtype.is_integer():
                        var_list.append(variances[col][0])
                        if variances[col][0] == 0:
                            master_col_list[col]+=1
                            zero_count += 1
                if np.mean(var_list) >= variance_threshold:
                    omit_cells.append(cell)
            
            drop_columns = []
            for k, v in master_col_list.items():
                if v >= len(omit_cells):
                    drop_columns.append(k)
            out_kpi = out_kpi.drop(drop_columns)
            
            omit_cells = []
            master_col_list = defaultdict(lambda:0)
            # run again after omitted cells are cleaned
            for cell in cells:
                temp = out_kpi.filter(pl.col('cell_carrier') == cell)
                variances = temp.var()
                var_list = []
                zero_count = 0
                for col in variances.columns:
                    if variances[col].dtype.is_float() | variances[col].dtype.is_integer():
                        var_list.append(variances[col][0])
            
                if np.mean(var_list) <= variance_threshold:
                    omit_cells.append(cell)
            #remove inactive cell carriers
            out_kpi = out_kpi.filter(~pl.col("cell_carrier").is_in(omit_cells))             
        else:
            if scaler is not None:
                quants = data.quantile(0.97, "nearest")
                for col in data.columns:
                    if data[col].dtype.is_float() | data[col].dtype.is_integer():
                        quant_val = quants[col][0]
                        data = data.with_columns(
                                pl.when(pl.col(col) > quant_val)
                                .then(quant_val)
                                .otherwise(pl.col(col))
                                .alias(col)
                            )
                        data = data.with_columns( scaler(data[col]))

            out_kpi=data
        
        out_kpi= out_kpi.fill_nan(0) 
        out_kpi = drop_near_zero_variance(out_kpi)
        return out_kpi
    
        
