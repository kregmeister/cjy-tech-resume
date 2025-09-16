#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:20:24 2025

@author: cjymain
"""

import pandas as pd
from warnings import simplefilter

# Ignores misleading Pandas performance warnings
simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from technically.const import DATE_BUCKETS, MIN_PERIODS
import numpy as np
import re
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer
)

# Ignores sklearn All-NaN RuntimeWarning
simplefilter(action='ignore', category=RuntimeWarning)

def reformat_names(item: str) -> str:
    """
    Re-formats future PostgreSQL column/table names to prevent them from being treated as constants.

    Args:
        item: The string to re-format.

    Examples:
        reformat_names('helloFromBoston') returns 'hello_from_boston'
        reformat_names('6043567') returns '_6043567'

    Returns:
        str: The re-formatted string.
    """
    if isinstance(item, str):
        # Removes all spaces and special characters from item
        item = re.sub(r'[^A-Za-z0-9_]', '', item)

        # Handles known exceptions to default rules
        if item == "permaTicker":
            item = "permaticker"
        elif item.startswith("US"):
            item = "us" + item[2:]
        elif item == "trailingPEG1Y":
            item = "trailing_peg_1y"
        elif item == "prefDVDS":
            item = "pref_dvds"

        # Converts camel case to snake case
        item = re.sub(r'(?<!^)(?=[A-Z])', '_', item)

        # Adds underscore to items beginning with a digit
        item = re.sub(r'^(\d)', r'_\1', item)

        return item.lower()
    else:
        return item

def np_shift(arr: np.ndarray, periods=1):
    """
    Shift numpy array (like pandas.shift).

    Args:
        arr (np.ndarray): The numpy array to be shifted.
        periods (int, optional): How many periods to shift. Default is 1.

    Returns:
        result (np.ndarray): The shifted numpy array.
    """
    result = np.empty_like(arr, dtype=float)
    if periods > 0:
        result[:periods] = np.nan
        result[periods:] = arr[:-periods]
    elif periods < 0:
        result[periods:] = np.nan
        result[:periods] = arr[-periods:]
    else:
        return arr.copy()
    return result

def np_rolling(arr: np.ndarray, period: int, calc_type: str, func_dict=None):
    """
    Conducts rolling window operations on numpy array.

    Args:
        arr (np.ndarray): The numpy array to roll.
        period (int): Size of rolling window.
        calc_type (str): Type of rolling window operation.
        func_dict (dict, optional): Function dict for custom rolling window operations.
            Keys are string names of functions to execute.
            Values are a pointer to the function. Default is None.

    Returns:
        result (np.ndarray): The rolled numpy array.
    """
    # Keep track of NaN positions
    null_mask = np.isnan(arr)

    # Get indices of non-NaN values
    valid_indices = np.where(~null_mask)[0]

    # Extract valid values
    valid_values = arr[valid_indices]

    # If we don't have enough non-NaN values to form even one window, return array of NaNs
    if len(valid_values) < period:
        return np.full_like(arr, np.nan)

    # Perform calculation on valid values
    if calc_type != "func":
        windows = np.lib.stride_tricks.sliding_window_view(valid_values, period)
        if calc_type == "mean":
            result_values = np.mean(windows, axis=1)
        elif calc_type == "sum":
            result_values = np.sum(windows, axis=1)
        elif calc_type == "std":
            result_values = np.std(windows, axis=1)
        elif calc_type == "min":
            result_values = np.min(windows, axis=1)
        elif calc_type == "max":
            result_values = np.max(windows, axis=1)
        else:
            raise ValueError("Invalid type. Must be 'mean', 'std', 'min', 'max', or 'func'.")
        # Create result array with same shape as input, filled with NaNs
        result = np.full_like(arr, np.nan)

        # Account for window size
        result_positions = valid_indices[period - 1:]

        # Place results
        result[result_positions] = result_values
        return result
    # Run a custom function on array
    elif calc_type == "func":
        func = func_dict.get("func")
        func_params = {k: v for k, v in func_dict.items() if k != "func"}

        output_len = arr.shape[0] - period + 1
        result = np.full(arr.shape[0], False, dtype=bool)
        for i in range(output_len):
            window = arr[i:i + period]
            result[i + period - 1] = func(window, **func_params)
        return result

def weighted_mean(subdf: pd.DataFrame, weight_col_name: str, group_by: str):
    """

    Args:
        subdf (pd.Series or np.ndarray): The pandas dataframe to get weighted mean for.
        weight_col_name (str): The column to weigh the mean by.
        group_by (str): The column to group the data by.

    Returns:

    """
    excluded_cols = [weight_col_name, group_by, "asset_type"]
    for col in subdf.columns:
        if col in excluded_cols:
            continue
        subdf[subdf[group_by] == "unknown"][col] = np.average(subdf[col], weights=subdf[weight_col_name])
    return subdf

def bucketizer(subdf: pd.DataFrame):
    """
    Separates one dataframe into several dataframes divided by date thresholds.

    Args:
        subdf (pd.DataFrame):
        buckets (list, optional): The date thresholds to divide each dataframe by.

    Returns:
        bucketed_dfs (list): List of bucketed dataframes.
    """
    bucketer = subdf["date"]
    bucketed_dfs = []
    for i in range(len(DATE_BUCKETS) + 1):
        if i == 0:
            condition = bucketer < DATE_BUCKETS[i]
        elif i != 0 and i != len(DATE_BUCKETS):
            condition = (bucketer < DATE_BUCKETS[i]) & (bucketer >= DATE_BUCKETS[i - 1])
        else:
            condition = bucketer >= DATE_BUCKETS[-1]
        bucket = subdf[condition]
        # For newer tickers to be properly bucketed
        if bucket.empty:
            continue
        bucketed_dfs.append(bucket)
    return bucketed_dfs

def scaler(scaler_type: str, subdf: pd.DataFrame | np.ndarray, bucketed=False, return_as="numpy"):
    """
    Scales numpy array or pandas dataframe.

    Args:
        scaler_type (str): Type of scaler to use. Options are Standard, MinMax, MaxAbs, Robust, QuantileTransformer, PowerTransformer.
        subdf: Dataframe containing columns to scale.
        bucketed (bool, optional): True scales each bucketed dataframe independently, False scales full dataframe. Defaults to False.
        return_as (str, optional): Datatype to return. Defaults to "numpy".

    Returns:
        np.ndarray or pd.DataFrame: Scaled numpy array or pandas dataframe.
    """
    if type(subdf) == pd.Series:
        subdf = pd.DataFrame(subdf)

    if scaler_type == "Standard":
        scaler_obj = StandardScaler()
    elif scaler_type == "MinMax":
        scaler_obj = MinMaxScaler()
    elif scaler_type == "MaxAbs":
        scaler_obj = MaxAbsScaler()
    elif scaler_type == "Robust":
        scaler_obj = RobustScaler()
    elif scaler_type == "QuantileTransformer":
        scaler_obj = QuantileTransformer()
    else:
        scaler_obj = PowerTransformer()

    if bucketed:
        buckets = bucketizer(subdf)
        scaled_buckets = []
        for bucket in buckets:
            scaler_obj.fit(bucket)
            scaled_buckets.append(scaler_obj.transform(bucket))
        scaled_df = np.concatenate(scaled_buckets)
    else:
        scaler_obj.fit(subdf)
        scaled_df = scaler_obj.transform(subdf)
    if return_as == "numpy":
        return scaled_df
    else:
        return pd.DataFrame(scaled_df, columns=subdf.columns)

def rolling_zscore(data: pd.Series | np.ndarray, window=250):
    """
    Compute rolling zscore using rolling window operations.

    Args:
        data (pd.Series or np.ndarray): The pandas series or numpy array to operate on.
        window: Size of rolling window.

    Returns:
        np.ndarray: The rolling zscore result as numpy array.
    """
    if len(data) <= window:
        window = MIN_PERIODS

    if isinstance(data, pd.Series):
        data = data.values

    mean = np.full_like(data, np.nan)
    mean[window - 1:] = np.convolve(data, np.ones(window) / window, mode='valid')
    sq = data ** 2
    mean_sq = np.full_like(data, np.nan)
    mean_sq[window - 1:] = np.convolve(sq, np.ones(window) / window, mode='valid')
    var = (mean_sq - mean ** 2) * (window / (window - 1))
    std = np.sqrt(var)
    return (data - mean) / std
    #if len(data) <= window:
    #    window = MIN_PERIODS
    #
    #if type(data) == np.ndarray:
    #    data = pd.Series(data)
    #return (data - data.rolling(window).mean()) / data.rolling(window).std()

def percentile(data: pd.Series | np.ndarray, percent: int | str):
    """
    Compute percentile.

    Args:
        data (pd.Series or np.ndarray): The pandas series or numpy array to operate on.
        percent (int or str): Percentile to compute.

    Examples:
        percentile(arr, 10)- 10th percentile
        percentile(ser, "25th")- 25th percentile

    Returns:
       np.ndarray: The percentile result as numpy array.
    """
    # If percent is formatted as "1st" or "50th"
    if isinstance(percent, str):
        percent = int(percent[:-2])

    if type(data) == pd.Series:
        data = data.dropna()
        A = data.values
    elif type(data) == np.ndarray:
        A = data[~np.isnan(data)]
    else:
        raise TypeError(
            "Input must be a NumPy array or Pandas Series."
        )

    return np.percentile(A, percent)

def line_best_fit(data: pd.Series | np.ndarray, period: int, return_as="list"):
    """
    Finds the line of best fit for rolling windows of data.

    Args:
        data (pd.Series or np.ndarray): The pandas series or numpy array to operate on.
        period: Length of rolling window.
        return_as (str, optional): Datatype to return. Options are "numpy" or "list". Defaults to "list".

    Returns:
        covariance_result: List or np.ndarray of results.
    """
    if isinstance(data, pd.Series):
        data = data.values
    X = np.asarray(range(len(data)))

    # Precompute sliding window sums
    X_slices = np.lib.stride_tricks.sliding_window_view(X, period)
    Y_slices = np.lib.stride_tricks.sliding_window_view(data, period)

    n_windows = X_slices.shape[0]
    sum_x = np.sum(X_slices, axis=1)
    sum_y = np.sum(Y_slices, axis=1)
    sum_xy = np.sum(X_slices * Y_slices, axis=1)
    sum_xx = np.sum(X_slices ** 2, axis=1)

    # Compute covariance for each window
    denominator = (period * sum_xx) - (sum_x ** 2)
    numerator = (period * sum_xy) - (sum_x * sum_y)

    covariances = np.full(n_windows, np.nan)

    mask = denominator != 0
    covariances[mask] = numerator[mask] / denominator[mask]

    # Fill in None for oldest values
    covariance_result = [np.nan] * (period - 1) + covariances.tolist()

    if return_as == "numpy":
        return np.array(covariance_result)
    else:
        return covariance_result

def failure_swings(y_window: pd.Series, swing_type: str, threshold: int | float):
    """
    Detects the occurrence of a failure swings using a threshold-based technical indicator signal.

    Args:
        y_window (pd.Series): A single window of values derived from a larger dataframe.
        swing_type (str): The direction to search for swings. Options are "bottom" or "top".
        threshold (int or float): indicator threshold (I.e. -100 for bullish CCI)

    Returns:
        bullish_swing or bearish_swing (bool): Whether a failure swing occurs or not for the window.
    """
    y = y_window.tolist()
    if swing_type == "bottom":
        low = min(y)
        if low > threshold:
            return False
        min_thresh = y.index(low) + 3
        if min_thresh > len(y) + 1:
            return False
        y_eval = y[min_thresh:]
        try:
            lowest_valley = min(
                [v for i, v in zip(range(len(y_eval) - 1), y_eval[:-1])
                 if y_eval[i] < y_eval[i + 1] and y_eval[i] < y_eval[i - 1]]
            )
        except ValueError:
            return False
        bullish_swing = (y[-2] == lowest_valley)
        return bullish_swing

    if swing_type == "top":
        high = max(y)
        if high < threshold:
            return False
        max_thresh = y.index(high) + 3
        if max_thresh > len(y)+1:
            return False
        y_eval = y[max_thresh:]
        try:
            highest_peak = max(
                [v for i, v in zip(range(len(y_eval) - 1), y_eval[:-1])
                 if y_eval[i] > y_eval[i + 1] and y_eval[i] > y_eval[i - 1]]
            )
        except ValueError:
            return False
        bearish_swing = (y[-2] == highest_peak)
        return bearish_swing