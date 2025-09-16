# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:31:58 2025

@author: cjymain
"""

import pandas as pd
import numpy as np

from technically.const import ML_FEATURE_ENGINEERING_CONFIG, TREND_PERIODS
import technically.utils.optimizations as utils


class ModelPreparation:
    """
    Prepares individual ticker data for core models.
    """

    def __init__(self, ticker, df: pd.DataFrame, mode: str):
        self.ticker = ticker
        self.df = df
        self.mode = mode

    def get_config(self):
        self.scaling_type = ML_FEATURE_ENGINEERING_CONFIG[self.mode]["cols_to_scale"]["type"]
        self.scaling_cols = ML_FEATURE_ENGINEERING_CONFIG[self.mode]["cols_to_scale"]["features"]

        self.rolling_periods = ML_FEATURE_ENGINEERING_CONFIG[self.mode]["cols_to_roll"]["windows"]
        self.rolling_cols = ML_FEATURE_ENGINEERING_CONFIG[self.mode]["cols_to_roll"]["features"]

        if self.mode == "technicals":  # See 'ml_config.json'
            self.lagging_periods = ML_FEATURE_ENGINEERING_CONFIG[self.mode]["cols_to_lag"]["lag_periods"]
            self.lagging_cols = ML_FEATURE_ENGINEERING_CONFIG[self.mode]["cols_to_lag"]["features"]

    def execute(self):
        self.get_config()

        self.scale(self.scaling_cols, self.scaling_type)
        self.roll(self.rolling_cols, self.rolling_periods)
        if self.mode == "technicals":
            self.lag(self.lagging_cols, self.lagging_periods)

        return self.df

    def scale(self, features_to_scale: list, scaling_type: str):
        """
        Scales features according to scaling_type

        :param features_to_scale: List of features to scale
        :param scaling_type: Can equal Standard, MinMax, MaxAbs, Robust, QuantileTransformer, PowerTransformer
        """
        # Not all tickers will have all columns listed in ml_config.json
        features = [f for f in features_to_scale if f in self.df.columns]

        # Scales
        scaled_df = utils.scaler(
            scaling_type,
            self.df[features].copy(),
            return_as="pandas"
        )

        # Applies changes
        self.df[features] = scaled_df
        return

    def roll(self, features_to_roll, windows):
        """
        Create rolling statistics for specified features.

        :param features_to_roll: List of feature names to create rolling stats for
        :param windows: List of window sizes for rolling calculations
        """
        # Not all tickers will have all columns listed in ml_config.json
        features = [f for f in features_to_roll if f in self.df.columns]

        for feature in features:
            for window_size in windows:
                arr = self.df[feature].copy().values

                for roll_type in ["mean", "std", "min", "max"]:
                    self.df[f"{feature}_{roll_type}_{window_size}"] = utils.np_rolling(
                        arr,
                        window_size,
                        calc_type=roll_type
                    )
        return

    def lag(self, features_to_lag, lag_periods):
        """
        Create lagged versions of specified features.

        df: Spark DataFrame
        features_to_lag: List of feature names to lag
        lag_periods: List of lag periods to create

        :return: Spark DataFrame with lagged features
        """
        # Not all tickers will have all columns listed in ml_config.json
        features = [f for f in features_to_lag if f in self.df.columns]

        for feature in features:
            for lag_period in lag_periods:
                self.df[f"{feature}_lagged_{lag_period}"] = self.df[feature].copy().shift(lag_period)
        return
