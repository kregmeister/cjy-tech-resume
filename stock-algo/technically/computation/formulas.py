#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:58:21 2023

@author: cjymain
"""

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
import pywt
from technically.const import TREND_PERIODS, MIN_PERIODS, TREND_STRENGTH_SCORING

from technically.utils import optimizations as utils


class TechnicalFormulas:
    """
    Calculates technical indicators and other features.
    """

    def __init__(self, df: pd.DataFrame, calc_num: int):
        self.df = df.copy()  # Columns added throughout
        self.og_df = df.copy()  # Includes only price data
        if calc_num > len(df):  # Calculates all periods
            self.calc_num = len(df)
            self.full = True
        else:  # Calculates increment of periods
            self.calc_num = calc_num + MIN_PERIODS
            self.full = False

    def persist(self):
        """
        Updates the current state of self.df to the space that instantiated the class.

        Returns:
            pd.DataFrame: updated self.df.
        """
        return self.df

    def apply(self, column: str, ser: pd.Series | np.ndarray, calc_num: int = None):
        """
        Applies the technical formula results to self.df.

        Args:
            column (str): new df column name.
            ser (pd.Series | np.ndarray): series to apply the technical formula to.
            calc_num (int): how many rows to be applied to self.df.
            Defaults to self.calc_num, but may need to be passed manually.

        Returns:
            None
        """
        if self.full:
            self.df[column] = ser
        else:
            if calc_num is None:
                calc_num = self.calc_num
            if isinstance(ser, np.ndarray | list):
                ser = pd.Series(ser, index=self.df.index[-calc_num:])
            else:
                ser.index = self.df.index[-calc_num:]

            self.df[column] = pd.Series([np.nan] * len(self.df), index=self.df.index)
            self.df.loc[calc_num:, column] = ser
        return

    # Incremental calculations (requires partial data)
    def demand_index(self, period=10, price_range=2, smoothing=10):
        """
        Recursively calculates demand index for each period.
        Generates self.df columns: demandIndex

        Args:
            period (int): Number of days for rolling mean of price range. Default is 10.
            price_range (int): Number of days for price range calculation. Default is 2.
            smoothing (int): Number of days for smoothing. Default is 10.

        Returns:
            None
        """
        df = self.og_df[["open", "high", "low", "close", "volume"]].iloc[-self.calc_num:].copy()
        P = (df["close"] - df["open"]) / df["open"]

        two_day_price_range = df["high"].rolling(window=price_range).max() - df["low"].rolling(window=price_range).min()
        VA = two_day_price_range.rolling(window=period).mean()
        K = (3 * df["close"]) / VA

        P = P * K
        V = df["volume"]

        change_mask = df["close"] > df["open"]
        BP = np.where(change_mask, V, V / P)

        SP = np.where(change_mask, V / P, V)

        pressure_mask = abs(BP) > abs(SP)
        DI = np.where(pressure_mask, SP / BP, BP / SP)
        DI_smoothed = pd.Series(DI).ewm(span=smoothing).mean()

        self.apply("demand_idx", DI_smoothed)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.demand_idx_period = period
        #self.df.demand_idx_window = price_range
        #self.df.demand_idx_smoothing = smoothing
        return

    def kalman_filter_multi(self, n_states=1, process_noise=0.01, measurement_noise=15.0, initial_error_covariance=100.0):
        """
        Kalman filter with multiple state vectors.
        Generates self.df columns: kalman_close, kalman_trend

        Args:
        N (int): Number of states. Default is 1.
        process_noise (float): Noise to add to each state vector. Default is 0.01.
        measurement_noise (float): Noise to add to each resulting value. Default is 15.0.
        initial_error_covariance (float): Initial error covariance. Default is 100.0.

        Returns:
            None
        """
        values = self.og_df["close"].iloc[-self.calc_num:].copy().values

        kf = KalmanFilter(dim_x=n_states, dim_z=1)

        kf.F = np.eye(n_states)  # State transition matrix
        kf.H = np.ones((1, n_states))  # Measurement matrix
        kf.P *= initial_error_covariance
        kf.R = measurement_noise
        kf.Q = process_noise

        # Starting with the first price and zero velocity
        kf.x = np.array([[values[0]]])

        filtered_values = []
        trends = []

        # Previous filtered price for trend calculation
        prev_filtered_value = None

        trend = 0
        for z in values:
            # Predict step
            kf.predict()

            # Update step with the new measurement
            kf.update(np.array([[z]]))

            # Extract the filtered price estimate
            filtered_value = kf.x[0, 0]
            filtered_values.append(filtered_value)

            # Tracks direction and length of trend
            if prev_filtered_value is not None:
                if filtered_value > prev_filtered_value:
                    if trend < 0:
                        trend = 0
                    trend += 1  # Upward trend
                elif filtered_value < prev_filtered_value:
                    if trend > 0:
                        trend = 0
                    trend += -1  # Downward trend
                else:
                    if trend > 0:
                        trend += 1  # No change
                    else:
                        trend += -1
            trends.append(trend)
            # Update previous filtered price
            prev_filtered_value = filtered_value

        self.apply("kalman_close", filtered_values)
        self.apply("kalman_trend", trends)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.kalman_close_states = n_states
        #self.df.kalman_close_process_noise = process_noise
        #self.df.kalman_close_measurement_noise = measurement_noise
        #self.df.kalman_close_initial_error_covariance = initial_error_covariance
        return

    def kalman_filter_single(self, process_noise=0.01, measurement_noise=15.0, initial_error_covariance=100.0, trending=True):
        """
        Kalman filter with a single state vector.
        Generates self.df columns: kalman_close, kalman_trend.

        Args:
            process_noise (float): Covariance of process noise. Default is 0.01.
            measurement_noise (float): Covariance of measurement noise. Default is 15.0.
            initial_error_covariance (float): Initial error covariance. Default is 100.0.
            trending (bool): Whether to track trend direction and length based on kalman filter outputs. Default is True.

        Returns:
            None
        """
        calc_num = self.calc_num + 200
        values = self.og_df["close"].iloc[-calc_num:].copy().values

        filtered_values = []
        trends = []

        prev_x = None
        x = values[0]
        P = initial_error_covariance
        trend = 0
        for z in values:
            # Predict step
            x_pred = x
            P_pred = P + process_noise
            # Kalman Gain
            K = P_pred / (P_pred + measurement_noise)
            # Update step
            x = x_pred + K * (z - x_pred)
            P = (1 - K) * P_pred

            filtered_values.append(x)

            # Tracks direction and length of trend
            if trending:
                if prev_x is not None:
                    if x > prev_x:
                        if trend < 0:
                            trend = 0
                        trend += 1  # Reverse up
                    elif x < prev_x:
                        if trend > 0:
                            trend = 0
                        trend += -1  # Reverse down
                    else:
                        if trend > 0:
                            trend += 1  # No reversal
                        else:
                            trend += -1
                trends.append(trend)
            # Update previous filtered price
            prev_x = x

        self.apply("kalman_close", filtered_values, calc_num=calc_num)
        self.apply("kalman_trend", trends, calc_num=calc_num)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.kalman_close_process_noise = process_noise
        #self.df.kalman_close_measurement_noise = measurement_noise
        #self.df.kalman_close_initial_error_covariance = initial_error_covariance
        return

    def wavelets(self, wavelet="db6", scale=0.3):
        """
        Calculates wavelet coefficients.
        Generates self.df columns: wavelet_db6

        Args:
            wavelet (str): Wavelet to calculate coefficients. Default is "db6".
            scale (float): Scaling factor. Default is 0.3.

        Returns:
            None
        """
        values = self.og_df["close"].iloc[-self.calc_num:].copy().values

        # Deconstruct price coefficients
        coefficients = pywt.wavedec(values, wavelet, mode='per')
        # Filter out coefficients based on scale
        coefficients[1:] = [
            pywt.threshold(i, value=scale * values.max())
            for i in coefficients[1:]
        ]
        # Reconstruct de-noised values
        reconstructed_signal = pywt.waverec(coefficients, wavelet, mode='per')

        self.apply(f"wavelet_{wavelet}", reconstructed_signal)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.wavelet_scale = scale
        return

    def percent_from_extreme(self, period=250, direction="max"):
        """
        Measures the percent difference between a given rolling window's max/min and closing price.
        Generates self.df columns: percentDiffMax or percentDiffMin

        Args:
            period (int): Size of the rolling window. Default is 250.
            direction (str): Options are "max" or "min". Default is "max".

        Returns:
            None
        """
        df = self.og_df[["high", "low", "close"]].iloc[-self.calc_num:].copy()

        # Sets period to min_len if DataFrame length falls between them
        if MIN_PERIODS <= len(df) <= period:
            period = MIN_PERIODS

        # Percent difference formula
        def _pc_diff(close, extreme):
            pc = (close - extreme).abs() / ((close + extreme) / 2) * 100
            return pc

        # Calculates
        if direction == "max":
            year_high = df["high"].rolling(window=period).max()
            pc_diff = _pc_diff(df["close"], year_high)
            self.apply("close_percent_from_max", pc_diff)

            # Will apply formula variables as attributes via df.attrs or dict and saved to models db
            #self.df.close_percent_from_max_period = period
            #self.df.close_percent_from_max_min_len = min_len
        else:
            year_low = df["low"].rolling(window=period).min()
            pc_diff = _pc_diff(df["close"], year_low)
            self.apply("close_percent_from_min", pc_diff)

            # Will apply formula variables as attributes via df.attrs or dict and saved to models db
            #self.df.close_percent_from_min_period = period
            #self.df.close_percent_from_min_min_len = min_len
        return

    def average_true_range(self, period=20):
        """
        Calculates the Average True Range (ATR) of a given time period.
        Generates self.df columns: atr

        Args:
            period (int): EMA length of True Range (TR). Default is 20.

        Returns:
            None
        """
        calc_num = self.calc_num + 100
        df = self.og_df[["high", "low", "close"]].iloc[-calc_num:].copy()

        alpha = 1/period

        # TR
        h_l = df['high'] - df['low']
        h_c = np.abs(df['high'] - df['close'].shift(1))
        l_c = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)

        # ATR
        atr = tr.ewm(alpha=alpha, adjust=False, min_periods=MIN_PERIODS).mean()

        self.apply("atr", atr, calc_num=calc_num)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.atr_period = period
        #self.df.atr_period_min_len = min_periods
        return

    def aroon_oscillator(self, period=20):
        """
        Calculates the Aroon Oscillator of a given time period.
        Generates self.df columns: aroon20

        Args:
            period (int): Window length of checking for new highs/lows. Default is 20.

        Returns:
            None
        """
        df = self.og_df[["high", "low"]].iloc[-self.calc_num:].copy()

        # Counts how many periods since the window's max/min
        periods_since_high = df['high'].rolling(window=period).apply(
            lambda x: period - 1 - x.argmax(), raw=True
        )
        periods_since_low = df['low'].rolling(window=period).apply(
            lambda x: period - 1 - x.argmin(), raw=True
        )

        # Calculates aroon directions
        aroon_up = ((period - periods_since_high) / period) * 100
        aroon_down = ((period - periods_since_low) / period) * 100

        aroon = aroon_up - aroon_down

        self.apply("aroon", aroon)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.aroon_period = period
        return

    def commodity_channel_index(self, period=20, smoothing=14):
        """
        Calculates the Commodity Channel Index (CCI) of a given time period and its moving average.
        Generates self.df columns: cci20, cci20_ma14

        Args:
            period (int): Length of the moving average window. Default is 20.
            smoothing (int): Length of EMA smoothing window. Default is 14.

        Returns:
            None
        """
        df = self.og_df[["high", "low", "close"]].iloc[-self.calc_num:].copy()

        # Typical price
        tp = (df['high'] + df['low'] + df['close']) / 3

        # Calculates moving average and mean deviation
        ma = tp.rolling(window=period).mean()
        md = tp.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )

        # Calculates CCI and smooths
        cci = (tp - ma) / (.015 * md)
        cci_ma = (cci.rolling(window=smoothing).sum()) / smoothing

        self.apply("cci", cci)
        self.apply("cci_ma", cci_ma)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.cci_period = period
        #self.df.cci_smoothing = smoothing
        return

    def relative_strength_index(self, period=14):
        """
        Calculates the Relative Strength Index (RSI) of a given time period.
        Generates self.df columns: rsi20

        Args:
            period (int): Length of each RS window. Default is 14.

        Returns:
            None
        """
        close = self.og_df['close'].iloc[-self.calc_num:].copy().values

        diffs = np.diff(close, prepend=np.nan)

        gains = np.where(diffs > 0, diffs, 0)
        losses = np.where(diffs < 0, -diffs, 0)

        avg_gain = np.zeros_like(close)  # Length of DataFrame
        avg_loss = np.zeros_like(close)

        # Initializes first RSI value
        avg_gain[period] = np.mean(gains[1:period+1])  # Exclude the first NaN
        avg_loss[period] = np.mean(losses[1:period+1])

        rsi = np.full_like(close, np.nan)

        for i in range(period+1, len(close)):  # Skips first window
            # Calculates average gain/loss for each window
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period

            if avg_loss[i] == 0:
                rs = np.inf  # Avoid division by zero
            else:
                # Calculate Relative Strength
                rs = avg_gain[i] / avg_loss[i]

            # Calculate RSI for period
            rsi[i] = 100 - (100 / (1 + rs))

        self.apply("rsi", rsi)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.rsi_period = period
        return

    def stdev_percent_of_sma(self, period=20):
        """
        Calculates the ratio of standard deviation to moving average for a given time period.
        Generates self.df columns: STDevPercent20

        Args:
            period (int): Length of the moving average and standard deviation window. Default is 20.

        Returns:
            None
        """
        series = self.og_df["close"].iloc[-self.calc_num:].copy()

        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        pc_std = (std / sma) * 100

        self.apply("stdev_percent_of_sma", pc_std)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.stdev_percent_of_sma_period = period
        return

    def bollinger_bands(self, period=20):
        """
        Calculates the Bollinger Bands of a given time period.
        Generates self.df columns: BollingerUpper20, BollingerSMA20, BollingerLower20, bollinger_band_range

        Args:
            period (int): Length of the moving average and standard deviation window. Default is 20.

        Returns:
            None
        """
        series = self.og_df["close"].iloc[-self.calc_num:].copy()

        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        upper_band = (sma + (std * 2))
        lower_band = (sma - (std * 2))

        self.apply("bollinger_upper_band", upper_band)
        self.apply("bollinger_sma", sma)
        self.apply("bollinger_lower_band", lower_band)
        self.apply("bollinger_band_range", upper_band - lower_band)


        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.bollinger_sma_period = period
        return

    def short_know_sure_thing(self, roc1=10, roc2=15, roc3=20, roc4=30, sma1=10, sma2=10, sma3=10, sma4=15, sig=9):
        """
        Calculates the short-term Know-sure-thing (KST) of a given time period.
        Generates self.df columns: kst, kstSig

        Args:
            roc1 (int): Rate of change period 1. Default is 10.
            roc2 (int): Rate of change period 2. Default is 15.
            roc3 (int): Rate of change period 3. Default is 20.
            roc4 (int): Rate of change period 4. Default is 30.
            sma1 (int): Moving average period 1. Default is 10.
            sma2 (int): Moving average period 2. Default is 10.
            sma3 (int): Moving average period 3. Default is 10.
            sma4 (int): Moving average period 4. Default is 15.
            sig (int): Length of the signal line moving average window. Default is 9.

        Returns:
            None
        """
        series = self.og_df["close"].iloc[-self.calc_num:].copy()

        # Calculates rate of change for each period
        shift_1 = series.shift(roc1)
        roc_10 = ((series / shift_1) - 1) * 100

        shift_2 = series.shift(roc2)
        roc_15 = ((series / shift_2) - 1) * 100

        shift_3 = series.shift(roc3)
        roc_20 = ((series / shift_3) - 1) * 100

        shift_4 = series.shift(roc4)
        roc_30 = ((series / shift_4) - 1) * 100

        # Calculates moving averages of the rates of change
        rcma1 = roc_10.rolling(window=sma1).mean()
        rcma2 = roc_15.rolling(window=sma2).mean()
        rcma3 = roc_20.rolling(window=sma3).mean()
        rcma4 = roc_30.rolling(window=sma4).mean()

        # Calculates KST and signal line
        kst = (rcma1 * 1) + (rcma2 * 2) + (rcma3 * 3) + (rcma4 * 4)
        signal = kst.rolling(window=sig).mean()

        self.apply("kst", kst)
        self.apply("kst_sig", signal)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.kst_roc1 = roc1
        #self.df.kst_roc2 = roc2
        #self.df.kst_roc3 = roc3
        #self.df.kst_roc4 = roc4
        #self.df.kst_sma1 = sma1
        #self.df.kst_sma2 = sma2
        #self.df.kst_sma3 = sma3
        #self.df.kst_sma4 = sma4
        #self.df.kst_sig = sig
        return

    def moving_average_convergence_divergence(self, ema_short=12, ema_long=26):
        """
        Calculates the Moving Average Convergence Divergence (MACD).
        Generates self.df columns: macd, macdSig, macd_hist

        Args:
        ema_short (int): Period of EMA short moving average. Default is 12.
        ema_long (int): Period of EMA long moving average. Default is 26.
        min_periods (int): Minimum number of periods required to calculate MACD. Default is 100.

        Returns:
            None
        """
        calc_num = self.calc_num + 50
        series = self.og_df["close"].iloc[-calc_num:].copy()

        twelve = series.ewm(span=ema_short, min_periods=MIN_PERIODS).mean()
        twenty_six = series.ewm(span=ema_long, min_periods=MIN_PERIODS).mean()

        macd = twelve - twenty_six
        signal = macd.ewm(span=9).mean()

        self.apply("macd", macd, calc_num=calc_num)
        self.apply("macd_sig", signal, calc_num=calc_num)
        self.apply("macd_hist", macd - signal, calc_num=calc_num)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.macd_ema_short = ema_short
        #self.df.macd_ema_long = ema_long
        #self.df.macd_min_len = min_periods
        return

    def williams_percent_r(self, period=14):
        """
        Calculates the Williams %R of a given time period.
        Generates self.df columns: williamsR14

        Args:
            period (int): Length of the rolling max/min window. Default is 14.

        Returns:
            None
        """
        df = self.og_df[["high", "low", "close"]].iloc[-self.calc_num:].copy()

        maxes = df['high'].rolling(period).max()
        mins = df['low'].rolling(period).min()

        wpr = ((maxes - df['close']) / (maxes - mins)) * -100

        self.apply("wpr", wpr)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.wpr_period = period
        return

    def stochastic_oscillator(self, period=20):
        """
        Calculates the stochastic oscillator of a given time period.
        Generates self.df columns: stochasticK20, stochasticD20

        Args:
            period (int): Length of the rolling max/min window. Default is 20.

        Returns
            None
        """
        df = self.og_df[["high", "low", "close"]].iloc[-self.calc_num:].copy()

        mins = df['low'].rolling(period).min()
        maxes = df['high'].rolling(period).max()

        # Calculates percentK and percentD
        percent_k = (df['close'] - mins) / (maxes - mins) * 100
        percent_d = percent_k.rolling(3).mean()

        self.apply("stochastic_k", percent_k)
        self.apply("stochastic_d", percent_d)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.stochastic_k_period = period
        return

    def parabolic_sar(self, start=0.02, increment=0.02, maximum=0.2):
        """
        Calculates the Parabolic SAR at standard parameters.
        Generates self.df columns: psar

        Args:
            start (float): Starting value. Default is 0.02.
            increment (float): Increment value. Default is 0.02.
            maximum (float): Maximum value. Default is 0.2.

        Returns:
            None
        """
        df = self.og_df[["high", "low"]].iloc[-self.calc_num:].copy()

        af_init = start
        af_max = maximum

        high = df['high'].values
        low = df['low'].values

        index = [0]
        # Initialize trend direction with first 2 rows
        if high[1] > high[0]:
            sar = [min(low[:2])]
            ep = [max(high[:2])]
            af = [af_init]
            trend = ["up"]
        else:
            sar = [max(high[:2])]
            ep = [min(low[:2])]
            af = [af_init]
            trend = ["down"]

        for i in range(len(df) - 2):
            x = i + 2
            y = i + 1
            if trend[-1] == "up":
                sar_m = sar[-1] + af[-1] * (ep[-1] - sar[-1])
                sar.append(min(sar_m, low[y], low[i]))
                if sar[-1] > low[x]:  # Trend up to down
                    sar[-1] = max(high[index[-1]:x])
                    ep.append(low[x])
                    af.append(af_init)
                    trend.append("down")
                    index.append(x)
                else:  # Trend remains up
                    if high[x] > ep[-1]:
                        ep.append(high[x])
                        af.append(min(af[-1] + increment, af_max))
                    trend.append("up")
            else:
                sar_m = sar[-1] - af[-1] * (sar[-1] - ep[-1])
                sar.append(max(sar_m, high[i], high[y]))
                if sar[-1] < high[x]:  # Trend down to up
                    sar[-1] = min(low[index[-1]:x])
                    ep.append(high[x])
                    af.append(af_init)
                    trend.append("up")
                    index.append(x)
                else:  # Trend remains down
                    if low[x] < ep[-1]:
                        ep.append(low[x])
                        af.append(min(af[-1] + increment, af_max))
                    trend.append("down")
        sar.insert(0, None)
        sar = pd.Series(sar)

        self.apply("psar", sar)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.psar_start = start
        #self.df.psar_increment = increment
        #self.df.psar_maximum = maximum
        return

    def average_directional_index(self, smoothing=14):
        """
        Calculates the Average Directional Index (ADX) of a given time period.
        Generates self.df columns: adx, dmi_plus, dmi_minus

        Args:
            smoothing (int): Denominator of alpha. Default is 14.

        Returns:
            None
        """
        calc_num = self.calc_num + 100
        df = self.og_df[["high", "low", "close"]].iloc[-calc_num:].copy()

        alpha = 1/smoothing

        # TR
        h_l = df['high'] - df['low']
        h_c = np.abs(df['high'] - df['close'].shift(1))
        l_c = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)

        # ATR
        atr = tr.ewm(alpha=alpha, adjust=False, min_periods=MIN_PERIODS).mean()
        atr.reset_index(drop=True, inplace=True)

        # DX+-
        h_ph = df['high'] - df['high'].shift(1)
        pl_l = df['low'].shift(1) - df['low']
        plus_dx = pd.Series(
            np.where(
                (h_ph > pl_l) & (h_ph > 0),
                h_ph,
                0.0
            )
        )
        minus_dx = pd.Series(
            np.where(
                (h_ph < pl_l) & (pl_l > 0),
                pl_l,
                0.0
            )
        )

        # DMI+-
        s_plus_dm = plus_dx.ewm(
            alpha=alpha, adjust=False, min_periods=MIN_PERIODS
        ).mean()
        s_minus_dm = minus_dx.ewm(
            alpha=alpha, adjust=False, min_periods=MIN_PERIODS
        ).mean()

        dmi_plus = (s_plus_dm / atr) * 100
        dmi_minus = (s_minus_dm / atr) * 100

        # DX & ADX
        dx = (np.abs(dmi_plus - dmi_minus) / (dmi_plus + dmi_minus)) * 100
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        self.apply("adx", adx, calc_num=calc_num)
        self.apply("dmi_plus", dmi_plus, calc_num=calc_num)
        self.apply("dmi_minus", dmi_minus, calc_num=calc_num)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.adx_smoothing = smoothing
        #self.df.adx_min_len = min_periods
        return

    def exponential_moving_average(self, period: int, column="close"):
        """
        Calculates the Exponential Moving Average (EMA) of a given time period.
        Generates self.df columns: ema_{period}

        Args:
            period (int): Number of periods to calculate EMA for.
            column (str): Column to calculate EMA for. Default is 'close'.

        Returns:
            None
        """
        # Ensures there are enough values to produce a stable EMA
        calc_num = period + self.calc_num
        ema = self.og_df[column].iloc[-calc_num:].copy().ewm(span=period, min_periods=MIN_PERIODS).mean()
        self.apply(f"ema_{period}", ema, calc_num=calc_num)
        return

    # Recursive calculations (requires full data)
    def kaufman_adaptive_moving_average(self, column: str | pd.Series, period: int, apply=True):
        """
        Calculates Kaufman's Adaptive Moving Average (KAMA) of a given time period with Average True Range (ATR) normalization.
        NOTE: When passing 'column' as string, the self.calc_num dataframe slicer is applied.
        When passing 'column' as pd.Series, full dataframe is applied.
        Generates self.df columns: kama20

        Args:
            column (str): Column in self.df OR pd.Series to calculate KAMA for.
            period (int): Number of periods to calculate KAMA for.
            apply: Whether to apply the result to self.df or return it as a stand-alone Pandas Series. Default is True.

        Returns:
            None
        """
        if isinstance(column, str):
            data = self.df[column].copy().values
            atr = self.df["atr"].copy().values
        else:
            data = column.values
            atr = self.df["atr"].copy().values

        # Change calculation
        change = np.pad(np.abs(data[period:] - data[:-period]), (period, 0), 'constant', constant_values=np.nan)

        # Volatility moving average
        vol_diff = np.abs(np.diff(data))
        volatility = np.pad(np.convolve(vol_diff, np.ones(period), 'valid'), (period, 0), 'constant',
                            constant_values=np.nan)

        # Efficiency ratio
        er = np.divide(change, volatility, where=volatility > 0, out=np.zeros_like(data, dtype=float))

        # Normalized Average True Range
        atr_mean = np.pad(np.convolve(atr, np.ones(MIN_PERIODS) / MIN_PERIODS, 'valid'), (MIN_PERIODS - 1, 0),
                          'constant', constant_values=np.nan)
        atr_normalized = atr / atr_mean

        # Ignores Numpy runtime warning
        with np.errstate(invalid="ignore"):
            # Dynamically calculate fast and slow period
            fast_period = np.maximum(2, 5 - atr_normalized * 3).astype(int)
            slow_period = np.maximum(20, 30 + atr_normalized * 10).astype(int)

        # Smoothing constants
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        adaptive_ma = np.zeros_like(data, dtype=float)

        start_idx = period
        if start_idx < len(data):
            adaptive_ma[start_idx-1] = np.mean(data[:start_idx])

            # Iterates through data points, calculating KAMA for each
            for i in range(start_idx, len(data)):
                if not np.isnan(data[i]) and not np.isnan(adaptive_ma[i-1]):
                    adaptive_ma[i] = adaptive_ma[i-1] + sc[i] * (data[i] - adaptive_ma[i-1])
                elif not np.isnan(adaptive_ma[i-1]):
                    adaptive_ma[i] = adaptive_ma[i-1]
        adaptive_ma[adaptive_ma == 0.000000] = np.nan

        if apply:
            self.df["kama"] = adaptive_ma

            # Will apply formula variables as attributes via df.attrs or dict and saved to models db
            #self.df.kama_period = period
            #self.df.kama_min_len = min_periods
            return
        else:
            return adaptive_ma

    def on_balance_volume(self, period=10):
        """
        Calculates the On-Balance Volume (OBV) of a given time period.
        Generates self.df columns: OBV

        Args:
            period (int): Number of periods to calculate OBV for. Default is 10.

        Returns:
            None
        """
        close = self.og_df['close'].copy().values
        volume = self.og_df['volume'].copy().values

        price_diff = np.diff(close)
        direction = np.sign(price_diff)
        direction = np.insert(direction, 0, 0)
        obv_changes = direction * volume
        obv = np.sum(np.lib.stride_tricks.sliding_window_view(obv_changes, period), axis=1)
        obv = np.insert(obv, range(period-1), np.nan)

        self.df["obv"] = obv

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.obv_period = period
        return

    def current_trend(self, y_col: str = "kama"):
        """
        Determines the current trend by calculating line best fit at different periods, averaging them, and taking percentiles.
        Generates self.df columns: {key}_trend_line for key in periods dict, prevailing_trend_line.

        Args:
            y_col: Column used to create trend lines. Default is "KAMA20".

        Returns:
            None
        """
        Y = self.df[y_col].copy().values
        Y = utils.rolling_zscore(Y)

        X_dict = {}
        for name, period in TREND_PERIODS.items():
            if len(Y) < period + 20:
                continue
            slopes = pd.Series(utils.line_best_fit(Y, period))
            X_dict[f"{name}_trend_slope"] = slopes

        trends_df = pd.DataFrame(X_dict)

        for name, period in TREND_PERIODS.items():
            column = f"{name}_trend_slope"

            try:
                X = trends_df[column]
            except KeyError:
                continue

            if len(X) < 20:
                continue
            try:
                percentile_up = utils.percentile(X, 75)
            except IndexError:
                percentile_up = np.inf

            try:
                percentile_down = utils.percentile(X, 25)
            except IndexError:
                percentile_down = -np.inf

            conditions = [
                (X >= percentile_up),
                (X <= percentile_down)
            ]
            res = np.select(conditions, [1, -1])

            new_column = "_".join(column.split("_")[:-1])
            trends_df[new_column] = res

        self.df = pd.concat([self.df, trends_df], axis=1)

    def trend_strength_score(self):
        """
        Creates a trend strength score based on a variety of factors.
        Weights provided from const.py.
        Generates self.df columns: trend_score

        Returns:
            None
        """

        subdf = self.df[[
            "close",
            "macd",
            "macd_hist",
            "dmi_plus",
            "dmi_minus",
            "obv",
            "adx",
            "atr",
            "bollinger_band_range"
        ]].copy()

        subdf["close_diff"] = utils.rolling_zscore(subdf["close"].diff())
        subdf["dmi_diff"] = subdf["dmi_plus"] - subdf["dmi_minus"]
        subdf["obv"] = utils.rolling_zscore(subdf["obv"])

        # Scales some subdf values from -1 to 1
        subdf[[
            "close_diff",
            "macd",
            "dmi_diff",
            "obv"
        ]] = 2 * utils.scaler(
            "MinMax",
            subdf[["close_diff", "macd", "dmi_diff", "obv"]],
            return_as="pandas"
        ) - 1

        # Scales other subdf values from 0 to 1
        subdf[[
            "adx",
            "atr",
            "bollinger_band_range"
        ]] = utils.scaler(
            "MinMax",
            subdf[["adx", "atr", "bollinger_band_range"]],
            return_as="pandas"
        )

        # Calculates initial trend scores
        weight_sum = sum(TREND_STRENGTH_SCORING.values())
        trend_score = sum(
            subdf[col] * weight
            for col, weight in TREND_STRENGTH_SCORING.items()
        ) / weight_sum

        # ADX multiplier (0-2)
        trend_score *= (subdf["adx"] * 2)

        # Takes adaptive MA
        subdf["trend_score"] = self.kaufman_adaptive_moving_average(trend_score, 10, apply=False)
        trend_score = subdf["trend_score"]

        # Separates positive trend scores and negative and scales them individually (-1 to 1)
        pos_scores = pd.Series(
            np.where(trend_score > 0, trend_score, np.nan)
        )
        neg_scores = pd.Series(
            np.where(trend_score < 0, abs(trend_score), np.nan)
        )

        scores = []
        for i, subset in enumerate([pos_scores, neg_scores]):
            subset_scaled = utils.scaler(
                "MinMax",
                subset,
                return_as="pandas"
            )
            if i == 1:
                subset_scaled = subset_scaled * -1

            scores.append(subset_scaled)
        self.df["trend_score"] = scores[0].fillna(scores[1])
        return


    def price_ceilings_floors(self, percentiles: tuple = (99, 1)):
        """
        Creates price ceilings and floors based on percentiles of daily price changes for each period.
        Generates self.df columns: priceCeiling, priceFloor, priceChange.

        Args:
            percentiles (tuple): Top and bottom percentiles for price ceiling and floor.
                Default is (99, 1).

        Returns:
            None
        """
        # Separates self.df into multiple DataFrames by date
        buckets = utils.bucketizer(self.df[["date", "close"]].copy())

        ceilings = []
        floors = []
        diffs = []
        for df in buckets:
            # Calculates daily change in price and percentiles of those changes
            price_changes = df["close"].diff()
            ceiling = utils.percentile(price_changes, percentiles[0])
            floor = utils.percentile(price_changes, percentiles[1])

            # Applies current ceilings and floors to detect extreme price changes
            diffs.append(price_changes)
            ceilings.append(df["close"].shift(1) + ceiling)
            floors.append(df["close"].shift(1) + floor)

        self.df["daily_price_change"] = np.concatenate(diffs)
        self.df["daily_price_ceiling"] = np.concatenate(ceilings)
        self.df["daily_price_floor"] = np.concatenate(floors)

        # Will apply formula variables as attributes via df.attrs or dict and saved to models db
        #self.df.price_ceilings_floors_percentiles = percentiles
        return

    def trend_swing_confirmation(self, window=5):
        """
        Calculates dynamic trend retracement targets based on trend swings.
        Generates self.df columns: trend_reversal_type, trend_reversal_threshold, current_trend, trend_confidence_factor

        Args:
            window (int, optional): Window size for swing point detection. Defaults to 5.

        Returns:
            None
        """
        # Select required columns once to avoid repeated copying
        subdf = self.df[[
            "trend_score",
            "high",
            "low",
            "close",
            "adx",
            "macd_hist",
            "bollinger_sma",
            "atr"
        ]].copy()

        # Vectorized swing point detection using rolling max/min
        high_mask = (subdf['high'].rolling(
            window=2 * window + 1,
            center=True,
            min_periods=window
        ).max() == subdf['high'])

        low_mask = subdf['low'].rolling(
            window=2 * window + 1,
            center=True,
            min_periods=window
        ).min() == subdf['low']

        # Initialize swing point columns
        subdf["swing_high"] = np.where(high_mask, subdf["high"], np.nan)
        subdf["swing_low"] = np.where(low_mask, subdf["low"], np.nan)

        # Ensure minimum distance between swing points using cumulative indexing
        high_indices = subdf.index[high_mask].to_numpy()
        low_indices = subdf.index[low_mask].to_numpy()

        # Filter indices to enforce minimum window distance
        valid_highs = [high_indices[0]] if len(high_indices) > 0 else []
        for idx in high_indices[1:]:
            if idx - valid_highs[-1] >= window:
                valid_highs.append(idx)

        valid_lows = [low_indices[0]] if len(low_indices) > 0 else []
        for idx in low_indices[1:]:
            if idx - valid_lows[-1] >= window:
                valid_lows.append(idx)

        # Update swing points only at valid indices
        subdf.loc[valid_highs, "swing_high"] = subdf.loc[valid_highs, "high"]
        subdf.loc[valid_lows, "swing_low"] = subdf.loc[valid_lows, "low"]

        # Forward-fill swing points and handle initial nulls
        subdf["last_swing_high"] = subdf["swing_high"].ffill().fillna(subdf["high"].iloc[0])
        subdf["last_swing_low"] = subdf["swing_low"].ffill().fillna(subdf["low"].iloc[0])

        # Calculate swing range
        subdf["swing_range"] = subdf["last_swing_high"] - subdf["last_swing_low"]

        # Base confidence from trend strength score
        confidence_factor = subdf["trend_score"].abs().clip(0, 1)

        # Fibonacci retracement levels
        fib_shallow = 0.382
        fib_deep = 0.618

        # Dynamic retracement level
        subdf["fib_retracement_level"] = fib_shallow + (confidence_factor * (fib_deep - fib_shallow))

        # Vectorized retracement target calculation
        subdf["reversal_target"] = np.where(
            subdf["trend_score"] > 0,
            subdf["last_swing_high"] - (subdf["swing_range"] * subdf["fib_retracement_level"]),
            subdf["last_swing_low"] + (subdf["swing_range"] * subdf["fib_retracement_level"])
        )

        # Detect trend direction changes
        trend_direction_change = (np.sign(subdf["trend_score"]) != np.sign(subdf["trend_score"].shift(1))) & (
                    subdf["trend_score"].shift(1) != 0)
        recent_trend_change = trend_direction_change.rolling(10).sum() > 0

        # Identify weak trends
        weak_trend = (subdf["trend_score"].abs() < subdf["trend_score"].abs().quantile(0.3)) | recent_trend_change

        # Volatility adjustment
        volatility_buffer = 4 - (confidence_factor * 0.5)
        subdf["reversal_threshold"] = np.where(
            subdf["trend_score"] > 0,
            subdf["reversal_target"] - (subdf["atr"] * volatility_buffer),
            subdf["reversal_target"] + (subdf["atr"] * volatility_buffer)
        )

        # Adaptive moving averages
        kama_fast = self.kaufman_adaptive_moving_average(subdf["reversal_threshold"], 10, apply=False)
        kama_slow = self.kaufman_adaptive_moving_average(subdf["reversal_threshold"], 20, apply=False)
        blend_weight = np.where(weak_trend, 0.95, 0.4)
        threshold = pd.Series((kama_slow * blend_weight) + (kama_fast * (1 - blend_weight)))

        # Calculate retracement slope
        retracement_slope = pd.Series(utils.line_best_fit(threshold, 10))
        slope_up_thresh = retracement_slope.quantile(0.8)
        slope_down_thresh = retracement_slope.quantile(0.2)

        # Detect reversals
        close = pd.Series(subdf["close"])
        reversal_conditions = (
            (threshold.shift(1) < close.shift(1)) & (threshold > close),
            (threshold.shift(1) > close.shift(1)) & (threshold < close)
        )
        reversals = pd.Series(np.select(reversal_conditions, ["to_down", "to_up"], "None"))
        reversal_idxs = reversals[reversals != "None"].index

        # Confirm reversals
        self.df["reversal_conf"] = "None"
        for i, idx in enumerate(reversal_idxs):
            try:
                next_idx = reversal_idxs[i + 1]
            except IndexError:
                next_idx = len(reversals)

            slope_segment = retracement_slope.iloc[idx:next_idx]
            if len(slope_segment) < 30:
                continue

            reversal_type = reversals.iloc[idx]
            if reversal_type == "to_up":
                pos_slope = slope_segment[slope_segment > slope_up_thresh].index
                if not pos_slope.empty:
                    self.df.loc[pos_slope[0], "reversal_conf"] = reversal_type
            elif reversal_type == "to_down":
                neg_slope = slope_segment[slope_segment < slope_down_thresh].index
                if not neg_slope.empty:
                    self.df.loc[neg_slope[0], "reversal_conf"] = reversal_type

        # Assign final columns to self.df
        self.df["trend_reversal_type"] = reversals
        self.df["trend_reversal_threshold"] = threshold
        self.df["current_trend"] = np.where(threshold > subdf["close"], -1, 1)
        self.df["trend_confidence_factor"] = confidence_factor
