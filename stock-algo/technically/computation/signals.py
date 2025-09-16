#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  14 10:58:21 2025

@author: cjymain
"""
import numpy as np
import pandas as pd
import traceback

from technically.const import INDICATOR_SIGNALS_CONFIG, MIN_PERIODS
from technically.utils import optimizations as utils
from technically.utils.handlers.db import PostgreSQL
from technically.utils.log import get_logger


class TechnicalIndicatorSignals:
    """
    Identifies technical indicator signals.
    """

    def __init__(self, ticker: str, df: pd.DataFrame, backtest_success_rates, calc_num: int):
        self.ticker = ticker
        self.df = df
        self.backtest_success_rates = backtest_success_rates  # Already grouped
        self.calc_num = calc_num + MIN_PERIODS
        self.full = True if self.calc_num > len(self.df) else False
        if self.check() or self.calc_num > len(self.df):
            self.full = True
        else:
            self.full = False

    def check(self):
        """
        Checks whether most recent backtesting success rates have been applied to scores recursively.

        Returns:
            resp (bool): True for recursive, False for incremental.
        """
        with PostgreSQL() as db:
            query = '''
                SELECT 
                    recent_backtest 
                FROM 
                    prices.metadata
                WHERE 
                    table_alias = :ticker;
            '''
            resp = db.run_query(
                query, params={
                    "ticker": self.ticker
                }
            ).fetchall()[0][0]
        return resp

    def reset_recent_backtest(self):
        """
        Sets recent_backtest to False once new backtest rates have been applied to scores recursively.

        Returns:
            None
        """
        with PostgreSQL() as db:
            query = '''
                UPDATE 
                    prices.metadata 
                SET 
                    recent_backtest = DEFAULT 
                WHERE 
                    table_alias = :ticker;
                '''
            db.run_query(
                query, params={
                    "ticker": self.ticker,
                },
                commit=True
            )
        return

    def calculate(self, indicator: str, identify=True):
        """
        Orchestrates steps to identify, categorize, and score technical indicator signal occurrences.

        Args:
            indicator (str): Name of indicator to find signals for (as listed in indicator_signals.json)
            identify (bool): Whether the signals should be identified or not. Default is True.

        Returns:
            None
        """
        try:
            if identify:  # Func signal_identification identifies signals
                results = self.signal_identification(indicator)
            else:  # Func other than signal_identification identifies signals
                results = self.candlestick_patterns()
            if results is None:
                return
            # Assigns signals as being reversal or confirmation
            categorized_results = self.assign_reversal_continuation(results)
            # Applies historic signal success rates as a percent to be applied to the signal 'score'
            self.apply_success_rates(categorized_results)
        # PARTIAL SIGNAL IDENTIFICATION WILL ONLY REMAIN IF ML MODELS CAN PROPERLY HANDLE/ACCOUNT FOR NULLS
        except Exception:
            get_logger().error(
                f"Error calculating signals for {indicator}.", extra={
                    "item_id": self.ticker,
                    "error": traceback.format_exc()
                }
            )
        return

    def persist(self):
        return self.df

    def signal_identification(self, indicator: str):
        """
        Handles the conditional logic of an indicator signal and finds periods where true.

        Args:
            indicator (str): The indicator to find signals for (as listed in indicator_signals.json)

        Returns:
            all_conditions (dict): Keys are signal name.
                Values are boolean pd.Series that indicate whether a signal occurs during that period.
        """
        # Skips indicators that do not have an entry in 'indicator_signals.json' or that are all null in the df
        if (indicator not in INDICATOR_SIGNALS_CONFIG["indicator"] or
                self.df[indicator].isnull().all()):
            return

        all_conditions = {}
        for signal_direction in ["bullish_signals", "bearish_signals"]:
            signal_prefix = signal_direction[:-7]

            # Isolates collection of bullish/bearish signals for the indicator
            indicator_signals = [
                i for i in INDICATOR_SIGNALS_CONFIG[signal_direction] if i["indicator"] == indicator
            ][0]

            # Iterates through each condition of the signal's set
            for signal_type, signal_conditions in list(indicator_signals.items())[1:]:
                signal_key = signal_prefix + signal_conditions["name"]

                # When a signal consists of multiple condition types
                if signal_type.startswith("multi"):
                    res = []
                    for multi_signal_type, multi_signal_conditions in list(signal_conditions.items())[1:]:
                        res_part = self._signal_construction(indicator, multi_signal_type, multi_signal_conditions)

                        if multi_signal_type[-1].isdigit():
                            res[-1] = pd.concat([res[-1], res_part], axis=1).any(axis=1)
                            continue

                        res.append(res_part)
                    res = pd.concat(res, axis=1).all(axis=1)
                else:
                    res = self._signal_construction(indicator, signal_type, signal_conditions)

                all_conditions[signal_key] = res
        return all_conditions

    def assign_reversal_continuation(self, signals: dict):
        """
        Uses current trend metric to categorize signal occurrences as reversal or continuation.
        The boolean pd.Series dict values are converted to string pd.Series: False will now equal "False".
        True will now equal "rev" or "cont".

        Args:
            signals: Dictionary of signal outcomes (Keys=signal_name, Values=pd.Series).

        Returns:
            signals (dict): Dictionary of signal outcomes. Keys are signal name.
                Values are string pd.Series that indicate whether a signal occurs during that period
                and whether that signal indicates reversal or continuation.
        """
        for sig_name, res in signals.items():
            res = res.astype(str)

            try:
                if sig_name.startswith("bullish"):
                    res[res == "True"] = np.where(self.df["current_trend"] == -1, "rev", "cont")
                elif sig_name.startswith("bearish"):
                    res[res == "True"] = np.where(self.df["current_trend"] == 1, "rev", "cont")
            except ValueError:
                continue

            signals[sig_name] = res
        return signals

    def apply_success_rates(self, signals: dict):
        """
        Applies success rates to indicator signal occurrences to quantify their impact on price activity.
        The string pd.Series dict values are converted to float pd.Series: "False" mapped to 0.0.
        "rev" and "cont" will be assigned floats based on backtested (or default) success rates.
        Generates self.df columns: {signal_name}_sig

        Args:
            signals: Dictionary of categorized signal outcomes.

        Returns:
            None
        """
        signals_not_backtested = []

        for sig_name, res in signals.items():
            sig_direction = sig_name[:7]

            try:
                matching_rates = self.backtest_success_rates[
                    self.backtest_success_rates["signal_name"] == sig_name
                ]

                # Applies backtested success rates
                rev_score = matching_rates["reversal_success_rate"].values[0]
                cont_score = matching_rates["continuation_success_rate"].values[0]
                res[res == "rev"] = rev_score
                res[res == "cont"] = cont_score
            except IndexError:  # Signal not backtested
                signals_not_backtested.append(sig_name)

                # Applies arbitrary default success rates so that models can interpret signal occurrence
                if sig_direction == "bullish":  # Defaults
                    res[res == "rev"] = 0.3
                    res[res == "cont"] = 0.15
                elif sig_direction == "bearish":
                    res[res == "rev"] = -0.3
                    res[res == "cont"] = -0.15

            # No signal always equals 0.0
            res[res == "False"] = 0.0
            res = res.astype(float)

            column = f"{sig_name}_ind"
            if self.full:
                self.df[column] = res
            else:
                self.df[column] = pd.Series([np.nan] * len(self.df), index=self.df.index)
                self.df.loc[self.calc_num:, column] = res

        if signals_not_backtested:  # 1+ signal rates not found
            get_logger().info(
                f"Could not find backtested success rates for {signals_not_backtested}. Defaults were applied.", extra={
                    "item_id": self.ticker
                }
            )
        return

    def _signal_construction(self, indicator: str, signal_type: str, signal_conditions: dict):
        """
        Constructs all signal conditions, checks where all are true, and creates boolean results for each period.

        Args:
            signal_type (str): Can equal 'threshold', 'crossover', 'failure_swing', 'divergence', or 'convergence'
            signal_conditions (dict): Dictionary of conditions required for a signal to occur.
                Keys are condition type, values are condition comparison points.

        Returns:
            pd.Series: Boolean pd.Series indicating whether all signal conditions are met.
        """
        if "feature" in signal_conditions:
            feature = self.df[signal_conditions["feature"]].values.copy()
        else:
            feature = self.df[indicator].values.copy()

        if "trend" in signal_conditions:
            feature = self._trend(feature, signal_conditions["trend"])
        if "zscored" in signal_conditions:
            feature = utils.rolling_zscore(feature, signal_conditions["zscored"])

        conditions = []
        if signal_type.startswith("threshold"):
            for cond_type, cond_value in signal_conditions.items():
                if cond_type == "above":
                    above = self._above_below(feature, cond_value)
                    cond = feature >= above
                elif cond_type == "below":
                    below = self._above_below(feature, cond_value)
                    cond = feature <= below
                elif cond_type == "length":
                    length, rolling_sum = self._length(conditions[-1], cond_value)
                    cond = rolling_sum == length
                elif cond_type == "current_trend":
                    cond = self._current_trend(cond_value)
                else:
                    continue

                if self.full:
                    conditions.append(cond)
                else:
                    conditions.append(cond[-self.calc_num:])

        elif signal_type.startswith("crossover"):
            feature_shifted = utils.np_shift(feature, 1)
            for cond_type, cond_value in signal_conditions.items():
                if cond_type == "above":
                    above, above_shifted = self._above_below(None, cond_value, "crossover")
                    cond = (feature >= above) & (feature_shifted <= above_shifted)
                elif cond_type == "below":
                    below, below_shifted = self._above_below(None, cond_value, "crossover")
                    cond = (feature <= below) & (feature_shifted >= below_shifted)
                elif cond_type == "current_trend":
                    cond = self._current_trend(cond_value)
                else:
                    continue

                if self.full:
                    conditions.append(cond)
                else:
                    conditions.append(cond[-self.calc_num:])

        elif signal_type.startswith("failure_swing"):
            thresh = signal_conditions["thresh"]
            for cond_type, cond_value in signal_conditions.items():
                if cond_type == "bottom":
                    period = cond_value
                    cond = utils.np_rolling(
                        feature,
                        period,
                        "func",
                        {"func": utils.failure_swings,
                         "swing_type": "bottom",
                         "threshold": thresh}
                    )
                elif cond_type == "top":
                    period = cond_value
                    cond = utils.np_rolling(
                        feature,
                        period,
                        "func",
                        {"func": utils.failure_swings,
                         "swing_type": "top",
                         "threshold": thresh}
                    )
                elif cond_type == "current_trend":
                    cond = self._current_trend(cond_value)
                else:
                    continue

                if self.full:
                    conditions.append(cond)
                else:
                    conditions.append(cond[-self.calc_num:])

        elif signal_type.startswith(("divergence", "convergence")):
            # CoFeature must be included in divergence
            cofeature = self.df[signal_conditions["cofeature"]].values.copy()

            # To assure similar scale, divergence features are always zscored
            feature = utils.rolling_zscore(feature, 250)
            cofeature = utils.rolling_zscore(cofeature, 250)
            if "trend" in signal_conditions:
                feature = self._trend(feature, signal_conditions["trend"])
                cofeature = self._trend(cofeature, signal_conditions["trend"])
            if "direction" in signal_conditions:
                dir_mask, diff = self._direction(feature, cofeature, signal_conditions["direction"])
                if not np.any(dir_mask):  # Ticker has never had specified trend direction
                    # Condition can never occur if the required direction never occurs; returns all False
                    return pd.Series(np.full(len(self.df), False, dtype=bool))
            else:
                diff = abs(feature - cofeature)
                dir_mask = np.full(len(feature), True, dtype=bool)

            # Acts as a placeholder so conditions series has same index as entire dataframe
            ph = np.full(len(feature), False, dtype=bool)
            valid_indices = np.nonzero(dir_mask)[0]
            diff = diff[valid_indices]

            for cond_type, cond_value in signal_conditions.items():
                if cond_type == "above":
                    above = self._above_below(diff, cond_value)
                    cond = diff >= above
                    ph[valid_indices] = cond
                    cond = ph
                elif cond_type == "below":
                    below = self._above_below(diff, cond_value)
                    cond = diff <= below
                    ph[valid_indices] = cond
                    cond = ph
                elif cond_type == "length":
                    length, rolling_sum = self._length(conditions[-1], cond_value)
                    cond = rolling_sum == length
                elif cond_type == "current_trend":
                    cond = self._current_trend(cond_value)
                else:
                    continue

                if self.full:
                    conditions.append(cond)
                else:
                    conditions.append(cond[-self.calc_num:])

        conditions_array = np.column_stack(conditions)
        # Checks where all conditions are true for a period
        if self.full:
            idx = self.df.index
        else:
            idx = self.df.index[-self.calc_num:]
        return pd.Series(np.all(conditions_array, axis=1), index=idx)

    def _above_below(self, feature, threshold, type="threshold"):
        """
        Gathers points of comparison for thresdhold and crossover-based conditions.

        Args:
            feature: The self.df column containing the comparison point.
            threshold: If a string, the comparison point is a percentile of param feature (i.e. "75th").
                If not, its a scalar (i.e. 30).
            type: Equals "threshold" or "crossover". Threshold is true every time, for example, RSI is above 70.
                Crossover is only true, for example, when RSI is below 70 one period, and above 70 the next period.

        Returns:
            Union[threshold, [threshold, threshold]]:
                - threshold: when type equals "threshold", returns one comparison point.
                - [threshold, threshold]: when type equals "crossover", returns two comparison points.

        Examples:
            return 70  # When self.df["rsi20"] > 70
            return self.df["bollingerUpper"]  # When self.df["close"] > self.df["bollingerUpper"]

            return [0, 0]  # When self.df["rsi20"] crosses above 0
            return [self.df["dmiPlus"], self.df["dmiPlus"].shift(1)]  # When self.df["dmiMinus"] crosses above self.df["dmiPlus"]

        """
        if type == "threshold":
            if threshold in self.df.columns:
                t = self.df[threshold].values.copy()
                return t
            elif isinstance(threshold, str):
                return utils.percentile(feature, threshold)
            else:
                return threshold
        elif type == "crossover":
            if isinstance(threshold, str):
                t = self.df[threshold].values.copy()
                t_shifted = utils.np_shift(t, 1)
                return [t, t_shifted]
            else:
                return [threshold, threshold]

    def _length(self, condition, length):
        """
        A condition that requires another condition to satisfy for X (length) periods in a row.
        Uses rolling sum of 1's (conditions hitting) at the length required to determine when length is satisfied.

        Args:
            condition: The condition to satisfy.
            length: The number of periods in a row.

        Returns:
            [length, rolling_sum]
        """
        return [length, utils.np_rolling(condition, length, "sum")]

    def _current_trend(self, target):
        """
        When a condition requires a trend to satisfy (uptrend or downtrend).

        Args:
            target: Type of trend.

        Returns:
            pd.Series: Boolean Series of whether the correct trend is satisfied.
        """
        return self.df["current_trend"].values == target

    def _trend(self, feature, period):
        """
        Fits feature to a trendline at X (period) periods.

        Args:
            feature: The feature to fit.
            period: Number of periods to fit.

        Returns:
            pd.Series: Rate of change in X.
        """
        return utils.line_best_fit(feature, period, return_as="numpy")

    def _direction(self, feature, cofeature, direction):
        """
        The direction in which to measure convergence and divergence.

        Args:
            feature: The self.df column containing the feature being tested for signals.
            cofeature: The self.df column containing the feature's chosen comparison point.
            direction: 1 or -1. 1 is only when feature > cofeature; -1 is only when feature < cofeature.

        Returns:
            list: First item is a mask (np.array) containing only values adhering to the correct direction.
                Second item is an empty array with len equal to len(dir_mask).
        """
        if direction == 1:
            dir_mask = feature > cofeature
            diff = np.zeros_like(feature)
            diff[dir_mask] = abs(feature[dir_mask] - cofeature[dir_mask])
        else:
            dir_mask = feature < cofeature
            diff = np.zeros_like(feature)
            diff[dir_mask] = abs(feature[dir_mask] - cofeature[dir_mask])
        return [dir_mask, diff]

    def total_indicator_score(self):
        """
        Sums all indicator signal scores to summarize price direction indication for the period.
        Generates self.df columns: total_score

        Returns:
            None
        """
        signal_columns = [col for col in self.df.columns if col.endswith("_ind")]
        total_score = self.df[[col for col in signal_columns]].sum(axis=1)
        self.df["total_score"] = total_score.rolling(3).mean()
        return

    def candlestick_patterns(self):
        """
        Tests price data for over 40 candlestick patterns.
        Generates self.df columns: candlestick

        Returns:
            None
        """
        df = self.df[["open", "high", "low", "close"]].copy()

        # Data preparation
        candle_body = df["close"] - df["open"]
        upper_tail = np.where(
            candle_body >= 0, df["high"] - df["close"], df["high"] - df["open"]
        )
        lower_tail = np.where(
            candle_body >= 0, df["open"] - df["low"], df["close"] - df["low"]
        )
        midpoint = (df["open"] + df["close"]) / 2

        # Format offset columns to gather historical relationships
        shifts = {}
        for i in range(1, 5):
            for column in ["open", "high", "low", "close", "midpoint"]:
                if column == "midpoint":
                    shifts[f"{column}-{i}"] = midpoint.shift(i)
                else:
                    shifts[f"{column}-{i}"] = df[column].shift(i)

        # Assistive functions/variables
        green_candles = candle_body[candle_body >= 0]
        red_candles = candle_body[candle_body < 0]

        # Definitions for common pattern qualifiers based on percentiles
        long_green, long_red = [
            np.percentile(green_candles, 60),
            np.percentile(abs(red_candles), 60)
        ]
        short_green, short_red = [
            np.percentile(green_candles, 40),
            np.percentile(abs(red_candles), 40)
        ]
        doji, price_match = [
            np.percentile(abs(candle_body), 7.5),
            np.percentile(abs(candle_body), 4)
        ]
        long_upper_tail, long_lower_tail = [
            np.percentile(upper_tail, 80),
            np.percentile(lower_tail, 80)
        ]
        short_upper_tail, short_lower_tail = [
            np.percentile(upper_tail, 20),
            np.percentile(lower_tail, 20)
        ]

        # Candlestick selection functions

        # 5-candle bullish
        def bullish_breakaway():
            return (
                    ((shifts["open-4"] - shifts["close-4"]) >= long_red) &
                    ((shifts["open-3"] - shifts["close-3"]) <= short_red) &
                    (shifts["open-3"] > shifts["close-3"]) &
                    ((shifts["open-1"] - shifts["close-1"]) <= short_red) &
                    (shifts["open-1"] > shifts["close-1"]) &
                    (df["close"] > df["open"]) &
                    (shifts["close-4"] > shifts["open-3"]) &
                    (shifts["close-1"] < shifts["close-3"]) &
                    (shifts["close-1"] < shifts["midpoint-2"]) &
                    (shifts["close-4"] > df["close"]) &
                    (df["close"] > shifts["open-3"])
            )

        def bullish_ladder():
            return (
                    ((shifts["open-4"] - shifts["close-4"]) >= long_red) &
                    ((shifts["open-3"] - shifts["close-3"]) >= long_red) &
                    ((shifts["open-2"] - shifts["close-2"]) >= long_red) &
                    (shifts["close-1"] < shifts["open-1"]) &
                    (df["close"] > df["open"]) &
                    (shifts["open-4"] > shifts["open-3"]) &
                    (shifts["open-3"] > shifts["open-2"]) &
                    (shifts["open-2"] > shifts["open-1"]) &
                    (shifts["close-4"] > shifts["close-3"]) &
                    (shifts["close-3"] > shifts["close-2"]) &
                    (shifts["close-2"] > shifts["close-1"]) &
                    ((shifts["high-1"] - shifts["open-1"]) > (shifts["open-1"] - shifts["close-1"])) &
                    (df["open"] > shifts["open-1"]) &
                    (df["close"] > shifts["high-1"])
            )

        # 5-candle bearish
        def bearish_breakaway():
            return (
                    ((shifts["close-4"] - shifts["open-4"]) >= long_green) &
                    ((shifts["close-3"] - shifts["open-3"]) <= short_green) &
                    (shifts["open-3"] < shifts["close-3"]) &
                    ((shifts["close-1"] - shifts["open-1"]) <= short_green) &
                    (shifts["open-1"] < shifts["close-1"]) &
                    (df["close"] < df["open"]) &
                    (shifts["close-4"] < shifts["open-3"]) &
                    (shifts["close-1"] > shifts["close-3"]) &
                    (shifts["close-1"] > shifts["midpoint-2"]) &
                    (shifts["close-4"] < df["close"]) &
                    (df["close"] < shifts["open-3"])
            )

        def bearish_ladder():
            return (
                    ((shifts["close-4"] - shifts["open-4"]) >= long_green) &
                    ((shifts["close-3"] - shifts["open-3"]) >= long_green) &
                    ((shifts["close-2"] - shifts["open-2"]) >= long_green) &
                    (shifts["close-1"] > shifts["open-1"]) &
                    (df["close"] < df["open"]) &
                    (shifts["open-4"] < shifts["open-3"]) &
                    (shifts["open-3"] < shifts["open-2"]) &
                    (shifts["open-2"] < shifts["open-1"]) &
                    (shifts["close-4"] < shifts["close-3"]) &
                    (shifts["close-3"] < shifts["close-2"]) &
                    (shifts["close-2"] < shifts["close-1"]) &
                    ((shifts["open-1"] - shifts["low-1"]) > (shifts["close-1"] - shifts["open-1"])) &
                    (df["open"] < shifts["open-1"]) &
                    (df["close"] < shifts["low-1"])
            )

        # 3-candle bullish
        def bullish_stick_sandwich():
            return (
                    (shifts["close-2"] < shifts["open-2"]) &
                    (shifts["open-1"] > shifts["close-2"]) &
                    (shifts["close-1"] > shifts["open-2"]) &
                    (shifts["open-2"] > shifts["open-1"]) &
                    (df["open"] > shifts["open-1"]) &
                    (df["close"] < df["open"]) &
                    (abs(df["close"] - shifts["close-2"]) <= price_match)
            )

        def bullish_unique_three_rivers():
            return (
                    ((shifts["open-2"] - shifts["close-2"]) >= long_red) &
                    (shifts["close-1"] > shifts["close-2"]) &
                    (shifts["close-1"] < shifts["open-1"]) &
                    (shifts["open-1"] < shifts["open-2"]) &
                    (shifts["low-1"] < shifts["low-2"]) &
                    ((df["close"] - df["open"]) <= short_green) &
                    (df["close"] > df["open"]) &
                    (df["open"] > shifts["low-1"])
            )

        def bullish_morning_star():
            return (
                    ((shifts["open-2"] - shifts["close-2"]) >= long_red) &
                    ((abs(shifts["close-1"] - shifts["open-1"])) <= short_green) &
                    ((df["close"] - df["open"]) >= long_green) &
                    (shifts["close-2"] > shifts["open-1"]) &
                    (shifts["close-2"] > shifts["close-1"]) &
                    (shifts["close-1"] < df["open"]) &
                    (df["open"] > shifts["open-1"]) &
                    (df["close"] > shifts["midpoint-2"])
            )

        def bullish_tri_star():
            return (
                    (abs(shifts["close-2"] - shifts["open-2"]) <= doji) &
                    (abs(shifts["close-1"] - shifts["open-1"]) <= doji) &
                    (abs(df["close"] - df["open"]) <= doji) &
                    (shifts["midpoint-1"] < shifts["low-2"]) &
                    (shifts["midpoint-1"] < df["low"])
            )

        def bullish_three_white_soldiers():
            return (
                    (shifts["open-2"] < shifts["open-1"]) &
                    (shifts["open-1"] < df["open"]) &
                    (shifts["close-2"] < shifts["close-1"]) &
                    (shifts["close-1"] < df["close"]) &
                    ((shifts["high-2"] - shifts["close-2"]) <= short_upper_tail) &
                    ((shifts["high-1"] - shifts["close-1"]) <= short_upper_tail) &
                    ((df["high"] - df["close"]) <= short_upper_tail)
            )

        # 3-candle bearish
        def bearish_three_black_crows():
            return (
                    (shifts["open-2"] > shifts["open-1"]) &
                    (shifts["open-1"] > df["open"]) &
                    (shifts["close-2"] > shifts["close-1"]) &
                    (shifts["close-1"] > df["close"]) &
                    ((shifts["close-2"] - shifts["low-2"]) <= short_lower_tail) &
                    ((shifts["close-1"] - shifts["low-1"]) <= short_lower_tail) &
                    ((df["close"] - df["low"]) <= short_lower_tail)
            )

        def bearish_evening_star():
            return (
                    ((shifts["close-2"] - shifts["open-2"]) >= long_green) &
                    ((abs(shifts["close-1"] - shifts["open-1"])) <= short_green) &
                    ((df["open"] - df["close"]) >= long_red) &
                    (shifts["close-2"] < shifts["open-1"]) &
                    (shifts["close-2"] < shifts["close-1"]) &
                    (shifts["close-1"] > df["open"]) &
                    (df["open"] < shifts["open-1"]) &
                    (df["close"] < shifts["midpoint-2"])
            )

        def bearish_tri_star():
            return (
                    (abs(shifts["close-2"] - shifts["open-2"]) <= doji) &
                    (abs(shifts["close-1"] - shifts["open-1"]) <= doji) &
                    (abs(df["close"] - df["open"]) <= doji) &
                    (shifts["midpoint-1"] > shifts["high-2"]) &
                    (shifts["midpoint-1"] > df["high"])
            )

        # 2-candle bullish
        def bullish_engulfing():
            return (
                    (df["close"] > shifts["open-1"]) &
                    (shifts["open-1"] > shifts["close-1"]) &
                    (shifts["close-1"] > df["open"])
            )

        def bullish_meeting_lines():
            return (
                    ((shifts["open-1"] - shifts["close-1"]) >= long_red) &
                    (df["close"] > df["open"]) &
                    (abs(df["close"] - shifts["close-1"]) <= price_match)
            )

        def bullish_harami():
            return (
                    ((shifts["open-1"] - shifts["close-1"]) >= long_red) &
                    (shifts["open-1"] > df["close"]) &
                    (df["close"] > df["open"]) &
                    (df["open"] > shifts["close-1"])
            )

        def bullish_harami_cross():
            return (
                    ((shifts["open-1"] - shifts["close-1"]) >= long_red) &
                    (abs(df["close"] - df["open"]) <= doji) &
                    (shifts["open-1"] > midpoint) &
                    (shifts["close-1"] < midpoint)
            )

        def bullish_piercing_line():
            return (
                    ((shifts["open-1"] - shifts["close-1"]) >= long_red) &
                    (df["open"] < shifts["close-1"]) &
                    (df["close"] > shifts["midpoint-1"])
            )

        def bullish_kicking():
            return (
                    (shifts["close-1"] < shifts["open-1"]) &
                    (df["close"] > df["open"]) &
                    ((shifts["high-1"] - shifts["open-1"]) <= short_upper_tail) &
                    ((df["high"] - df["close"]) <= short_upper_tail) &
                    ((shifts["close-1"] - shifts["low-1"]) <= short_lower_tail) &
                    ((df["open"] - df["low"]) <= short_lower_tail) &
                    (shifts["open-1"] < df["open"])
            )

        def bullish_homing_pigeon():
            return (
                    ((shifts["open-1"] - shifts["close-1"]) >= long_red) &
                    ((df["open"] - df["close"]) <= short_red) &
                    (df["open"] > df["close"]) &
                    (shifts["open-1"] > df["open"]) &
                    (shifts["close-1"] < df["close"])
            )

        def bullish_matching_low():
            return (
                    (shifts["close-1"] < shifts["open-1"]) &
                    (df["close"] < df["open"]) &
                    (abs(shifts["close-1"] - df["close"]) <= price_match)
            )

        def bullish_doji_star():
            return (
                    ((shifts["open-1"] - shifts["close-1"]) >= long_red) &
                    (abs(df["close"] - df["open"]) <= doji) &
                    (midpoint < shifts["close-1"])
            )

        # 2-candle bearish
        def bearish_shooting_star():
            return (
                    (midpoint > shifts["close-1"]) &
                    (shifts["close-1"] > shifts["open-1"]) &
                    ((abs(df["close"] - df["open"])) <= short_green) &
                    (((df["open"] - df["low"]) <= short_lower_tail) |
                     ((df["close"] - df["low"]) <= short_lower_tail)) &
                    (((df["high"] - df["open"]) >= long_upper_tail) |
                     ((df["high"] - df["close"]) >= long_upper_tail))
            )

        def bearish_engulfing():
            return (
                    (shifts["close-1"] > shifts["open-1"]) &
                    (df["close"] < df["open"]) &
                    (shifts["close-1"] < df["open"]) &
                    (shifts["open-1"] > df["close"])
            )

        def bearish_meeting_lines():
            return (
                    ((shifts["close-1"] - shifts["open-1"]) >= long_green) &
                    (df["close"] < df["open"]) &
                    (abs(df["close"] - shifts["close-1"]) <= price_match)
            )

        def bearish_harami():
            return (
                    ((shifts["close-1"] - shifts["open-1"]) >= long_green) &
                    (shifts["open-1"] < df["close"]) &
                    (df["close"] < df["open"]) &
                    (df["open"] < shifts["close-1"])
            )

        def bearish_harami_cross():
            return (
                    ((shifts["close-1"] - shifts["open-1"]) >= long_green) &
                    (abs(df["close"] - df["open"]) <= doji) &
                    (shifts["open-1"] < midpoint) &
                    (shifts["close-1"] > midpoint)
            )

        def bearish_dark_cloud_cover():
            return (
                    ((shifts["close-1"] - shifts["open-1"]) >= long_green) &
                    (df["open"] > shifts["close-1"]) &
                    (df["close"] < shifts["midpoint-1"])
            )

        def bearish_kicking():
            return (
                    (shifts["close-1"] > shifts["open-1"]) &
                    (df["close"] < df["open"]) &
                    ((shifts["high-1"] - shifts["close-1"]) <= short_upper_tail) &
                    ((df["high"] - df["open"]) <= short_upper_tail) &
                    ((shifts["open-1"] - shifts["low-1"]) <= short_lower_tail) &
                    ((df["close"] - df["low"]) <= short_lower_tail) &
                    (shifts["open-1"] > df["open"])
            )

        def bearish_matching_high():
            return (
                    (shifts["close-1"] > shifts["open-1"]) &
                    (df["close"] > df["open"]) &
                    (abs(shifts["close-1"] - df["close"]) <= price_match)
            )

        # 1-candle bullish
        def bullish_belt_hold():
            return (
                    ((df["close"] - df["open"]) >= long_green) &
                    ((df["high"] - df["close"]) >= short_upper_tail) &
                    ((df["open"] - df["low"]) <= short_lower_tail)
            )

        def bullish_inverted_hammer():
            return (
                    ((abs(df["close"] - df["open"])) <= short_green) &
                    (((df["open"] - df["low"]) <= short_lower_tail) |
                     ((df["close"] - df["low"]) <= short_lower_tail)) &
                    (((df["high"] - df["open"]) >= long_upper_tail) |
                     ((df["high"] - df["close"]) >= long_upper_tail))
            )

        def bullish_hammer():
            return (
                    (shifts["close-1"] > df["close"]) &
                    ((abs(df["close"] - df["open"])) <= short_green) &
                    (((df["high"] - df["open"]) <= short_upper_tail) |
                     ((df["high"] - df["close"]) <= short_upper_tail)) &
                    (((df["open"] - df["low"]) >= long_lower_tail) |
                     ((df["close"] - df["low"]) >= long_lower_tail))
            )

        # 1-candle bearish
        def bearish_belt_hold():
            return (
                    ((df["open"] - df["close"]) >= long_red) &
                    ((df["high"] - df["open"]) <= short_upper_tail) &
                    ((df["close"] - df["low"]) >= short_lower_tail)
            )

        def bearish_hanging_man():
            return (
                    (shifts["close-1"] < df["close"]) &
                    ((abs(df["close"] - df["open"])) <= short_green) &
                    (((df["high"] - df["open"]) <= short_upper_tail) |
                     ((df["high"] - df["close"]) <= short_upper_tail)) &
                    (((df["open"] - df["low"]) >= long_lower_tail) |
                     ((df["close"] - df["low"]) >= long_lower_tail))
            )

        patterns = [
            bullish_breakaway, bullish_ladder, bearish_breakaway, bearish_ladder,
            bullish_stick_sandwich, bullish_unique_three_rivers, bullish_morning_star, bullish_tri_star,
            bullish_three_white_soldiers, bearish_three_black_crows, bearish_evening_star, bearish_tri_star,
            bullish_engulfing, bullish_meeting_lines, bullish_harami, bullish_harami_cross,
            bullish_piercing_line, bullish_kicking, bullish_homing_pigeon, bullish_matching_low,
            bullish_doji_star, bearish_shooting_star, bearish_engulfing, bearish_meeting_lines,
            bearish_harami, bearish_harami_cross, bearish_dark_cloud_cover, bearish_kicking,
            bearish_matching_high, bullish_belt_hold, bullish_inverted_hammer, bullish_hammer,
            bearish_belt_hold, bearish_hanging_man
        ]
        # Checks for occurrences of all above candlestick patterns
        all_patterns = {func.__name__: func() for func in patterns}
        return all_patterns
