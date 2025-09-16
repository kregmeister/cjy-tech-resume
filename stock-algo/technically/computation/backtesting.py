#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:52:22 2025

@author: cjymain
"""

import pandas as pd

from technically.utils.log import get_logger
from technically.const import CURRENT_DATE, TC_PATH
from technically.utils.handlers.db import PostgreSQL, get_duckdb


class SignalSuccessRates:
    """
    Calculates historical technical indicator success rates.
    """

    def __init__(self, df: pd.DataFrame, ticker: str, exchange: str, cap: str, sector: str):
        self.df = df
        self.ticker = ticker
        self.exchange = exchange
        self.cap = cap
        self.sector = sector
        self.conn = get_duckdb()
        self.pq_path = TC_PATH + f"/parquet/daily/exchange={exchange}/cap={cap}/sector={sector}/ticker={ticker}/*.parquet"

    def execute(self):
        with PostgreSQL() as self.conn:
            ready = self.check()
            if not ready:
                return False

            signals_df = self.test_conditions(self.df)
            self.to_psql(signals_df)
            return True

    def check(self):
        """
        Determines whether to conduct backtesting or not.

        Returns:
            bool: True for yes, False for no.
        """
        # Checks if self.ticker has been backtested
        exists_query = '''
            SELECT 
                COUNT(1) 
            FROM 
                models.backtest_success_rates 
            WHERE 
                ticker = :ticker;
            '''
        exists = self.conn.run_query(
            exists_query,
            params={'ticker': self.ticker},
            return_as="tuple"
        )[0][0]

        # Immediately triggers backtesting where no entry exists
        if exists == 0:
            return True

        # Finds the date of the latest backtest for self.ticker
        if CURRENT_DATE.weekday() == 4:  # Only re-trains on Friday
            last_trained_query = '''
                SELECT 
                    date 
                FROM 
                    models.backtest_success_rates
                WHERE 
                    ticker = :ticker
                ORDER BY
                    date DESC 
                    LIMIT 1;
                '''
            last_trained = self.conn.run_query(
                last_trained_query,
                params={'ticker': self.ticker},
                return_as="tuple"
            )[0][0]

            # Triggers backtesting if 28 days or more since last backtest
            if (CURRENT_DATE - last_trained).days >= 28:
                return True
            else:  # Less than 28 days since trained
                return False
        else:  # Not the weekend
            return False

    def test_conditions(self, df: pd.DataFrame):
        """
        Uses conditions to test whether a technical indicator signal confirms or denies.

        Args:
            df (pd.DataFrame): DataFrame containing technical indicator signals.

        Returns:
            df (pd.DataFrame): DataFrame containing counts, successes, and failures
                for each technical indicator signal for the ticker.
        """
        df["date"] = pd.to_datetime(
            df["date"], format='ISO8601'
        ).dt.date
        date = CURRENT_DATE

        # Selects each indicator signal column
        columns = [col for col in df.columns if col.endswith("_ind")]

        results = []
        for signal in columns:
            # All signals start with 'bullish_' or 'bearish_' except for candlestick
            indicator_name = signal.split("_")[1] if signal[7] == "_" else "candlestick"
            sig_direction = signal[:7]  # Equals 'bullish' or 'bearish'

            # Initialize counters
            reversal_count = 0
            reversal_success = 0
            reversal_failure = 0
            continuation_count = 0
            continuation_success = 0
            continuation_failure = 0

            # Defines masks for signal occurrence, counts them, and isolates data where they occur
            if sig_direction == "bullish":
                # Masks for occurrence of reversal & continuation signals
                bullish_rev = (df[signal] > 0) & (df["current_trend"] == -1)
                bullish_cont = (df[signal] > 0) & (df["current_trend"] == 1)
                # Count signal occurrences
                reversal_count = len(bullish_rev[bullish_rev])
                continuation_count = len(bullish_cont[bullish_cont])
                # Creates Dataframe of occurrences with columns required for confirmation
                signal_df = df[["current_trend", "reversal_conf"]][bullish_rev | bullish_cont]
            else:
                bearish_rev = (df[signal] < 0) & (df["current_trend"] == 1)
                bearish_cont = (df[signal] < 0) & (df["current_trend"] == -1)
                reversal_count = len(bearish_rev[bearish_rev])
                continuation_count = len(bearish_cont[bearish_cont])
                signal_df = df[["current_trend", "reversal_conf"]][bearish_rev | bearish_cont]

            # Iterates through signal occurrences
            for idx in signal_df.index:
                current_trend = signal_df.loc[idx, "current_trend"]  # 1 or -1

                # Constructs window for signal to confirm/deny
                end_idx = min(idx + 60, len(df) - 1)
                segment = df.loc[idx:end_idx]

                if sig_direction == "bullish":
                    if current_trend == 1:
                        # Continuation confirmed if trend persists for window
                        if all(segment["reversal_conf"] == "None"):
                            continuation_success += 1
                        # Continuation denied if trend reverses within window
                        else:
                            continuation_failure += 1
                    elif current_trend == -1:
                        # Reversal confirmed if trend reverses within window
                        if any(segment["reversal_conf"] == "to_up"):
                            reversal_success += 1
                        # Reversal denied if trend persists for window
                        else:
                            reversal_failure += 1
                elif sig_direction == "bearish":
                    if current_trend == -1:
                        if all(segment["reversal_conf"] == "None"):
                            continuation_success += 1
                        else:
                            continuation_failure += 1
                    elif current_trend == 1:
                        if any(segment["reversal_conf"] == "to_down"):
                            reversal_success += 1
                        else:
                            reversal_failure += 1

            results.append({
                "date": date,
                "ticker": self.ticker,
                "sector": self.sector,
                "cap": self.cap,
                "exchange": self.exchange,
                "indicator": indicator_name,
                #"indicator_params": {
                #    key: value for key, value in self.df_meta.items() if key.startswith(indicator_name)
                #},
                "signal_name": "_".join(signal.split("_")[:-1]),  # Removes '_ind' suffix
                "reversal_count": reversal_count,
                "reversal_success": reversal_success,
                "reversal_failure": reversal_failure,
                "continuation_count": continuation_count,
                "continuation_success": continuation_success,
                "continuation_failure": continuation_failure
            })
        return pd.DataFrame(results)

    def to_psql(self, df: pd.DataFrame):
        """
        Writes backtesting results to DuckDB.

        Args:
            df (pd.DataFrame): Resultant DataFrame.

        Returns:
            None
        """
        # Replaces old training results if exists, creates new entry if not
        with self.conn.DataFrameToPostgreSQL(self.conn, "backtest_success_rates", "models", df) as databridge:
            databridge.insert_or_replace_df_into_table("ticker")

        # Ensures signals.py exec is full next execution to properly apply new backtest success rates
        query = '''
            UPDATE prices.metadata SET recent_backtest = true 
            WHERE table_alias = :ticker;
            '''
        self.conn.run_query(
            query,
            params={
                "ticker": self.ticker,
            },
            commit=True
        )

        get_logger().info(
            "Successfully inserted backtest success rates.",
            extra={"item_id": self.ticker}
        )

