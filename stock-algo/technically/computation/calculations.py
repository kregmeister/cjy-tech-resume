#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:23:24 2025

@author: cjymain
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

from technically.const import TC_PATH, CURRENT_DATE, INCREMENTAL_CALC_COLS
from technically.computation.formulas import TechnicalFormulas
from technically.computation.signals import TechnicalIndicatorSignals as Signals
from technically.core.cleaning import PriceAdjustments
from technically.computation.backtesting import SignalSuccessRates
from technically.utils.handlers.db import PostgreSQL, get_duckdb
from technically.utils.log import get_logger, timer
from duckdb import IOException


class TechnicalsExecutor:
    """
    Facilitates the recursive calculation, scoring, and storing of candlesticks, technical indicators, EMAs, etc.
    """

    def __init__(self, ticker: str, exchange: str, cap: str, sector: str, last_processed, initialized: bool, backtest_success_rates):
        self.ticker = ticker
        self.exchange = exchange
        self.cap = cap
        self.sector = sector
        self.last_processed = last_processed
        self.initialized = initialized
        if not initialized:
            self.calc_period = np.inf
        else:
            self.calc_period = abs(CURRENT_DATE - last_processed).days
        self.backtest_success_rates = backtest_success_rates[backtest_success_rates["sector"] == sector]
        self.pq_path = TC_PATH + f"/parquet/daily/exchange={exchange}/cap={cap}/sector={sector}/ticker={ticker}/*.parquet"

    def execute(self):
        raw_df = self.extract()
        if not isinstance(raw_df, pd.DataFrame):
            return
        integrity = self.check(raw_df)

        # Whether data passed integrity tests
        if integrity:
            calc_df = self.calculate(raw_df)
            scored_df = self.score(calc_df)
            self.backtest(scored_df)
            self.load(scored_df)

    @timer()
    def extract(self):
        """
        Gathers price data from DuckDB and stores it in self.df (pd.DataFrame).

        Returns:
            df (pd.DataFrame): All price data.
        """
        with PostgreSQL() as conn:
            # Acquires price data for ticker from prices.duck
            extract_query = '''
                SELECT
                    '{ticker}' AS ticker,
                    '{exchange}' AS exchange,
                    '{cap}' AS cap,
                    '{sector}' AS sector,
                    *
                FROM
                    prices.{table}
                ORDER BY
                    date;
                '''
            df = conn.run_query(
                extract_query,
                params={
                    "ticker": self.ticker,
                    "exchange": self.exchange,
                    "cap": self.cap,
                    "sector": self.sector,
                    "table": self.ticker
                },
                return_as="pandas"
            )
            return df

    @timer()
    def merge(self, df: pd.DataFrame, cols: list, pkey: str = "date"):
        """
        Merges incremental calculations with full calculations (retrieved from existing parquet file).

        Args:
            df (pd.DataFrame): The incremental calculations df to merge.
            cols (list): A list of columns to merge.
            pkey: Column to merge on. Defaults to "date".

        Returns:
            df (pd.DataFrame): The merged df.
        """
        # Extracts full existing calculations from existing parquet file to df
        cols_str = ", ".join(cols)
        try:
            existing_calc_df = get_duckdb().execute(f'''
                SELECT
                    date,
                    {cols_str}
                FROM
                    '{self.pq_path}'
                WHERE date <= '{self.last_processed}'
                ORDER BY
                    date;
                '''
            ).df()
        except IOException as e:
            with PostgreSQL() as conn:
                process_reset_query = '''
                    UPDATE prices.metadata SET last_processed = DEFAULT 
                    WHERE table_alias = :ticker
                    '''
                conn.run_query(
                    process_reset_query,
                    params={
                        "ticker": self.ticker,
                    }
                )
            raise e

        # Ensures "date" column is date type and is properly applied to both dataframes
        existing_calc_df[pkey] = pd.to_datetime(existing_calc_df[pkey]).dt.date
        incremental_calc_df = df[[pkey] + cols].copy()
        incremental_calc_df[pkey] = pd.to_datetime(incremental_calc_df[pkey]).dt.date

        df_merged = pd.merge(
            existing_calc_df,
            incremental_calc_df,
            on=pkey,
            how='outer',
            suffixes=('_old', '_new')
        )

        # Existing data takes precedence over new data where dates overlap
        for col in cols:
            df_merged[col] = df_merged[f'{col}_old'].combine_first(df_merged[f'{col}_new'])

        df_merged = df_merged.drop(
            columns=[col for col in df_merged.columns if col.endswith('_old') or col.endswith('_new')]
        )
        df_merged.drop("date", axis=1, inplace=True)
        df[cols] = df_merged
        return df

    @timer()
    def check(self, df: pd.DataFrame):
        """
        Triggers series of checks to determine whether to conduct technical analysis or not.

        Returns:
            bool: True for pass, False for fail.
        """
        # Verifies ticker has consistent price activity
        return (PriceAdjustments(df).
                price_activity_check(self.ticker))

    @timer()
    def calculate(self, df: pd.DataFrame):
        """
        Triggers calculations that produce technical indicators as new columns in self.df.

        Returns:
            df (pd.DataFrame): Full price and technical data.
        """
        # Initialize class that handles indicator calculations
        tc = TechnicalFormulas(df.copy(), self.calc_period)

        # Use self.df to solve for functions
        tc.average_true_range()
        tc.exponential_moving_average(5)
        tc.exponential_moving_average(10)
        tc.exponential_moving_average(20)
        tc.exponential_moving_average(50)
        tc.exponential_moving_average(100)
        tc.exponential_moving_average(250)
        tc.kalman_filter_single()
        tc.percent_from_extreme()
        tc.percent_from_extreme(direction="min")
        tc.demand_index()
        tc.aroon_oscillator()
        tc.commodity_channel_index()
        tc.relative_strength_index()
        tc.stdev_percent_of_sma()
        tc.short_know_sure_thing()
        tc.bollinger_bands()
        tc.moving_average_convergence_divergence()
        tc.williams_percent_r()
        tc.stochastic_oscillator()
        tc.parabolic_sar()
        tc.average_directional_index()

        # All calculated columns pulled from tc Class to variable df
        df = tc.persist()

        # If calculations have been done before, incremental data is merged with full data
        if self.initialized:
            df = self.merge(df, INCREMENTAL_CALC_COLS)

        # Advanced calculations that require full data
        tc.kaufman_adaptive_moving_average("close", 20)
        tc.on_balance_volume()
        tc.current_trend()
        tc.trend_strength_score()
        tc.price_ceilings_floors()
        tc.trend_swing_confirmation()

        df = tc.persist()
        return df

    @timer()
    def score(self, df: pd.DataFrame):
        """
        Triggers calculations that check for technical indicator signals
        (pre-defined in indicatorSignals.json) and
        assigns scores (determined by backtesting) to indicator occurrences.
        Results are assigned as new columns in self.df.

        Returns:
            df (pd.DataFrame): Full price, technical, and indicator scores data.
        """

        # Initialize class that handles indicator signal identification
        score = Signals(self.ticker, df.copy(), self.backtest_success_rates, self.calc_period)

        # The passed strings must match indicator list strings at top of indicatorSignals.json
        score.calculate("demand_idx")
        score.calculate("wpr")
        score.calculate("macd")
        score.calculate("cci")
        score.calculate("rsi")
        score.calculate("bollinger_sma")
        score.calculate("kst")
        score.calculate("stochastic_k")
        score.calculate("adx")
        score.calculate("psar")
        score.calculate("candlestick", identify=False)
        # Sums technical indicator scores
        score.total_indicator_score()
        score.reset_recent_backtest()

        df = score.persist()

        # Merge only occurs when scoring is incremental
        if not score.full:
            signal_cols = [col for col in df.columns if col.endswith("_ind")]
            df = self.merge(df, signal_cols)
        return df

    @timer()
    def load(self, df: pd.DataFrame):
        """
        Writes self.df to a parquet file (includes all columns generated from above functions).

        Returns:
            None
        """
        latest_date = df["date"].iloc[-1]

        # Converts Pandas DataFrame to PyArrow Table
        parquet_table = pa.Table.from_pandas(df)

        # Writes PyArrow Table to a parquet dataset at proper partition
        pq.write_to_dataset(
            parquet_table,
            root_path=TC_PATH + "/parquet/daily/",
            partition_cols=["exchange", "cap", "sector", "ticker"],
            compression="zstd",
            existing_data_behavior="delete_matching"  # Overwrites existing parquet files
        )

        with PostgreSQL() as conn:
            update_query = '''
                UPDATE
                    prices.metadata
                SET
                    last_processed = :date
                WHERE
                    table_alias = :ticker;
                '''
            conn.run_query(
                update_query,
                params={
                    "date": latest_date,
                    "ticker": self.ticker,
                },
                commit=True
            )
        return

    @timer()
    def backtest(self, df: pd.DataFrame):
        """
        Triggers backtesting of technical indicator signals.

        Returns:
            None
        """
        # Initiates backtesting
        SignalSuccessRates(
            # Passes only columns required for backtesting
            df[
                ["date", "current_trend", "reversal_conf"] +
                [column for column in df.columns if column.endswith("_ind")]
            ].copy(),
            self.ticker,
            self.exchange,
            self.cap,
            self.sector
        ).execute()
        return

class FundamentalsExecutor:
    """
    Facilitates the recursive calculation, scoring, and storing of candlesticks, technical indicators, EMAs, etc.
    """

    def __init__(self, ticker: str, exchange: str, cap: str, sector: str, last_processed, initialized: bool, backtest_success_rates):
        self.ticker = ticker
        self.exchange = exchange
        self.cap = cap
        self.sector = sector

    def execute(self):
        raw_df = self.extract()
        if raw_df is None:
            return
        calc_df = self.transform(raw_df)
        self.load(calc_df)

    @timer()
    def extract(self):
        """
        Gathers fundamental statements data from DuckDB and stores it in self.fund_df (pd.DataFrame).

        Returns:
            fund_df (pd.DataFrame): Joined fundamental statements data.
        """
        with PostgreSQL() as conn:
            # Identifies ticker statement tables
            statements_query = '''
                SELECT 
                    table_name 
                FROM 
                    information_schema.tables
                WHERE
                    table_schema = 'fundamentals'
                    AND table_name SIMILAR TO '{ticker}_.*'; 
                '''
            statements = conn.run_query(
                statements_query,
                params={"table": self.ticker},
                return_as="tuple"
            )
            stmts = [stmt[0] for stmt in statements]

            # Merges all available ticker statement data
            if len(stmts) == 1:
                extract_query = '''
                    SELECT
                        '{ticker}' AS ticker,
                        '{exchange}' AS exchange,
                        '{cap}' AS cap,
                        '{sector}' AS sector,
                        * 
                    FROM 
                        fundamentals.{table1}
                    ORDER BY 
                        date;
                    '''
                query_params = {
                    "ticker": self.ticker,
                    "exchange": self.exchange,
                    "cap": self.cap,
                    "sector": self.sector,
                    "table1": stmts[0]
                }
            elif len(stmts) == 2:
                extract_query = '''
                    SELECT
                        '{ticker}' AS ticker,
                        '{exchange}' AS exchange,
                        '{cap}' AS cap,
                        '{sector}' AS sector,
                        fundamentals.{table1}.*, 
                        fundamentals.{table2}.*
                    FROM 
                        fundamentals.{table1}
                        JOIN "fundamentals.{table2}" ON
                            fundamentals.{table1}.date = fundamentals.{table2}.date;
                    '''
                query_params = {
                    "ticker": self.ticker,
                    "exchange": self.exchange,
                    "cap": self.cap,
                    "sector": self.sector,
                    "table1": stmts[0],
                    "table2": stmts[1]
                }
            elif len(stmts) == 3:
                extract_query = '''
                    SELECT
                        '{ticker}' AS ticker,
                        '{exchange}' AS exchange,
                        '{cap}' AS cap,
                        '{sector}' AS sector,
                        fundamentals.{table1}.*, 
                        fundamentals.{table2}.*,
                        fundamentals.{table3}.*
                    FROM 
                        fundamentals.{table1}
                        JOIN fundamentals.{table2} ON
                            fundamentals.{table1}.date = fundamentals.{table2}.date
                            JOIN fundamentals.{table3} ON
                                fundamentals.{table2}.date = fundamentals.{table3}.date;
                    '''
                query_params = {
                    'ticker': self.ticker,
                    'exchange': self.exchange,
                    'cap': self.cap,
                    'sector': self.sector,
                    "table1": stmts[0],
                    "table2": stmts[1],
                    "table3": stmts[2]
                }
            else:  # No ticker statement tables detected
                extract_query = None
                query_params = None

            if extract_query:
                fund_df = conn.run_query(
                    extract_query,
                    params=query_params
                )
            else:
                fund_df = None
            return fund_df

    @timer()
    def transform(self, df: pd.DataFrame):
        """
        Triggers calculations that produce fundamental indicators as new columns in self.final_df (pd.DataFrame).

        Returns:
            df (pd.DataFrame): Full fundamental statements and indicators data.
        """

        # Handles duplicate date columns
        if "date" not in df.columns and "date_1" in df.columns:
            df = df.rename(columns={"date_1": "date"})
        elif "date_1" in df.columns:
            df = df.drop(columns=["date_1"])
        if "date_2" in df.columns:
            df = df.drop(columns=["date_2"])
        if "date_3" in df.columns:
            df = df.drop(columns=["date_3"])

        ratios = [
            "netMargin = df.netinc / df.revenue",
            "grossMargin = df.grossProfit / df.revenue",
            "operatingMargin = df.opinc / df.revenue",
            "returnOnEquity = df.netinc / df.equity",
            "returnOnAssets = df.netinc / df.totalAssets",
            "currentRatio = df.assetsCurrent / df.liabilitiesCurrent",
            "quickRatio = (df.cashAndEq + df.acctRec) / df.liabilitiesCurrent",
            "debtEquity = df.debt / df.equity",
            "debtAssets = df.totalLiabilities / df.totalAssets",
            "interestCoverage = df.ebit / df.intexp",
            "assetTurnover = df.revenue / df.totalAssets",
            "receivablesTurnover = df.revenue / df.acctRec"
        ]

        for formula in ratios:
            # I.E. 'netMargin'
            formula_name = formula.split("=")[0].strip()
            try:
                # Adds the variable embedded in each ratio object as a column
                df = pd.eval(formula, target=df)

                # Ensures ratio is a float
                df[formula_name] = df[formula_name].astype(np.float64)
            except (KeyError, AttributeError, ValueError) as e:
                get_logger().warning(
                    "Issue calculating a fundamental ratio.", extra={
                        "item_id": self.ticker,
                        "formula": formula_name,
                        "error": e
                    }
                )
                df[formula_name] = 0.0

        return df

    @timer()
    def load(self, df: pd.DataFrame):
        """
        Writes self.final_df to parquet file (includes all columns from above functions).

        Returns:
            None
        """
        # Converts Pandas DataFrame to PyArrow Table
        parquet_table = pa.Table.from_pandas(df)

        # Writes PyArrow Table to a parquet dataset at proper partition
        pq.write_to_dataset(
            parquet_table,
            root_path=TC_PATH + "/parquet/quarterly/",
            partition_cols=["exchange", "cap", "sector", "ticker"],
            compression="zstd",
            existing_data_behavior="delete_matching"  # Overwrites existing parquet files
        )
        return