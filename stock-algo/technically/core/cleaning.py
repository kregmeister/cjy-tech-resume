#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:59:57 2024

@author: cjymain
"""
from technically.utils.handlers.db import PostgreSQL
from technically.utils.log import get_logger, timer
from technically.const import TC_PATH, MIN_PERIODS

import traceback
from glob import glob
import shutil


class DatabaseChecks:
    """
    Deletes dead_ticker data and ensures data reliability.
    """

    @timer()
    def execute(self):
        # Open multiple database connections simultaneously
        with PostgreSQL() as self.conn:
            extract_tickers_query = '''
                SELECT
                    table_alias as ticker,
                    exchange,
                    cap_category,
                    sector,
                    designation,
                    prices_initialized,
                    fundamentals_initialized
                FROM 
                    prices.metadata;
                '''
            self.tickers_df = self.conn.run_query(
                extract_tickers_query,
                return_as="pandas"
            )

        self.remove_dead_tickers()
        self.remove_duplicate_parquet()

    def remove_duplicate_parquet(self):
        """
        When a ticker's parquet partition changes (I.E. large cap --> mega cap), the old parquet must be removed.

        Returns:
            None
        """

        for idx, row in self.tickers_df.iterrows():
            # How parquet directories format spaces
            try:
                row['sector'] = row['sector'].replace(' ', '%20')
            except AttributeError:  # Sector is None
                continue

            for pq_type in ["daily", "quarterly"]:
                ticker_parquet_paths = glob(
                    TC_PATH + f"/parquet/{pq_type}/**/ticker={row['ticker']}",
                    recursive=True
                )
                if len(ticker_parquet_paths) > 1:
                    for path in ticker_parquet_paths:
                        # Path to ticker's parquet does not match its metadata (metadata has changed)
                        if path != (f"{TC_PATH}/parquet/{pq_type}/exchange={row['exchange']}/"
                                    f"cap={row['cap_category']}/sector={row['sector']}/ticker={row['ticker']}"):
                            shutil.rmtree(path)
                            get_logger().info(
                                f"Removed duplicate {pq_type} parquet.", extra={
                                    "item_id": row["ticker"],
                                    "path": path
                                }
                            )

    def remove_dead_tickers(self):
        """
        Removes all traces of dead tickers (except for its metadata entry).

        Returns:
            None
        """

        # Extracts tickers designated as 'dead' and ensures that have not already been removed from databases
        dead_tickers = self.tickers_df['ticker'][
            (self.tickers_df['designation'] == 'dead_ticker') &
            (self.tickers_df['prices_initialized'] == True) &
            (self.tickers_df['fundamentals_initialized'] == True)
        ].tolist()

        for ticker in dead_tickers:
            # Updates its metadata entry
            dead_ticker_meta_query = '''
                UPDATE
                    prices.metadata
                SET
                    prices_initialized = false,
                    newest_db_prices = DEFAULT,
                    fundamentals_initialized = false,
                    newest_db_statements = DEFAULT
                WHERE
                    table_alias = :ticker;
                '''
            self.conn.run_query(
                dead_ticker_meta_query,
                params={"ticker": ticker},
                commit=True
            )

            # Drops price table
            drop_price_query = '''
                DROP TABLE IF EXISTS prices.{ticker};
                '''
            self.conn.run_query(
                drop_price_query,
                params={"ticker": ticker},
                commit=True
            )

            # Drops fundamentals tables
            drop_inc_stmt_query = '''
                DROP TABLE IF EXISTS fundamentals.{ticker}_income_statement;
                '''
            self.conn.run_query(
                drop_inc_stmt_query,
                params={"ticker": ticker},
                commit=True
            )
            drop_balance_sheet_query = '''
                DROP TABLE IF EXISTS fundamentals.{ticker}_balance_sheet;
                '''
            self.conn.run_query(
                drop_balance_sheet_query,
                params={"ticker": ticker},
                commit=True
            )
            drop_cash_flow_query = '''
                DROP TABLE IF EXISTS fundamentals.{ticker}_cash_flow;
                '''
            self.conn.run_query(
                drop_cash_flow_query,
                params={"ticker": ticker},
                commit=True
            )

            # Drops models data
            drop_models_query = '''
                DELETE FROM models.signal_success_rates 
                    WHERE ticker = :ticker;
                '''
            self.conn.run_query(
                drop_models_query,
                params={"ticker": ticker},
                commit=True
            )

            # Removes parquet files associated with ticker
            for pq_path in glob(TC_PATH + f"/parquet/**/ticker={ticker}", recursive=True):
                try:
                    shutil.rmtree(pq_path)
                except Exception:
                    get_logger().error(
                        "A dead ticker could not have its parquet removed.", extra={
                            "item_id": ticker,
                            "path": pq_path,
                            "error": traceback.format_exc()
                        }
                    )
                    continue

                get_logger().info(
                    "Dead ticker's data deleted.", extra={
                        "item_id": ticker,
                        "path": pq_path
                    }
                )


class PriceAdjustments:
    """
    Examines price data for corporate actions (splits, dividends) and adjusts or un-adjusts for them.
    """

    def __init__(self, price_df, columns_to_adjust=("open", "high", "low", "close", "volume")):
        self.df = price_df
        self.columns = columns_to_adjust

    def price_activity_check(self, ticker: str, threshold=25000):
        """
        Cuts out data points where price activity is lower than threshold.
        Tickers are assigned different designations based on whether the most recent period is cut off ("dead_ticker")
        or if the valid stretches of data are less than MIN_PERIODS periods ("not_enough_data").

        Args:
            ticker (str): Ticker symbol.
            threshold (int): The minimum 5-day average in dollars traded (share price * volume) for a ticker to remain active.
        """

        with PostgreSQL() as conn:
            self.df["date"] = self.df["date"].astype(str)

            df_len = len(self.df)

            # Tickers with less than global var MIN_PERIODS rows skipped
            if df_len < MIN_PERIODS:
                designation_update_query = '''
                   UPDATE
                       prices.metadata
                   SET designation = 'not_enough_data'
                   WHERE 
                     table_alias = :ticker
                     AND is_active = true;
                    '''
                conn.run_query(
                    designation_update_query,
                    params={"ticker": ticker},
                    commit=True
                )

                get_logger().warning(
                    "Not enough data to conduct analysis.", extra={
                        "item_id": ticker
                    }
                )
                return False

            # Finds where a 5-day MA of dollarsTraded is less than threshold
            dollarsTraded = self.df["close"] * self.df["volume"]
            weekly_volume_sums = dollarsTraded.rolling(window=20).mean()
            violations = self.df["date"][weekly_volume_sums <= threshold].tolist()

            # All dates pass check
            if violations == []:
                return True

            # Filter violations out
            df_filtered = self.df[~self.df["date"].isin(violations)].copy()

            # Measures length of passing sequences
            index_diffs = df_filtered.index.to_series().diff().fillna(1)
            sequences = (index_diffs != 1).cumsum()

            # Filters out failing sequences
            df_filtered.loc[:, 'idx_group'] = sequences
            sequence_sizes = df_filtered.groupby("idx_group").size()

            # Filters out passing sequences less than MIN_PERIODS
            valid_sequences = sequence_sizes[sequence_sizes > MIN_PERIODS].index

            # Final filter
            passing_dates = tuple(
                df_filtered["date"][df_filtered['idx_group'].isin(valid_sequences)]
            )

            if passing_dates == ():  # No passing sequences
                set_dead_ticker_query = '''
                    UPDATE
                        prices.metadata
                    SET 
                        designation = 'dead_ticker',
                        newest_db_prices = DEFAULT
                    WHERE 
                        table_alias = :ticker;
                   '''
                conn.run_query(
                    set_dead_ticker_query,
                    params={"ticker": ticker},
                    commit=True
                )
                get_logger().info(
                    "Ticker inactive and designated as 'dead_ticker'.", extra={
                        "item_id": ticker
                    }
                )
                return False
            elif passing_dates[-1] == self.df.date.values[-1]:  # Most recent data passes
                remove_inactive_rows_query = '''
                    DELETE FROM 
                        prices.{ticker}
                    WHERE 
                        date < :date_cutoff;
                    '''
                conn.run_query(
                    remove_inactive_rows_query,
                    params={"ticker": ticker, "date_cutoff": passing_dates[0]},
                    commit=True
                )
                get_logger().info(
                    f"Dates preceding {passing_dates[0]} have been removed due to inadequate price activity.", extra={
                        "item_id": ticker
                    }
                )
                return True
            else:  # Some historical data passes
                remove_inactive_rows_query = '''
                    DELETE FROM 
                        prices.{ticker}
                    WHERE 
                        date > :date_cutoff;
                    '''
                conn.run_query(
                    remove_inactive_rows_query,
                    params={"ticker": ticker, "date_cutoff": passing_dates[-1]},
                    commit=True
                )

                set_recent_inactive_query = '''
                    UPDATE 
                        prices.metadata
                    SET 
                        designation = 'recent_dates_inactive',
                        newest_db_prices = :date_cutoff
                    WHERE 
                        table_alias = :ticker;
                    '''
                conn.run_query(
                    set_recent_inactive_query,
                    params={"ticker": ticker, "date_cutoff": passing_dates[-1]},
                    commit=True
                )
                get_logger().info(
                    f"Dates exceeding {passing_dates[-1]} have been removed due to inadequate price activity.", extra={
                        "item_id": ticker
                    }
                )
                return False

def adjust_for_splits(df):
    """
    Adjusts incoming Tiingo price data for splits, accounting for API inconsistency in applying stock splits.

    Args:
        df (pd.DataFrame): Dataframe containing incoming Tiingo price data.

    Returns:
        df (pd.DataFrame): Dataframe containing adjusted and rounded Tiingo price data.
    """
    for idx in df.index[df["split_factor"] != 1]:
        if idx == 0:
            continue
        close = df["close"].loc[idx]

        # Close previous the split with the split_factor applied
        prev_close = df["close"].loc[idx - 1] / df["split_factor"].loc[idx]
        likely_split = prev_close + (prev_close * 0.25) > close > prev_close - (prev_close * 0.25)
        if not likely_split:
            continue

        for column in df.columns:  # Applies split-factor
            if column == "volume":
                df.loc[:idx - 1, column] = (
                        df.loc[:idx - 1, column] * df.loc[idx, "split_factor"]
                ).round().astype("int64")
            elif column in ["open", "high", "low", "close"]:
                df.loc[:idx - 1, column] /= df.loc[idx, "split_factor"]
    return df.round(4)


def unadjust_for_splits(df):  # Currently unused
    for idx in df.index[df["split_factor"] != 1]:  # Finds splits
        for column in df.columns:  # Applies split-factor
            if column == "volume":
                df.loc[:idx - 1, column] = (
                        df.loc[:idx - 1, column] / df.loc[idx, "split_factor"]
                ).round().astype("int64")
            elif column in ["open", "high", "low", "close"]:
                df.loc[:idx - 1, column] *= df.loc[idx, "split_factor"]
    return df.round(4)

def market_cap_category(cap: int | float):
    """
    Assigns a market cap category for a ticker based on its most current market cap value.

    Args:
        cap: Most current market cap value.

    Returns:
        category (str): Market cap category.
    """
    try:
        if cap >= 2.0 * (10 ** 11):
            return "mega"
        elif cap >= 1.0 * (10 ** 10):
            return "large"
        elif cap >= 2.0 * (10 ** 9):
            return "mid"
        elif cap >= 3.0 * (10 ** 8):
            return "small"
        else:
            return "micro"
    except TypeError:  # Market cap is null
        return "unknown"

