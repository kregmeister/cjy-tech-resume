#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:56:27 2024

@author: cjymain
"""

import duckdb
import threading
from sqlalchemy import create_engine
from sqlalchemy import quoted_name, text, inspect
import pandas as pd
import numpy as np
import traceback
import atexit
import os

from technically.utils.log import get_logger, timer
from technically.utils.handlers.auth import get_credentials


# PostgreSQL user name equals system user name
USER = os.getlogin()

if USER == 'craig99':
    DB_PW = get_credentials(['prod_database_password'])
else:
    DB_PW = 'Shadow123'

DATABASE_URL = f"postgresql://{USER}:{DB_PW}@localhost:5432/technically"

# Global Postgres engine
engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10)

# Global, thread-safe DuckDB in-memory connection(s)
thread_local = threading.local()
def get_duckdb():
    if not hasattr(thread_local, "db"):
        thread_local.db = duckdb.connect(":memory:")
        thread_local.db.execute(f'''
            ATTACH 'dbname=technically user={USER} host=localhost password={DB_PW}' 
            AS tc (TYPE postgres);
            '''
        )
    return thread_local.db

# Closes DB connections when Python Interpreter shuts down
def db_cleanup():
    engine.dispose()  # Postgres
    if hasattr(thread_local, "db"):  # DuckDB threads
        thread_local.db.close()
        del thread_local.db
atexit.register(db_cleanup)


class PostgreSQL:
    """
    Yields a connection to the PostgreSQL database from the pool and manages DB operations.
    """

    def __enter__(self):
        self.conn = engine.connect()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            if exc_type is not None:
                get_logger().error(
                    "Exception in class PostgreSQL context.", extra={
                        "error": traceback.format_exc()
                    }
                )
                return True
        except Exception as e:
            # Unable to close DB connection
            raise e
        finally:
            self.conn.close()

    def _format_query(self, query: str, params: dict):
        """
        Properly quotes table & schema names from query parameters.

        Args:
            params (dict): Dictionary of parameters to quote.

        Example:
            Before quoting: "SELECT * FROM 'aapl';"
            After quoting: "SELECT * FROM aapl;"

        Returns:
            Params with quoted table/schema names.

        """
        if params:
            formatters = {
                key: quoted_name(value, True)
                if key.startswith(("schema", "table"))
                else value
                for key, value in params.items()
            }

            stmt = text(query.format(**formatters))
        else:
            stmt = text(query)
        return stmt

    def run_query(self, query: str, params: dict = {}, return_as: str = None, commit: bool = False):
        """
        Securely runs a query and returns results (if applicable).
        If a pandas DataFrame is included in the query, please use DataFrameToPostgresSQL instead.

        Args:
            query (str): An unformatted SQL query.
                Dynamic schema and table names should be in brackets (SELECT * FROM {schema}.{table}).
                Any other dynamic values should be PostgreSQL formatable (WHERE name = :name).
            params (dict, optional): A dictionary of parameters to pass to the query.
                Dynamic schema and table keys must begin with "schema" and "table", respectively.
                All keys must match proper aliases in the query. Defaults to {}.
            return_as (str, optional): None (write), "pandas", "numpy", or "tuple". Defaults to None.
            commit (bool, optional): If True, commit the query to the database. Defaults to False.

        Example:
            run_query(
                "SELECT * FROM {schema1}.{table1} WHERE close > :close;",
                params={
                    "schema1": "prices",
                    "table1": "aapl",
                    "close": 210.00
                },
                return_as="pandas"
            )

        Returns:
            Based on return_as.
        """

        stmt = self._format_query(query, params)

        if return_as == "pandas":
            resp = pd.read_sql(stmt, self.conn, params=params)
        elif return_as == "numpy":
            resp = np.array(self.conn.execute(stmt, params or {}).fetchall())
        elif return_as == "tuple":
            resp = tuple(self.conn.execute(stmt, params or {}).fetchall())
        else:
            resp = self.conn.execute(stmt, params or {})

        if commit:
            self.conn.commit()
        return resp

    def has_table(self, table_name: str, schema: str) -> bool:
        """
        Checks if a table exists at the given schema.

        Args:
            table_name (str): The table name.
            schema (str): The schema name.

        Returns:
            True if the table exists. False if not.
        """
        return inspect(self.conn).has_table(quoted_name(table_name, True), schema)

    class DataFrameToPostgreSQL:
        """
        Temporarily registers a pandas DataFrame as a DuckDB view.
        Is a data bridge that allows for seamless querying of pandas DataFrames with PostgreSQL tables.
        """

        def __init__(self, psql_conn, psql_table: str, psql_schema: str, df: pd.DataFrame, df_view_name: str = "df", attached_db_name: str = "tc"):
            self.psql_conn = psql_conn
            self.psql_table = psql_table
            self.psql_schema = psql_schema
            self.df = df
            self.df_view_name = df_view_name
            self.attached_db_name = attached_db_name

        def __enter__(self):
            get_duckdb().register(self.df_view_name, self.df)
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            try:
                if exc_type is not None:
                    get_logger().error(
                        "Exception in class DataFrameToDuckDB context.", extra={
                            "error": traceback.format_exc()
                        }
                    )
                    return True
            except Exception as e:
                # Unable to un-register DataFrame
                raise e
            finally:
                get_duckdb().unregister(self.df_view_name)

        def _insert_psql_alias(self, query: str):
            """
            Inserts PostgreSQL alias ('tc' by default), in front of all PostgreSQL table references.
            """
            return query.replace(self.psql_schema, self.attached_db_name + "." + self.psql_schema)

        def create_table_from_df(self, primary_key: list = None, indexes: dict = None):
            """
            Creates a table from the given DataFrame through this process:
            1. Leverage DuckDB's tight integration with pandas to write the DataFrame to an in-memory DuckDB table.
            2. Use DuckDB's PostgreSQL integration tool to write in-memory (DuckDB) table to persistent storage (PostgreSQL).
            Please use use_df_as_table() for insert statements.

            Args:
                primary_key (list, optional): A list of columns to use for PK of the table. Defaults to None.
                indexes (dict, optional): A dictionary specifying indexes to create for the table.
                    Structure: {idx_name: [unique (bool), [col1, col2, col3]]}.
                    True means index is unique. False is not. Defaults to None.

            Returns:
                None
            """

            # Writes table to PostgreSQL from DuckDB
            try:
                get_duckdb().execute(f'''
                    CREATE OR REPLACE TABLE {self.attached_db_name}.{self.psql_schema}.{self.psql_table}
                    AS SELECT * FROM {self.df_view_name};
                    '''
                )
            except duckdb.CatalogException:
                return False

            # Creates primary keys
            if primary_key:
                pk_query = '''
                    ALTER TABLE {schema}.{table}
                        ADD PRIMARY KEY ({pk_columns})
                    '''
                self.psql_conn.run_query(
                    pk_query,
                    params={
                        "schema": self.psql_schema,
                        "table": self.psql_table,
                        "pk_columns": ", ".join(primary_key)
                    },
                    commit=True
                )

            # Creates indexes
            if indexes:
                for idx_name, [unique, idx_cols] in indexes.items():
                    if unique:
                        unique_slc = "UNIQUE"
                    else:
                        unique_slc = ""
                    idx_query = '''
                        CREATE {unique} INDEX {index_name} IF NOT EXISTS 
                        ON {schema}.{table}({columns});
                        '''
                    self.psql_conn.run_query(
                        idx_query,
                        params={
                            "unique": unique_slc,
                            "index_name": idx_name,
                            "schema": self.psql_schema,
                            "table": self.psql_table,
                            "columns": ", ".join(idx_cols)
                        }
                    )

            # Ensures that the new table is registered
            self.psql_conn.conn.commit()
            return True

        def insert_or_replace_df_into_table(self, replace_by: str):
            """
            DuckDB integration with PostgreSQL has a limitation of not detecting indexes/primary keys of PostgreSQL
            tables. This rigidly bypasses this limitation (mimics 'INSERT OR REPLACE INTO' clauses).
            Note that the corresponding PostgreSQL table must already exist.

            Args:
                replace_by (str): The column name which conflicts with the primary key column.

            Returns:
                None
            """
            try:
                # Deletes rows that would cause conflict from PostgreSQL
                get_duckdb().execute(f'''
                    DELETE FROM {self.attached_db_name}.{self.psql_schema}.{self.psql_table}
                        WHERE {replace_by} IN (SELECT {replace_by} FROM {self.df_view_name});
                    '''
                )
                # Re-inserts deleted rows with updated values to PostgreSQL
                get_duckdb().execute(f'''
                    INSERT INTO {self.attached_db_name}.{self.psql_schema}.{self.psql_table}
                        SELECT * FROM {self.df_view_name};
                    '''
                )
                self.psql_conn.conn.commit()
                return True
            except duckdb.CatalogException:
                return "no_table"
            except duckdb.BinderException:
                return "binder_error"
            except Exception:
                return traceback.format_exc()


        @timer(_id="table_name")
        def use_df_as_table(self, query: str, params: dict = {}, commit: bool = True):
            """
            Reads data from PostgreSQL into a DuckDB table so that pandas DataFrames can interact directly with PostgreSQL.
            This is possible because DuckDB can treat a DataFrame as an SQL table.
            In query, remembering to pass PostgreSQL table's schema.
            The database alias is applied automatically.

            Args:
                self.psql_schema (str): The PostgreSQL schema name.
                query (str): The unformatted SQL query. See run_query documentation for detailed information.
                params (dict, optional): Parameters to pass to the query. Defaults to {}.
                commit (bool, optional): Whether to commit the changes to PostgreSQL. Defaults to False.

            Examples:
                use_df_as_table(
                    query = '''
                        DELETE FROM tc.{schema}.{table} WHERE
                            tc.{schema}.{table}.{column} NOT IN (SELECT df.{column} FROM df);
                        ''',
                    params = {
                        "schema": "prices",
                        "table": "metadata",
                        "column": "permaticker",
                    }
                )

            Returns:
                None
            """

            query = self._insert_psql_alias(query)

            # Properly quotes params to sqlalchemy TextClause, then re-formatted back to python string
            stmt = str(self.psql_conn._format_query(query, params))

            # Executes statement in DuckDB and COMMIT in PostgreSQL
            get_duckdb().execute(stmt)
            if commit:
                self.psql_conn.conn.commit()

            return

    def add_missing_columns(self, table_name: str, columns: list):
        """
        Adds missing columns to a DB table.

        Args:
            table_name (str): Name of DB table.
            columns (list): List of column names.

        Returns:
            None
        """

        # Retrieve current existing columns
        query = '''
            SELECT 
                column_name 
            FROM 
                information_schema.columns 
            WHERE 
                table_name = :table_name;
            '''
        existing_cols = self.run_query(
            query,
            params={"table_name": table_name},
            return_as="tuple"
        )
        existing_cols = (col[0] for col in existing_cols)

        columns_to_add = (col for col in columns if col not in existing_cols)

        for column in columns_to_add:
            query = '''
                ALTER TABLE {table}
                    ADD COLUMN :column DOUBLE;
                '''
            self.run_query(
                query,
                params={"table": table_name, "column": column},
                commit=True
            )
        return
