#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 20:18:11 2025

@author: cjymain
"""

from technically.utils.handlers.db_old import DuckDB
from technically.utils.handlers.db import PostgreSQL
from technically.utils.optimizations import reformat_names
from technically.utils.log import log_setup

import sys
import re
import os

USER = os.getlogin()

log_setup(log_name="duckdb_to_psql")

for db_type in ["prices", "fundamentals"]:
    with PostgreSQL() as conn:
        with DuckDB(f"/home/{USER}/.technically/sql/{db_type}.duck") as db:
            db.execute("INSTALL postgres;")
            db.execute("LOAD postgres;")

            tables = db.sql("SELECT table_name FROM duckdb_tables;").fetchall()
            #tables = [("metadata",)]

            # Attaches PostgreSQL DB partition to DuckDB connection
            db.execute(f'''
                ATTACH 'dbname=technically user={USER} host=localhost password=Shadow123' 
                AS tc (TYPE postgres, SCHEMA '{db_type}');
                '''
            )

            for (table,) in tables:
                table_format = db.sql(f'''DESCRIBE "{table}";''').df()

                # Formats columns & their data types so that they can cleanly be passed to create table statement
                # Also passes table primary key if it contains one
                columns = [
                    f"{name} {dtype} {pkey}MARY KEY"
                    if pkey == "PRI"
                    else f"{name} {dtype} DEFAULT {default}"
                    if default != None
                    else f"{name} {dtype}"
                    for name, dtype, pkey, default
                    in zip(
                        table_format["column_name"].values,
                        table_format["column_type"].values,
                        table_format["key"].values,
                        table_format["default"].values
                    )
                ]

                # Re-formats table names to follow PostgreSQL naming conventions
                new_table = reformat_names(table)

                if re.match(r'[^a-z0-9_]', new_table):
                    continue

                # Drops new table if it exists in PostgreSQL
                drop_query = '''DROP TABLE IF EXISTS {schema}.{table};'''
                conn.run_query(
                    drop_query,
                    params={"schema": db_type, "table": new_table},
                    commit=True
                )

                # Creates or re-creates table in PostgreSQL
                table_conversion_query = f'''
                    CREATE TABLE tc.{new_table} (
                        {', '.join(columns)}
                    );
                '''
                db.execute(table_conversion_query)

                # Writes table data from DuckDB into PostgreSQL
                db.execute(f"INSERT INTO tc.{new_table} SELECT * FROM {table};")

                # Re-formats column names to follow PostgreSQL naming conventions
                for column in table_format["column_name"].values:
                    new_column = reformat_names(column)
                    if new_column != column:
                        db.execute(f'''ALTER TABLE tc.{new_table} RENAME column "{column}" TO {new_column};''')

                # Table changes exclusive to prices.metadata
                if new_table == "metadata":
                    # Lowercases permaticker entry prefix (i.e. "US", to "us")
                    update_query = '''
                        UPDATE
                            prices.metadata
                        SET
                            permaticker = LOWER(SUBSTR(permaticker, 1, 2)) || SUBSTR(permaticker, 3);
                        '''
                    conn.run_query(
                        update_query,
                        commit=True
                    )

                    update_query = '''
                        UPDATE
                            prices.metadata
                        SET
                            designation = 'dead_ticker'
                        WHERE
                            designation = 'deadTicker';
                        '''
                    conn.run_query(
                        update_query,
                        commit=True
                    )

                    update_query = '''
                        UPDATE
                            prices.metadata
                        SET
                            designation = 'recent_dates_inactive'
                        WHERE
                            designation = 'recentDatesInactive';
                        '''
                    conn.run_query(
                        update_query,
                        commit=True
                    )

                    # Properly sets/re-sets column defaults
                    for column in ["cap_category", "sector", "industry", "sic_sector", "sic_industry"]:
                        change_default_query = '''
                            ALTER TABLE prices.metadata
                                ALTER COLUMN {column}
                                SET DEFAULT 'unknown';
                            '''
                        conn.run_query(
                            change_default_query,
                            params={"column": column},
                            commit=True
                        )

                    # Sets columns to have "NOT NULL" attribute
                    for column in ["exchange", "asset_type"]:
                        set_not_null_query = '''
                            ALTER TABLE prices.metadata
                                ALTER COLUMN {column} 
                                SET NOT NULL;
                            '''



