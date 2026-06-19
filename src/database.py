# Create SQLite database and tables, then insert data
import os
from typing import Literal
import pandas as pd
import numpy as np
#import yaml
from sqlalchemy import create_engine

from src.config import *
import sqlalchemy as sa

def create_database(db_url, df_train_test, df_test_rul, datasets_info=DATASETS_META):
    engine = sa.create_engine(db_url)
   
    query_drop_if_exists_list = [
        "DROP VIEW IF EXISTS ml_train;",
        "DROP VIEW IF EXISTS ml_test;",
        "DROP VIEW IF EXISTS units_summary;",
        "DROP VIEW IF EXISTS ml_test_last_cycle;",
        "DROP TABLE IF EXISTS test_rul;",
        "DROP TABLE IF EXISTS engine_cycles;",
        "DROP TABLE IF EXISTS datasets_meta;",
    ]
    
    query_create_table_datasets_meta = """
        CREATE TABLE datasets_meta ( 
            dataset_id TEXT PRIMARY KEY,
            condition TEXT,
            fault_mode TEXT,
            train_file TEXT,
            test_file TEXT, 
            rul_file TEXT
        );
    """

    query_create_table_engine_cycles = """
        CREATE TABLE engine_cycles (    
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            split TEXT NOT NULL,
            unit_nr INTEGER NOT NULL,
            time_cycles INTEGER NOT NULL,

            setting1 REAL, setting2 REAL, setting3 REAL,

            sensor1 REAL, sensor2 REAL, sensor3 REAL,
            sensor4 REAL, sensor5 REAL, sensor6 REAL,
            sensor7 REAL, sensor8 REAL, sensor9 REAL,
            sensor10 REAL, sensor11 REAL, sensor12 REAL,
            sensor13 REAL, sensor14 REAL, sensor15 REAL,
            sensor16 REAL, sensor17 REAL, sensor18 REAL,
            sensor19 REAL, sensor20 REAL, sensor21 REAL,

            RUL INTEGER,

            UNIQUE(dataset_id, split, unit_nr, time_cycles),
            FOREIGN KEY (dataset_id) REFERENCES datasets_meta(dataset_id)
        );
    """

    query_create_table_test_rul = """
    CREATE TABLE test_rul (
        dataset_id TEXT NOT NULL,
        unit_nr INTEGER NOT NULL,
        true_RUL INTEGER NOT NULL,

        PRIMARY KEY (dataset_id, unit_nr),
        FOREIGN KEY (dataset_id) REFERENCES datasets_meta(dataset_id)
    );
    """

    query_create_view_train = """
    CREATE VIEW ml_train AS
    SELECT
        ec.dataset_id,
        dm.condition,
        dm.fault_mode,
        ec.unit_nr,
        ec.time_cycles,
        ec.setting1, ec.setting2, ec.setting3,
        ec.sensor1, ec.sensor2, ec.sensor3, ec.sensor4, ec.sensor5,
        ec.sensor6, ec.sensor7, ec.sensor8, ec.sensor9, ec.sensor10,
        ec.sensor11, ec.sensor12, ec.sensor13, ec.sensor14, ec.sensor15,
        ec.sensor16, ec.sensor17, ec.sensor18, ec.sensor19, ec.sensor20, ec.sensor21,
        ec.RUL
    FROM engine_cycles ec
    JOIN datasets_meta dm
    ON ec.dataset_id = dm.dataset_id
    WHERE ec.split = 'train';
    """
    query_create_view_test = """
    CREATE VIEW ml_test AS
    SELECT
        ec.dataset_id,
        dm.condition,
        dm.fault_mode,
        ec.unit_nr,
        ec.time_cycles,
        ec.setting1, ec.setting2, ec.setting3,
        ec.sensor1, ec.sensor2, ec.sensor3, ec.sensor4, ec.sensor5,
        ec.sensor6, ec.sensor7, ec.sensor8, ec.sensor9, ec.sensor10,
        ec.sensor11, ec.sensor12, ec.sensor13, ec.sensor14, ec.sensor15,
        ec.sensor16, ec.sensor17, ec.sensor18, ec.sensor19, ec.sensor20, ec.sensor21
    FROM engine_cycles ec
    JOIN datasets_meta dm
    ON ec.dataset_id = dm.dataset_id
    WHERE ec.split = 'test';
    """
    # The view ml_test_last_cycle contains only the last cycle for each unit in the test split, 
    # which is useful for evaluating RUL predictions against the true RUL values in the test_rul table.
    query_create_view_test_last_cycle = """
    CREATE VIEW ml_test_last_cycle AS
    SELECT
        ec.*,
        dm.condition,
        dm.fault_mode
    FROM engine_cycles AS ec
    JOIN datasets_meta dm
    ON ec.dataset_id = dm.dataset_id
    WHERE ec.split = 'test'
    AND ec.time_cycles = (
        SELECT MAX(sub.time_cycles)
        FROM engine_cycles AS sub
        WHERE sub.dataset_id = ec.dataset_id
            AND sub.unit_nr = ec.unit_nr
            AND sub.split = 'test'
    );
    """
    query_create_view_summary = """
    CREATE VIEW units_summary AS
    SELECT
        dataset_id,
        split,
        unit_nr,
        MIN(time_cycles) AS cycles_min,
        MAX(time_cycles) AS cycles_max,
        COUNT(*) AS cycles_count
    FROM engine_cycles
    GROUP BY dataset_id, split, unit_nr;
    """

    with engine.begin() as conn:
        for query in query_drop_if_exists_list:
            conn.execute(sa.text(query))
        
        conn.execute(sa.text(query_create_table_datasets_meta))
        conn.execute(sa.text(query_create_table_engine_cycles))
        conn.execute(sa.text(query_create_table_test_rul))

        pd.DataFrame(datasets_info).to_sql("datasets_meta", conn, if_exists="append", index=False)
        df_train_test.to_sql("engine_cycles", conn, if_exists="append", index=False)
        df_test_rul.to_sql("test_rul", conn, if_exists="append", index=False)        
        
        conn.execute(sa.text(query_create_view_train))
        conn.execute(sa.text(query_create_view_test))
        conn.execute(sa.text(query_create_view_summary))
        conn.execute(sa.text(query_create_view_test_last_cycle))
        
    engine.dispose()
    print('\n' + '-'*10 + "Database created successfully" + '-'*10)
    print(f"Database saved: {db_url}")

# Function to load data from the database
def load_data_from_database(db_url=DATABASE_URL):
    engine = sa.create_engine(db_url)
    with engine.connect() as conn:
        df_train = pd.read_sql_table('ml_train', con=conn)
        df_test = pd.read_sql_table('ml_test', con=conn)
        df_summary = pd.read_sql_table('units_summary', con=conn)
        df_test_rul = pd.read_sql_table('test_rul', con=conn)
        df_test_last_cycle = pd.read_sql_table('ml_test_last_cycle', con=conn)
    engine.dispose()
    for df in [df_train, df_test, df_test_last_cycle, df_test_rul, df_summary]:
        df.columns = df.columns.astype(str)
    return df_train, df_test, df_summary, df_test_rul, df_test_last_cycle

