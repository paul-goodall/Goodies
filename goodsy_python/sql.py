import os
import sys
import math
from pathlib import Path

import pyodbc, struct
import pandas as pd
from pandasql import sqldf
import sqlalchemy as sa
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from dotenv import dotenv_values

#
# ==============================================================================
#
script_path = Path(__file__).resolve()
script_dir, script_name = os.path.split(script_path)
#
# ==============================================================================
#


def database_connection_setup():
    config = dotenv_values(wirely_secrets_path)

    # Ammend the database name as required:
    config['DATABASE_NAME'] = 'Wirely-Reporting'
    my_Driver = '{ODBC Driver 18 for SQL Server}'
    connection_string_reporting = f'''
    Driver={my_Driver};
    Server=tcp:{config['DATABASE_HOST']},{config['DATABASE_PORT']};
    Database={config['DATABASE_NAME']};
    UID={config['DATABASE_USERNAME']};
    PWD={config['DATABASE_PASSWORD']};
    Encrypt=yes;
    TrustServerCertificate=no;
    Connection Timeout=30
    '''

    config['DATABASE_NAME'] = 'Wirely-Production'
    my_Driver = '{ODBC Driver 18 for SQL Server}'
    connection_string_production = f'''
    Driver={my_Driver};
    Server=tcp:{config['DATABASE_HOST']},{config['DATABASE_PORT']};
    Database={config['DATABASE_NAME']};
    UID={config['DATABASE_USERNAME']};
    PWD={config['DATABASE_PASSWORD']};
    Encrypt=yes;
    TrustServerCertificate=no;
    Connection Timeout=30
    '''

    connection_url_reporting  = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string_reporting})
    connection_url_production = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string_production})

    engine_R = create_engine(connection_url_reporting)
    engine_P = create_engine(connection_url_production)

    options = {}
    options['connection_string_reporting']  = connection_string_reporting
    options['connection_string_production'] = connection_string_production
    options['engine_R'] = engine_R
    options['engine_P'] = engine_P

    return options

# ==============================================================================

def drop_table(csr,table_name):
    qry = f'''
    drop table if exists {table_name};
    '''
    run_qry_noreturnval(csr,qry)

# ==============================================================================

def rename_table(csr,old_name,new_name):
    qry = f'''
    EXEC sp_rename '{old_name}', '{new_name}'
    '''
    run_qry_noreturnval(csr,qry)

# ==============================================================================

def duck_qry(q):
    '''modified to work with duckdb >= 1.1. permanent connection matters if one creates tables in duckdb as well'''
    if not hasattr(duck_qry, 'connection'):
        duck_qry.connection = duckdb.connect()
        duck_qry.connection.execute("set python_scan_all_frames=true")

    return duck_qry.connection.sql(q).to_df()

# ==============================================================================

def run_qry_noreturnval_onetry(cx_str,qry,verbose=True):
    if verbose:
        t1 = time.time()
        print(f'========== Query ===========')
        print(qry)

    cnxn = pyodbc.connect(cx_str)
    crsr = cnxn.cursor()
    rows = crsr.execute(qry)
    cnxn.commit()
    crsr.close()
    cnxn.close()

    if verbose:
        t2 = time.time()
        dt = t2 - t1
        print(f'Query took: {dt:0.2f} seconds')
        print(f'============================\n')

# ==============================================================================

def run_qry_noreturnval(cx_str,qry,verbose=True):
    try:
        run_qry_noreturnval_onetry(cx_str,qry,verbose)
    except pyodbc.OperationalError:
        run_qry_noreturnval_onetry(cx_str,qry,verbose)
    except:
        run_qry_noreturnval_onetry(cx_str,qry,verbose)
#
# ==============================================================================
#


#
# ==============================================================================
#

#
# ==============================================================================
#

#
# ==============================================================================
#
