import os
import sys
import math
from pathlib import Path
import subprocess
import time
import duckdb
import pyodbc, struct
import pandas as pd
from pandasql import sqldf
import sqlalchemy as sa
import geopandas as gpd
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

os_type = subprocess.check_output(['uname','-s']).decode().replace('\n','')

# IF MAC then use PYODBC
# IF LINUX then use PYMMSQL
if 'Darwin' in os_type:
    import pyodbc

if 'Linux' in os_type:
    import pymssql

# ==============================================================================

def get_database_settings(env_path='~/.env'):
    opts = {}
    if 'Darwin' in os_type:
        env = dotenv_values(env_path)
        opts['host'] = env['DATABASE_HOST']
        opts['user'] = env['DATABASE_USERNAME']
        opts['pswd'] = env['DATABASE_PASSWORD']
        opts['port'] = env['DATABASE_PORT']
        opts['dbnm'] = env['DATABASE_NAME']

    if 'Linux' in os_type:
        opts['host'] = os.environ['DATABASE_HOST']
        opts['user'] = os.environ['DATABASE_USERNAME']
        opts['pswd'] = os.environ['DATABASE_PASSWORD']
        opts['port'] = os.environ['DATABASE_PORT']
        opts['dbnm'] = os.environ['DATABASE_NAME']

    return opts

# ==============================================================================

def database_connection_setup_mssql(enum=1,env_path='~/.env'):

    # IF MAC   then use enum=0 ==> PYODBC
    # IF LINUX then use enum=1 ==> PYMMSQL
    engine_choices = ['mssql+pyodbc','mssql+pymssql']
    engine_type = engine_choices[enum]

    # Refer to:
    # https://docs.sqlalchemy.org/en/13/core/engines.html#postgresql

    db = get_database_settings(env_path)

    if engine_type == 'mssql+pyodbc':
        my_Driver = '{ODBC Driver 18 for SQL Server}'
        conx = f'''
        Driver={my_Driver};
        Server=tcp:{db['host']},{db['port']};
        Database={db['dbnm']};
        UID={db['user']};
        PWD={db['pswd']};
        Encrypt=yes;
        TrustServerCertificate=no;
        Connection Timeout=30
        '''
        urlx = URL.create(engine_type, query={"odbc_connect": conx})

    if engine_type == 'mssql+pymssql':
        urlx = f'''{engine_type}://{db['user']}:{db['pswd']}@{db['host']}:{db['port']}/{db['dbnm']}'''
        # not sure about this one:
        conx = urlx

    engine = create_engine(urlx)

    options = {}
    options['urlx'] = urlx
    options['conx'] = conx
    options['engine'] = engine

    return options

# ==============================================================================

def database_connection_setup_postgres(enum=1,env_path='~/.env'):

    engine_choices = ['postgresql','postgresql+psycopg2','postgresql+pg8000']
    engine_type = engine_choices[enum]

    # Refer to:
    # https://docs.sqlalchemy.org/en/13/core/engines.html#postgresql

    db = get_database_settings(env_path)

    urlx = f'''{engine_type}://{db['user']}:{db['pswd']}@{db['host']}:{db['port']}/{db['dbnm']}'''
    conx = urlx

    engine = create_engine(urlx,plugins=["geoalchemy2"],)

    options = {}
    options['urlx'] = urlx
    options['conx'] = conx
    options['engine'] = engine

    return options

# ==============================================================================

def database_connection_setup_mysql(enum=1,env_path='~/.env'):

    engine_choices = ['mysql','mysql+mysqldb','mysql+pymysql']
    engine_type = engine_choices[enum]

    # Refer to:
    # https://docs.sqlalchemy.org/en/13/core/engines.html#postgresql

    db = get_database_settings(env_path)

    urlx = f'''{engine_type}://{db['user']}:{db['pswd']}@{db['host']}:{db['port']}/{db['dbnm']}'''
    conx = urlx

    engine = create_engine(urlx)

    options = {}
    options['urlx'] = urlx
    options['conx'] = conx
    options['engine'] = engine

    return options


# ==============================================================================

def database_connection_setup_sqlite(filepath=None):

    # filepath should be None or the absolute path to a db-file:
    # /absolute/path/to/foo.db

    # Refer to:
    # https://docs.sqlalchemy.org/en/13/core/engines.html#postgresql

    if filepath is None:
        # create an in-memory database:
        engine = create_engine('sqlite://')
    else:
        engine = create_engine(f'sqlite:///{filepath}')

    return engine



# ==============================================================================

def drop_table(csr,table_name):
    qry = f'''
    drop table if exists {table_name};
    '''
    run_qry_noreturnval(csr,qry)

# ==============================================================================

def rename_table(csr,old_name,new_name):
    # This might be specific to MsSQL
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

# ==============================================================================

def sql2pd(qry,engine,verbose=True):
    if verbose:
        t1 = time.time()
        print(f'========== Query ===========')
        print(qry)
    with engine.begin() as conn:
        df1 = pd.read_sql_query(sa.text(qry), conn)
    if verbose:
        t2 = time.time()
        dt = t2 - t1
        print(f'Query took: {dt:0.2f} seconds')
        print(f'============================\n')
    return df1


# ==============================================================================

def sql2gdf(qry,engine,verbose=True):
  if verbose:
      t1 = time.time()
      print(f'========== Query ===========')
      print(qry)
  with engine.begin() as conn:
      df1 = gpd.read_postgis(qry,engine)
  if verbose:
      t2 = time.time()
      dt = t2 - t1
      print(f'Query took: {dt:0.2f} seconds')
      print(f'============================\n')
  return df1



# ==============================================================================

def list_tables(engine_R):
    # This might be specific to MsSQL
    qry = '''
    SELECT *
    FROM INFORMATION_SCHEMA.TABLES
    WHERE table_type = 'BASE TABLE' and table_schema = 'dbo'
    '''
    df_tables = sql2pd(qry,engine_R)
    return df_tables

# ==============================================================================

def get_table_sizes(engine_R):
    # This might be specific to MsSQL
    qry_pgres = '''
    SELECT
        t.name AS TableName,
        s.name AS SchemaName,
        p.rows,
        CAST(ROUND(((SUM(a.total_pages) * 8) / (1024.00 * 1024.00)), 2) AS NUMERIC(36, 2)) AS Size_GB
    FROM
        sys.tables t
    INNER JOIN
        sys.indexes i ON t.object_id = i.object_id
    INNER JOIN
        sys.partitions p ON i.object_id = p.object_id AND i.index_id = p.index_id
    INNER JOIN
        sys.allocation_units a ON p.partition_id = a.container_id
    LEFT OUTER JOIN
        sys.schemas s ON t.schema_id = s.schema_id
    WHERE
        t.name NOT LIKE 'dt%'
        AND t.is_ms_shipped = 0
        AND i.object_id > 255
    GROUP BY
        t.name, s.name, p.rows
    ORDER BY
        Size_GB DESC, t.name
    '''
    df_tables = sql2pd(qry_pgres,engine_R)

    nr = df_tables.rows.sum()
    ts = df_tables.Size_GB.sum()
    new_row = {'TableName':'(TOTALS)', 'SchemaName':'', 'rows':nr, 'Size_GB':ts}
    df_tables = pd.concat([df_tables, pd.DataFrame([new_row])], ignore_index=True)
    return df_tables


# ==============================================================================

#
# ==============================================================================
#

#
# ==============================================================================
#
