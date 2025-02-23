import os
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
from pandasql import sqldf

pd.set_option('mode.chained_assignment', None)
#
# ==============================================================================
#
script_path = Path(__file__).resolve()
script_dir, script_name = os.path.split(script_path)
#
# ==============================================================================
#

def df_cast(df, index_cols, feature_cols, value_col, join_str='_'):
    df = df.pivot_table(
    values=value_col, index=index_cols, columns=feature_cols,
    fill_value=0, aggfunc='mean')

    df.columns = df.columns.map(join_str.join)
    df = df.reset_index()
    return df

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
