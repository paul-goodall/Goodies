import os
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
from pandasql import sqldf

import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

blackred   = truncate_colormap(plt.get_cmap('Reds_r'), 0.0, 0.5)
blackgreen = truncate_colormap(plt.get_cmap('Greens_r'), 0.0, 0.5)
blackblue  = truncate_colormap(plt.get_cmap('Blues_r'), 0.0, 0.5)
#
# ==============================================================================
#

#
# ==============================================================================
#

#
# ==============================================================================
#
