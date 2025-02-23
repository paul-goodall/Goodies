import os
import sys
import math
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

#
# ==============================================================================
#
script_path = Path(__file__).resolve()
script_dir, script_name = os.path.split(script_path)
#
# ==============================================================================
#
day_names   = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
day_numbers = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
#
# ==============================================================================
#
def get_timetag():
    ct = str(datetime.datetime.now()).split('.')[0]
    ct = ct.replace('-','_')
    ct = ct.replace(' ','_')
    ct = ct.replace(':','_')
    return ct

#
# ==============================================================================
#

def date_info_df(my_dt):
    if type(my_dt) == str:
        my_dt = pd.DataFrame({'Dt':my_dt}, index=[0])
        my_dt = my_dt['Dt']

    my_dt = pd.to_datetime(my_dt, format="%Y-%m-%d")
    my_dt = my_dt.dt
    dct = {}
    dct['WeekDay'] = my_dt.day_name().tolist()
    dct['Year']    = my_dt.year.tolist()
    dct['Month']   = my_dt.month.tolist()
    dct['DayOfMonth'] = my_dt.day.tolist()
    dct['DOM'] = dct['DayOfMonth']
    dct['DayOfYear']  = my_dt.day_of_year.tolist()
    dct['DOY'] = dct['DayOfYear']
    dct['DaysInYear'] = [ pd.to_datetime(f'{yr}-12-31').day_of_year for yr in dct['Year'] ]
    dct['NumericDate'] = np.array(dct['Year']) + (np.array(dct['DayOfYear'])-0.5)/np.array(dct['DaysInYear'])
    dct['DayName_1stJan'] = [ pd.to_datetime(f'{yr}-01-01').day_name() for yr in dct['Year'] ]
    day1offset = [ day_names[x] for x in dct['DayName_1stJan'] ]
    dct['WeekOfYear'] = np.ceil((np.array(dct['DayOfYear']) + np.array(day1offset) - 0.5)/7).astype(int)
    dct['WOY'] = dct['WeekOfYear']
    df = pd.DataFrame(dct)
    return df

#
# ==============================================================================
#

#
# ==============================================================================
#
