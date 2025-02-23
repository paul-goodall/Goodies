import os
import sys
import math
import time
import duckdb
import hashlib
from pathlib import Path

#
# ==============================================================================
#
script_path = Path(__file__).resolve()
script_dir, script_name = os.path.split(script_path)

#
# ==============================================================================
#
def string2hash(ss):
    b_ss = bytes(ss, 'utf-8')
    return hashlib.sha1(b_ss).hexdigest()
#
# ==============================================================================
#


# ==============================================================================
#
def wget(url,filename,go=True):
    com = 'wget -q -O ' + filename + ' "' + url + '"'
    if go:
        os.system(com)
    return com
#
# ==============================================================================
#

def linux_fastcombine_csvs(my_csv_glob, first_file, outfile, go=False):

    com = f'''
    ulimit -n 2048;
    head -n 1 {first_file} > {outfile};
    tail -n +2 -q {my_csv_glob} >> {outfile};
    '''
    print(com)
    if go:
        os.system(com)

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
