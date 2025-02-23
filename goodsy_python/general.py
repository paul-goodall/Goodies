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

def create_wget_string(url,filename):
    com = 'wget -O ' + filename + ' "' + url + '"'
    return com
#
# ==============================================================================
#
def download_image(url,filename):
    com = 'wget -O ' + filename + ' "' + url + '"'
    os.system(com)
#
# ==============================================================================
#

def linux_fastcombine_csvs(my_csv_glob, first_file, outfile):
    if not os.path.exists(outfile):
        com = 'ulimit -n 2048; head -n 1 first_file > outfile; tail -n +2 -q my_csv_glob >> outfile;'
        com = com.replace('outfile', outfile)
        com = com.replace('my_csv_glob', my_csv_glob)
        com = com.replace('first_file', first_file)
        print(com)
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
