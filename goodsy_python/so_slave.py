import os
import sys
import glob
import plotly
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--procnum', type=int, 
                    help='the processor number inside the cluster. (Default: 1)', 
                    default=1)

parser.add_argument('-n', '--numprocs', type=int, 
                    help='the number of workers in the cluster. (Default: 1)', 
                    default=1)

args = parser.parse_args()

nprocs  = args.numprocs
procnum = args.procnum

# -------------------------
