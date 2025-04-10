import os
import sys
import pandas as pd
from glob import glob
import argparse

# Change this
my_dir = '/Path/To/project'
# ========

parser = argparse.ArgumentParser(prog='Some Script Title', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-s', '--stringarg', type=str,   default='foo', help='Example string arg')
parser.add_argument('-f', '--floatarg',  type=float, default=1.0,   help='Example float arg')
parser.add_argument('-i', '--intarg',    type=int,   default=121,   help='Example int arg')

args = parser.parse_args()

# ========
# DO STUFF

v1 = args.stringarg
v2 = args.floatarg
v3 = args.intarg
