import os
import re
import glob
import argparse

call_path = os.getcwd()

parser = argparse.ArgumentParser(prog='sanitise_filenames', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-g', '--glob',   type=str, default='', help='file pattern to glob for change')
parser.add_argument('-l', '--lower',  type=int, default=1, help='force to lower case?')
parser.add_argument('-u', '--underscorechars',  type=str, default='- ', help='chars to replace with underscore')
parser.add_argument('-d', '--deletechars',  type=str, default=',\\[]', help='chars to replace with nothing')

args = parser.parse_args()

ff = glob.glob(args.glob)

for old in ff:
    new = old
    if args.lower == 1:
        new = new.lower()
    for s in args.underscorechars:
        new = new.replace(s,'_')
    for s in args.deletechars:
        new = new.replace(s,'')
    new = new.replace('_+','_')
    # remove duplicate underscores:
    new = re.sub('_+', '_', new)
    com = f'mv "{old}" {new}'
    print(com)
    os.system(com)
