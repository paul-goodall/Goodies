import os
import sys
import cv2
import math
import glob
import json
import time
import copy
import random
import pickle
import imutils
import statistics
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from skimage import data
from skimage import transform
from astropy.io import fits
from datetime import datetime

pd.set_option('mode.chained_assignment', None)

import bz2
import _pickle as cPickle

# ========================================
# save a pickle:
def save_pickle(data, filename, compress=True):
    if compress:
        with bz2.BZ2File(filename + '.pbz2', 'w') as f:
            cPickle.dump(data, f)
    else:
        dbfile = open(filename, 'ab')
        pickle.dump(data, dbfile)
        dbfile.close()

# ========================================
# load a pickle:
def load_pickle(filename):
    if filename[-4:] == 'pbz2':
        data = bz2.BZ2File(filename, 'rb')
        data = cPickle.load(data)
    else:
        dbfile = open(outfile, 'rb')
        data = pickle.load(dbfile)
        dbfile.close()
    return data


# ========================================
def frames2movie(output_video_path=None, frames_pattern=None, fps=10):
    if output_video_path is None:
        output_video_path = '~/dummy.mov'
    if frames_pattern is None:
        print("frames_pattern must be specified.")
    else:
        com = ''
        com = com + 'ffmpeg -hide_banner -loglevel error -y -framerate my_fps '
        com = com + '-i frames_pattern '
        com = com + '-c:v libx264 -pix_fmt yuv420p -vf format=yuv420p -c:a copy -r my_fps '
        com = com + 'output_video_path '
        com = com.replace('my_fps', str(fps))
        com = com.replace('frames_pattern', frames_pattern)
        com = com.replace('output_video_path', output_video_path)
        print(com)
        os.system(com)
# ========================================
def linux_fastcombine_csvs(my_csv_glob, first_file, outfile):
    if not os.path.exists(outfile):
        com = 'ulimit -n 2048; head -n 1 first_file > outfile; tail -n +2 -q my_csv_glob >> outfile;'
        com = com.replace('outfile', outfile)
        com = com.replace('my_csv_glob', my_csv_glob)
        com = com.replace('first_file', first_file)
        print(com)
        os.system(com)
def df_to_data_dict(df):
    dict = {}
    for col in df.columns:
        dict[col] = []
        dict[col] = list(df[col])
    return (dict)
# ========================
def data_dict_to_df(mySpec, string_features=[]):
    df = pd.DataFrame()
    for key in mySpec.keys():
        if key in string_features:
            df[key] = mySpec[key]
        else:
            df[key] = np.array(mySpec[key]).flatten()
    return (df)

# ========================================

def superfast_transform(image, x, y, a):
    height, width = image.shape[:2]
    xc = width/2
    yc = height/2
    c1 = math.cos(a * np.pi/180.0)
    s1 = math.sin(a * np.pi/180.0)
    m11 = c1
    m12 = -s1
    m13 = (1-c1)*xc + s1*yc + x
    m21 = s1
    m22 = c1
    m23 = -s1*xc + (1-c1)*yc + y
    rotate_matrix2d = np.array([[m11,m12,m13],[m21,m22,m23]])
    image = cv2.warpAffine(src=image, M=rotate_matrix2d, dsize=(width, height))
    return (image)

# =============================================
def superfast_reversetransform(image, x, y, a):
    height, width = image.shape[:2]
    xc = width/2 + x
    yc = height/2 + y
    c1 = math.cos(-a * np.pi/180.0)
    s1 = math.sin(-a * np.pi/180.0)
    m11 = c1
    m12 = -s1
    m13 = (1-c1)*xc + s1*yc -x
    m21 = s1
    m22 = c1
    m23 = -s1*xc + (1-c1)*yc -y
    rotate_matrix2d = np.array([[m11,m12,m13],[m21,m22,m23]])
    image = cv2.warpAffine(src=image, M=rotate_matrix2d, dsize=(width, height))
    return (image)
# ========================================
# Produce a 2D Gaussian:
def Gaussian2D(xy, xo, yo, A, sig, offset):
    x, y = xy
    pi = 3.141592654
    xo = float(xo)
    yo = float(yo)
    dr2 = (x-xo)**2 + (y-yo)**2
    g = offset + A * np.exp(-dr2/(2 * sig**2))
    return g.ravel()
# ========================================
def get_movie_frame(movie_file, frame_num):
    vid_capture = cv2.VideoCapture(movie_file)
    if (vid_capture.isOpened() == False):
      print("Error opening the video file")
    # Read fps and frame count
    else:
        frame_c = 0
        while(vid_capture.isOpened()):
            ret, img = vid_capture.read()
            if frame_c == frame_num:
                break
            vid_capture.release()
            frame_c += 1
    return (img)

# ============
def get_movie_metadata(movie_file, input_dictionary=None):
    if input_dictionary is None:
        input_dictionary = {}
    vid_capture = cv2.VideoCapture(movie_file)
    if (vid_capture.isOpened() == False):
      print("Error opening the video file")
    # Read fps and frame count
    else:
        input_dictionary['movie']['nx'] = int(vid_capture.get(3))
        input_dictionary['movie']['ny'] = int(vid_capture.get(4))
        input_dictionary['movie']['hz'] = vid_capture.get(5)
        input_dictionary['movie']['nz'] = int(vid_capture.get(7))
    vid_capture.release()
    return (input_dictionary)

# ============
def xvals2d(nx=1, ny=1, image=None):
    if not image is None:
        (ny,nx) = image.shape[:2]
    return np.arange(nx).reshape(1,nx) * np.ones(nx*ny).reshape(ny,nx)

# ============
def yvals2d(nx=1, ny=1, image=None):
    if not image is None:
        (ny,nx) = image.shape[:2]
    return np.arange(ny).reshape(ny,1) * np.ones(nx*ny).reshape(ny,nx)

# ============
def rvals2d(nx=1, ny=1, xc=None, yc=None, image=None):
    if not image is None:
        (ny,nx) = image.shape[:2]
    if xc is None:
        xc=nx/2
    if yc is None:
        yc=ny/2
    xv = xvals2d(nx=nx,ny=ny) - xc
    yv = yvals2d(nx=nx,ny=ny) - yc
    rv = (xv*xv + yv*yv)**0.5
    return rv

# ============
# print some useful information about a numpy array:
def np_info(nda):
    print("type\t: ", nda.dtype,"\nshape\t: ", nda.shape, "\nmin\t: ", nda.min(), "\nmax\t: ", nda.max(),"\n")
