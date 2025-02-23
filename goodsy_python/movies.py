import os
import sys
import cv2
import math
import glob
import json
import time
import copy
import random
import imutils
import statistics
import numpy as np
import pandas as pd

from pathlib import Path

from scipy import signal
import scipy.optimize as opt
from scipy.optimize import curve_fit

from skimage import data
from skimage import transform
from astropy.io import fits
from pandasql import sqldf
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.ticker as ticker

import pickle
import bz2
import _pickle as cPickle

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdalconst

from shapely.geometry import Point, Polygon
import geopandas as gpd

import io
from PIL import Image
import hashlib
import base64

pd.set_option('mode.chained_assignment', None)
#
# ==============================================================================
#
script_path = Path(__file__).resolve()
script_dir, script_name = os.path.split(script_path)
#
# ==============================================================================
#

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


# ===============================================


def create_smart_table_html(df, html_file, my_page_title='My Smart Table'):

    my_table_header = '<tr class="header">\n'
    for cn in list(df.columns):
        my_table_header += f'<th>{cn}</th>\n'
    my_table_header += '</tr>'

    my_table_rows = ''
    for ind, row in df.iterrows():
        my_table_rows += '<tr>\n'
        for cn in list(df.columns):
            my_table_rows += f'<td>{row[cn]}</td>\n'
        my_table_rows += '</tr>\n'

    my_table_contents = my_table_header + '\n' + my_table_rows

    smart_table_html = '''
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    * {
      box-sizing: border-box;
    }

    #myInput {
      background-image: url('/css/searchicon.png');
      background-position: 10px 10px;
      background-repeat: no-repeat;
      width: 100%;
      font-size: 16px;
      padding: 12px 20px 12px 40px;
      border: 1px solid #ddd;
      margin-bottom: 12px;
    }

    #myTable {
      border-collapse: collapse;
      width: 100%;
      border: 1px solid #ddd;
      font-size: 18px;
    }

    #myTable th, #myTable td {
      text-align: left;
      padding: 12px;
    }

    #myTable tr {
      border-bottom: 1px solid #ddd;
    }

    #myTable tr.header, #myTable tr:hover {
      background-color: #f1f1f1;
    }
    </style>
    </head>
    <body>

    <h2>my_page_title</h2>

    <input type="text" id="myInput" onkeyup="myFunction()" placeholder="Search for names.." title="Type in a name">

    <table id="myTable">
      my_table_contents
    </table>

    <script>
    function myFunction() {
      var input, filter, table, tr, td, i, txtValue;
      input = document.getElementById("myInput");
      filter = input.value.toUpperCase();
      table = document.getElementById("myTable");
      tr = table.getElementsByTagName("tr");
      for (i = 0; i < tr.length; i++) {
        td = tr[i].getElementsByTagName("td")[0];
        if (td) {
          txtValue = td.textContent || td.innerText;
          if (txtValue.toUpperCase().indexOf(filter) > -1) {
            tr[i].style.display = "";
          } else {
            tr[i].style.display = "none";
          }
        }
      }
    }
    </script>

    </body>
    </html>
    '''

    smart_table_html = smart_table_html.replace('my_page_title', my_page_title)
    smart_table_html = smart_table_html.replace('my_table_contents', my_table_contents)

    wTXT(smart_table_html, html_file)

# ==========================================


# ============
# print some useful information about a numpy array:
def np_info(nda):
    print("type\t: ", nda.dtype,"\nshape\t: ", nda.shape, "\nmin\t: ", nda.min(), "\nmax\t: ", nda.max(),"\n")
