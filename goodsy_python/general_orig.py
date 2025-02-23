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

gdal.UseExceptions()
pd.set_option('mode.chained_assignment', None)

# ==============================================================================
os.system('export CV_IO_MAX_IMAGE_PIXELS=1099511627776')
os.system('export OPENCV_IO_MAX_IMAGE_PIXELS=1099511627776')
os.environ["CV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
# ==============================================================================
#

script_path = Path(__file__).resolve()
script_dir, script_name = os.path.split(script_path)

#
# ==============================================================================
#
class ds_constants():
    r_earth = 6371000
    deg2rad = 2 * math.pi / 360
    lat_1deg = deg2rad * r_earth

    def lon_1deg(lat):
        return ( self.lat_1deg * math.cos(lat * self.deg2rad) )

cn = ds_constants
#
# ==============================================================================
#
def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
#
# ==============================================================================
#
def load_json(fname):
    f = open (fname, "r")
    # Reading from file
    data = json.loads(f.read())
    # Closing file
    f.close()
    return data
#
# ==============================================================================
#
def glob_files(fpat):
    ff = glob.glob(fpat)
    ff.sort()
    return ff
#
# ==============================================================================
#
def save_textfile(data, outfile):
    with open(outfile, 'a') as f_txt:
        print(data, file=f_txt)
#
# ==============================================================================
#
def save_dict2json(dicty, f_js, my_indent=4):
    # convert dictionary to JSON string
    json_data = json.dumps(dicty, indent=my_indent)
    # write the JSON string to a file
    with open(f_js, 'w') as f:
        f.write(json_data)
#
# ==============================================================================
#
def determine_image_depth(im):
    max_vals = np.array([1,255,65535])
    img_max = im.max()
    delta = abs(max_vals - img_max)
    im_threshold = max_vals[np.where(delta == min(delta))][0]
    im_threshold = im_threshold.astype(np.float32) * 1.0
    return (im_threshold)
#
# ==============================================================================
#
def load_img(filename, switch_rgb=True, normalise=True,flip_y = True):
    im_suffix = os.path.splitext(filename)[-1]
    x = cv2.imread(filename, -1)
    if flip_y:
        x = np.flipud(x)
    x = x[:,:,0:3]
    if switch_rgb:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    if normalise:
        col_max = determine_image_depth(x)
        x = x.astype(np.float32)/col_max
    return x
#
# ==============================================================================
#
def save_img(x, filename, switch_rgb=True, depth=16,flip_y = True):
    col_max = determine_image_depth(x)
    #print(col_max)
    im_suffix = os.path.splitext(filename)[-1]

    if im_suffix == '.png':
        if depth not in [8,16]:
            depth = 16
    if im_suffix == '.tif':
        if depth not in [8,16,32]:
            depth = 16
    if im_suffix in ['.jpg','.jpeg']:
        depth = 8

    if flip_y:
        x = np.flipud(x)

    x = x.astype(np.float32)/col_max
    if switch_rgb:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    if depth == 8:
        x = (x*255).astype(np.uint8)
    if depth == 16:
        x = (x*65535).astype(np.uint16)
    if depth == 32:
        x = (x*1.0).astype(np.float32)
    cv2.imwrite(filename, x)
#
# ==============================================================================
#
# save a pickle:
def save_pickle(data, filename, compress=True):
    if compress:
        with bz2.BZ2File(filename + '.pbz2', 'w') as f:
            cPickle.dump(data, f)
    else:
        if os.path.exists(filename):
            com = f'rm {filename}'
            os.system(com)
        dbfile = open(filename, 'ab')
        pickle.dump(data, dbfile)
        dbfile.close()
#
# ==============================================================================
#
# load a pickle:
def load_pickle(filename):
    if filename[-4:] == 'pbz2':
        #data = bz2.BZ2File(filename, 'rb')
        #data = cPickle.load(data)
        data = pd.read_pickle(filename,'bz2')
    else:
        #dbfile = open(filename, 'rb')
        #data = pickle.load(dbfile)
        #dbfile.close()
        data = pd.read_pickle(filename)
    return data
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

def read_textfile(outfile):
    with open(outfile, 'r') as f_txt:
        content = f_txt.read()
        f_txt.close()
    return content
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

    wtext(smart_table_html, html_file, mode='overwrite')

# ==========================================

#
# ==============================================================================
#
def wfits(data, filename, hdr=None, overwrite=True, output_verify='exception'):
    if hdr is None:
        fits.writeto(filename, data, overwrite=overwrite, output_verify=output_verify)
    elif type(hdr) is dict:
        hdr2 = fits.Header()
        for key in hdr:
            hdr2[key] = hdr[key]
        fits.writeto(filename, data, hdr2, overwrite=overwrite, output_verify=output_verify)
    else:
        # Assume here it's a FITS header object:
        print('Got astropy.io hdr')
        fits.writeto(filename, data, hdr, overwrite=overwrite, output_verify=output_verify)
#
# ==============================================================================
#
def rfits(filename):
    data, hdr = fits.getdata(filename, 0, header=True)
    return(data, hdr)
#
# ==============================================================================
#
def rtext(fn):
    f = open(fn, "r")
    txt = f.read()
    f.close()
    return txt

def wtext(txt, fn, mode='overwrite'):
    open_mode = 'w'
    if mode == 'append':
        open_mode = 'a'
    f = open(fn, open_mode)
    f.write(txt)
    f.close()


# write a pickle:
def wpkl(data, filename, compress=False):

    if filename == 'to_string':
        return pickle.dumps(data, 0).decode()

    s3 = False
    if 's3://' in filename:
        s3 = True

    if s3:
        fs = s3fs.S3FileSystem(anon=False)
        pickle.dump(data, fs.open(filename, 'wb'))
    else:
        if compress:
            with bz2.BZ2File(filename + '.pbz2', 'w') as f:
                cPickle.dump(data, f)
        else:
            if os.path.exists(filename):
                os.remove(filename)
            dbfile = open(filename, 'ab')
            pickle.dump(data, dbfile)
            dbfile.close()


# read a pickle:
def rpkl(filename,pickle_string=None):

    if filename == 'from_string':
        return pickle.loads(pickle_string.encode())

    s3 = False
    if 's3://' in filename:
        s3 = True

    if s3:
        fs = s3fs.S3FileSystem(anon=False)
        data = pickle.load(fs.open(filename, 'rb'))
    else:
        if filename[-4:] == 'pbz2':
            data = bz2.BZ2File(filename, 'rb')
            data = cPickle.load(data)
        else:
            dbfile = open(filename, 'rb')
            data = pickle.load(dbfile)
            dbfile.close()
    return data

#
# ==============================================================================
#
def xvals(nx,ny,nz=1,norm=False):
    x = np.arange(nx).reshape([1,1,nx])
    y = np.arange(ny).reshape([1,ny,1])
    z = np.arange(nz).reshape([nz,1,1])
    m = x * (1+0*y) * (1+0*z)
    if nz == 1:
        m = m.reshape([ny,nx])
    if norm:
        m = m/np.max(m)
    return m
#
# ==============================================================================
#
def yvals(nx,ny,nz=1,norm=False):
    x = np.arange(nx).reshape([1,1,nx])
    y = np.arange(ny).reshape([1,ny,1])
    z = np.arange(nz).reshape([nz,1,1])
    m = (1+0*x) * y * (1+0*z)
    if nz == 1:
        m = m.reshape([ny,nx])
    if norm:
        m = m/np.max(m)
    return m
#
# ==============================================================================
#
def zvals(nx,ny,nz=1,norm=False):
    x = np.arange(nx).reshape([1,1,nx])
    y = np.arange(ny).reshape([1,ny,1])
    z = np.arange(nz).reshape([nz,1,1])
    m = (1+0*x) * (1+0*y) * z
    if nz == 1:
        m = m.reshape([ny,nx])
    if norm:
        m = m/np.max(m)
    return m
#
# ==============================================================================
#
def svals(nx,ny,nz=1,norm=False):
    m = xvals(nx,ny,nz) + nx*yvals(1,ny,nz) + nx*ny*zvals(1,1,nz)
    if nz == 1:
        m = m.reshape([ny,nx])
    if norm:
        m = m/np.max(m)
    return m
#
# ==============================================================================
#
def rvals(nx,ny,nz=1,x0=0,y0=0,z0=0,norm=False):
    x = xvals(nx,ny,nz) - x0
    y = yvals(nx,ny,nz) - y0
    z = zvals(nx,ny,nz) - z0
    m = (x**2 + y**2 + z**2)**0.5
    if nz == 1:
        m = m.reshape([ny,nx])
    if norm:
        m = m/np.max(m)
    return m
#
# ==============================================================================
#
def rgb_xvals(nx,ny,nz=1,norm=False):
    c = np.arange(3).reshape([1,1,1,3])
    x = np.arange(nx).reshape([1,1,nx,1])
    y = np.arange(ny).reshape([1,ny,1,1])
    z = np.arange(nz).reshape([nz,1,1,1])
    m = (1+0*c) * x * (1+0*y) * (1+0*z)
    if nz == 1:
        m = m.reshape([ny,nx,3])
    if norm:
        m = m/np.max(m)
    return m
#
# ==============================================================================
#
def rgb_yvals(nx,ny,nz=1,norm=False):
    c = np.arange(3).reshape([1,1,1,3])
    x = np.arange(nx).reshape([1,1,nx,1])
    y = np.arange(ny).reshape([1,ny,1,1])
    z = np.arange(nz).reshape([nz,1,1,1])
    m = (1+0*c) * (1+0*x) * y * (1+0*z)
    if nz == 1:
        m = m.reshape([ny,nx,3])
    if norm:
        m = m/np.max(m)
    return m
#
# ==============================================================================
#
def rgb_zvals(nx,ny,nz=1,norm=False):
    c = np.arange(3).reshape([1,1,1,3])
    x = np.arange(nx).reshape([1,1,nx,1])
    y = np.arange(ny).reshape([1,ny,1,1])
    z = np.arange(nz).reshape([nz,1,1,1])
    m = (1+0*c) * (1+0*x) * (1+0*y) * z
    if nz == 1:
        m = m.reshape([ny,nx,3])
    if norm:
        m = m/np.max(m)
    return m
#
# ==============================================================================
#
def rgb_svals(nx,ny,nz=1,norm=False):
    m = rgb_xvals(nx,ny,nz) + nx*rgb_yvals(1,ny,nz) + nx*ny*rgb_zvals(1,1,nz)
    if nz == 1:
        m = m.reshape([ny,nx,3])
    if norm:
        m = m/np.max(m)
    return m
#
# ==============================================================================
#
def rgb_rvals(nx,ny,nz=1,x0=0,y0=0,z0=0,norm=False):
    x = rgb_xvals(nx,ny,nz) - x0
    y = rgb_yvals(nx,ny,nz) - y0
    z = rgb_zvals(nx,ny,nz) - z0
    m = (x**2 + y**2 + z**2)**0.5
    if nz == 1:
        m = m.reshape([ny,nx,3])
    if norm:
        m = m/np.max(m)
    return m
#
# ==============================================================================
#
def downsize_image(fn=None,pc='10%',outfile='test.jpg',qual=90, comfile=None):
    if fn is None:
        print('must specify an input image name.\n')
    else:
        com = f'convert {fn}  -resize {pc} -quality {qual} {outfile}'
        print(com)
        if comfile is None:
            os.system(com)
        else:
            com += ';\n'
            save_textfile(com, comfile)
            print(f'command written to {comfile}')
#
# ==============================================================================
#
# Create meaningful image headers:
def create_astro_hdr(fname,xoff=0,yoff=0):
    gd = gdal.Open(fname)
    naxis1 = gd.RasterXSize
    naxis2 = gd.RasterYSize
    crval1,cdelt1,crpix1,crval2,crpix2,cdelt2 = gd.GetGeoTransform()

    # correct for the offsets:
    crval1 += xoff
    crval2 += yoff

    p1 = crval1 + (0 - crpix1)*cdelt1
    p2 = crval1 + (naxis1 - crpix1)*cdelt1
    q1 = crval2 + (0 - crpix2)*cdelt2
    q2 = crval2 + (naxis2 - crpix2)*cdelt2

    # Let's reference everything to the centre of the image:
    cx0 = naxis1 / 2
    cy0 = naxis2 / 2
    cp0 = 0.5*(p1 + p2)
    cq0 = 0.5*(q1 + q2)

    # I flip the image y-axes, so:
    cdelt2 = -cdelt2

    pix_dx = cn.deg2rad * cdelt1 * cn.r_earth
    pix_dy = cn.deg2rad * cdelt2 * cn.r_earth
    h = {}
    h['naxis1'] = naxis1
    h['naxis2'] = naxis2
    h['naxis3'] = 3
    h['cdelt1'] = cdelt1
    h['cdelt2'] = cdelt2
    h['cdelt3'] = 1
    h['crpix1'] = cx0
    h['crpix2'] = cy0
    h['crpix3'] = 0
    h['crval1'] = cp0
    h['crval2'] = cq0
    h['crval3'] = 0
    h['crota1'] = 0
    h['crota2'] = 0
    h['crota3'] = 0
    h['pix_dx'] = pix_dx
    h['pix_dy'] = pix_dy
    return h
#
# ==============================================================================
#

#
# ==============================================================================
#
def cv2_transform(im, angle=0, scale=1.0):
    im_x = im.shape[1]
    im_y = im.shape[0]
    center = (im_x//2, im_y//2)
    rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
    im_trans = cv2.warpAffine(im, rot_mat, (im_x, im_y))
    return im_trans
#
# ==============================================================================
#

def slice_geo_image(im0, hdr0,x1,x2,y1,y2):
    im = im0[y1:y2,x1:x2,:].copy()
    ny0,nx0,nz0 = im0.shape
    cx0 = 0.5*(x1+x2)
    cy0 = 0.5*(y1+y2)
    ny,nx,nz = im.shape
    cx = nx / 2
    cy = ny / 2
    pp0,qq0 = pixels2coords_hdr(cx0, cy0, hdr0)
    hdr = hdr0.copy()
    hdr['crval1'] = pp0
    hdr['crval2'] = qq0
    hdr['naxis1'] = nx
    hdr['naxis2'] = ny
    hdr['crpix1'] = cx
    hdr['crpix2'] = cy
    hdr['translationX'] = 0
    hdr['translationY'] = 0
    return im, hdr

#
# ==============================================================================
#
def coords2pixels_hdr(pp,qq, hdr):
    if 'crota1' in hdr:
        cr1 = hdr['crota1']
    else:
        cr1 = 0.0
    cr1c = math.cos(cr1*math.pi/180)
    cr1s = math.sin(cr1*math.pi/180)
    p0   = hdr['crval1']
    q0   = hdr['crval2']
    x0   = hdr['crpix1']
    y0   = hdr['crpix2']
    cd1  = hdr['cdelt1']
    cd2  = hdr['cdelt2']
    dp   = pp - p0
    dq   = qq - q0
    xx   = x0 + (dp/cd1)*cr1c + (dq/cd2)*cr1s
    yy   = y0 - (dp/cd1)*cr1s + (dq/cd2)*cr1c
    return xx,yy
#
# ==============================================================================
#
def pixels2coords_hdr(xx,yy, hdr):
    if 'crota1' in hdr:
        cr1 = hdr['crota1']
    else:
        cr1 = 0.0
    cr1c = math.cos(cr1*math.pi/180)
    cr1s = math.sin(cr1*math.pi/180)
    p0   = hdr['crval1']
    q0   = hdr['crval2']
    x0   = hdr['crpix1']
    y0   = hdr['crpix2']
    cd1  = hdr['cdelt1']
    cd2  = hdr['cdelt2']
    dx   = xx - x0
    dy   = yy - y0
    pp   = p0 + (dx*cd1)*cr1c - (dy*cd2)*cr1s
    qq   = q0 + (dx*cd1)*cr1s + (dy*cd2)*cr1c
    return pp,qq
#
# ==============================================================================
#
def get_coord_bounds(hdr):
    p1,q1 = pixels2coords_hdr(-0.5,-0.5, hdr)
    p2,q2 = pixels2coords_hdr(hdr['naxis1']-0.5,hdr['naxis2']-0.5, hdr)
    return p1,p2,q1,q2
#
# ==============================================================================
#
def coord_rotation(x1,y1,theta,nx,ny,scale):
    cr1c = math.cos(theta*math.pi/180)
    cr1s = math.sin(theta*math.pi/180)
    x0 = nx/2
    y0 = ny/2
    dx   = x1 - x0
    dy   = y1 - y0
    x2   = x0 + scale*(dx*cr1c - dy*cr1s)
    y2   = y0 + scale*(dx*cr1s + dy*cr1c)
    return x2,y2
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

#
# ==============================================================================
#


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

    wtext(smart_table_html, html_file, mode='overwrite')

# ==========================================


# ============
# print some useful information about a numpy array:
def np_info(nda):
    print("type\t: ", nda.dtype,"\nshape\t: ", nda.shape, "\nmin\t: ", nda.min(), "\nmax\t: ", nda.max(),"\n")
