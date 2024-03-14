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


# ==========================================
#
# ==============================================================================
#
ddef gdf2labelme(geo_df_xy,label_col,im_in,json_out,im_blob=False):

    jpg_out = json_out.replace('.json','.jpg')
    f_im = os.path.basename(jpg_out)
    im = Image.open(im_in)
    nx, ny = im.size

    shp0 = {}
    shp0['label']    = ''
    shp0['points']   = []
    shp0['group_id'] = None
    shp0['description'] = ''
    shp0['shape_type'] = 'polygon'
    shp0['flags'] = {}

    labelme = {}
    labelme['version'] = "5.3.1"
    labelme['flags']  = {}
    labelme['shapes'] = []
    labelme['imagePath'] = ''
    labelme['imageData'] = ''
    labelme['imageHeight'] = nx
    labelme['imageWidth']  = ny

    shapes = []
    for i,row in geo_df.iterrows():
        shp = shp0.copy()
        shp['label'] = row[label_col]
        vx, vy = row.geometry.exterior.coords.xy
        vx = list(np.array(vx))
        vy = list(ny - np.array(vy))
        ptys = [[x[0],x[1]] for x in zip(vx,vy)]
        shp['points'] = ptys
        shapes += [shp]

    labelme['shapes'] = shapes

    if im_blob:
        im1blob = open(im_in, "rb")
        im1blob = base64.b64encode(im1blob.read()).decode()
        labelme['imageData']   = im1blob
    labelme['imagePath']   = f_im
    labelme['imageHeight'] = nx
    labelme['imageWidth']  = ny


    im.save(jpg_out)
    json_data = json.dumps(labelme, indent=4)
    with open(json_out, 'w') as f:
        f.write(json_data)

#
# =====================

# Takes a GeoDF with multiple Geometry columns that may well be MultiPolygons and makes it tall,
# with an individual polygon per row.
def gdf_wide2tall(gdf, geom_cols, id_col, keep_cols=[]):
    
    if type(id_col) == str:
        id_cols = [id_col]
    else:
        id_cols = id_col.copy()

    gdf['tag'] = ''
    first_row = True
    for idc in id_cols:
        if first_row:
            gdf['tag'] = [f'{row[idc]}' for ind,row in gdf.iterrows()]
            first_row = False
        else:
            gdf['tag'] = [f'{row.tag}_{row[idc]}' for ind,row in gdf.iterrows()]
    
    id_col = 'tag'
    id_cols += ['tag']
    keep_cols += id_cols
    keep_cols_df = gdf[keep_cols].copy()
    keep_cols_df = keep_cols_df.drop_duplicates() 
        
    # Remove the empty polygons
    empty_polygon = wkt.loads('POLYGON EMPTY')
    geom_list_dupes = {}
    for i in range(len(gdf)):
        for cn in geom_cols:
            gr  = gdf.iloc[i]
            gid = gr[id_col]
            gg  = gr[cn]
            #print('gg:')
            #print(type(gg))
            test = 0
            if type(gg) == shapely.geometry.polygon.Polygon:
                test = 1
            if type(gg) == shapely.geometry.multipolygon.MultiPolygon:
                test = 1   
            if test == 1:
                if gg is not None:
                    if not gg.is_empty:
                        gname_suffix = f'n_{i:02}'
                        gname = f'{cn}_{gname_suffix}'
                        geom_list_dupes[gname] = {}
                        geom_list_dupes[gname][id_col] = gid
                        geom_list_dupes[gname]['geom'] = gg
                        geom_list_dupes[gname]['geometry_label'] = cn
                        geom_list_dupes[gname]['suffix'] = gname_suffix
                
    # dedupe the Multipolygons:
    geom_list = {}
    geom_list[id_col] = []
    geom_list['id_count'] = []
    geom_list['uid']  = []
    geom_list['geometry_label'] = []
    geom_list['geometry']  = []
    for cn in geom_list_dupes.keys():
        idc = 1
        gl  = geom_list_dupes[cn]
        gg  = gl['geom']
        gid = gl[id_col]
        ggl = gl['geometry_label']
        if gg.geom_type == 'Polygon':
            uid = f'{gid}_' + cn.replace(gl['suffix'],'') + f'{idc:03}'
            geom_list[id_col] += [gid]
            geom_list['uid']  += [uid]
            geom_list['id_count'] += [idc]
            geom_list['geometry_label'] += [ggl]
            geom_list['geometry'] += [gg]
            continue
        if gg.geom_type == 'MultiPolygon':
            for gg2 in gg.geoms:
                uid = f'{gid}_' + cn.replace(gl['suffix'],'') + f'{idc:03}'
                geom_list[id_col] += [gid]
                geom_list['uid']  += [uid]
                geom_list['id_count'] += [idc]
                geom_list['geometry_label'] += [ggl]
                geom_list['geometry'] += [gg2]
                idc += 1
                
    geom_list['geometry_label'] = [gl.replace('geomcol_','') for gl in geom_list['geometry_label']]
    
    new_gdf = gpd.GeoDataFrame(geom_list)
    
    new_gdf = new_gdf.merge(keep_cols_df, how='left', on='tag')
    return new_gdf


#
# ==============================================================================



# Produce a 2D Gaussian:
def Gaussian2D(xy, xo, yo, A, sig, offset):
    x, y = xy
    pi = 3.141592654
    xo = float(xo)
    yo = float(yo)
    dr2 = (x-xo)**2 + (y-yo)**2
    g = offset + A * np.exp(-dr2/(2 * sig**2))
    return g.ravel()



# transforms an image by a-degrees anticlockwise, x-pixels right and y-pixels down
# CHANGED Y to -Y because we want UP ot be positive-Y:
def superfast_transform(image, x, y, a):
    height, width = image.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=a, scale=1)
    rotate_matrix[0,2] += x
    rotate_matrix[1,2] -= y
    image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
    return (image)

def superfast_transform_old(image, x, y, a):
    height, width = image.shape[:2]
    center = (width/2, height/2)
    if not a == 0:
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=a, scale=1)
      image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
    translation_matrix = np.array([ [1, 0, x],[0, 1, y] ], dtype=np.float32)
    image = cv2.warpAffine(src=image, M=translation_matrix, dsize=(width, height))
    return (image)


def fast_rotate(image, a):
    height, width = image.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=a, scale=1)
    image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
    return (image)


def superfast_rotate(image, a):
    height, width = image.shape[:2]
    center = (width/2, height/2)
    if not a == 0:
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=a, scale=1)
      image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
    return (image)


def superfast_translate(image, x, y):
  height, width = image.shape[:2]
  center = (width/2, height/2)
  translation_matrix = np.array([ [1, 0, x],[0, 1, y] ], dtype=np.float32)
  image = cv2.warpAffine(src=image, M=translation_matrix, dsize=(width, height))
  return (image)


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



# ====================================================================
# General functions:

# Create the useful xvals, yvals and rvals
# To simplify things greatly they are limited here to 2D.
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

# Pad with zeroes:
# f'{3:05}'

# ============
# grab an image section from another image and make it rectangular
class unwarp_subimage:
  def __init__(self, image, im1vertices, newdims=None):
    if newdims is None:
        (h,w) = image.shape[:2]
    else:
        w = newdims[0]
        h = newdims[1]
    im2vertices = np.array([[0, 0], [0, h], [w, h], [w, 0]])
    tform = transform.ProjectiveTransform()
    tform.estimate(im2vertices, im1vertices)
    self.im = transform.warp(image, tform, output_shape=(h, w))


def int255_to_float01(image):
    return image.astype(np.float32)/255


def float01_to_int255(image):
    image = image * 255
    return image.astype(np.uint8)


# easy to remember alias:
def rgb_swap_rb(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


# remember - cv2.imread reads the channels into BGR.
class read_png:
  def __init__(self, filename, switch_rgb=False):
    # All files by default will be converted to floats to prevent rounding errors
    x = cv2.imread(filename, -1)

    self.orig_dtype = x.dtype.name

    if switch_rgb:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    x = x.astype(np.float32)
    self.filename = filename
    self.im = x

class read_png_grey:
  def __init__(self, filename):
    x = cv2.imread(filename, -1)
    #ndarray_info(x)
    self.orig_dtype = x.dtype.name
    x = x.astype(np.float32)
    #ndarray_info(x)
    fac_r = 0.30
    fac_g = 0.58
    fac_b = 0.12
    x = fac_r*x[:,:,2] + fac_g*x[:,:,1] + fac_b*x[:,:,0]
    #ndarray_info(x)
    self.filename = filename
    self.im = x

def bgr2grey(im):
    fac_r = 0.30
    fac_g = 0.58
    fac_b = 0.12
    x = fac_r*im[:,:,2] + fac_g*im[:,:,1] + fac_b*im[:,:,0]
    return (x)

def rgb2grey(im):
    fac_r = 0.30
    fac_g = 0.58
    fac_b = 0.12
    x = fac_r*im[:,:,0] + fac_g*im[:,:,1] + fac_b*im[:,:,2]
    return (x)

def write_png(x, filename, switch_rgb=False, depth=16):
    if switch_rgb:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, x)

def write_png_16(x, filename, switch_rgb=False):
    if switch_rgb:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    x = x.astype(np.uint16)
    cv2.imwrite(filename, x)


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


def superfast_reversetrans(image, x, y, a):
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


class read_png:
  def __init__(self, filename, switch_rgb=False):
    # All files by default will be converted to floats to prevent rounding errors
    x = cv2.imread(filename, -1)
    x = np.flipud(x)
    x = x[:,:,0:3]
    self.orig_dtype = x.dtype.name
    max_vals = np.array([1,255,65535])
    img_max = x.max()
    delta = abs(max_vals - img_max)
    col_max = max_vals[np.where(delta == min(delta))][0]
    col_max = col_max.astype(np.float32) * 1.0
    if switch_rgb:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    x = x.astype(np.float32)/col_max

    frame_num = ''
    file_path_no_suffix, file_extension = os.path.splitext(filename)
    file_name = os.path.basename(filename)
    if 'frame_' in filename:
        frame_num = filename.split('frame_')[-1].replace(file_extension,'')

    self.file_path = filename
    self.file_name = file_name
    self.frame_num = frame_num
    self.im = x


def determine_image_depth(im):
    max_vals = np.array([1,255,65535])
    img_max = im.max()
    delta = abs(max_vals - img_max)
    im_threshold = max_vals[np.where(delta == min(delta))][0]
    return (im_threshold)


def write_png(x, filename, switch_rgb=False, depth=16):
    if switch_rgb:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    x = np.flipud(x)
    x = (x*65535).astype(np.uint16)
    cv2.imwrite(filename, x)
