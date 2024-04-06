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

from matplotlib import colors as mcolors
import plotly.graph_objects as go
import random


import math
from colorsys import hls_to_rgb
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


def plot_colortable(colors, *, ncols=2, sort_colors=True):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig




def hls2rgb(h,l=0.5,s=1.0):
    return hls_to_rgb(h, l, s)

    
def rgb2hex(r, g, b):
    if (0.0 <= r <= 1.0) and (0.0 <= g <= 1.0) and (0.0 <= b <= 1.0):
        r = int(r*255)
        g = int(g*255)
        b = int(b*255)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def hls2hex(h,l=0.5,s=1.0):
    r,g,b = hls_to_rgb(h, l, s)
    return rgb2hex(r, g, b)

def equidistant_colours(n=5, l=0.5, s=1.0, output='list'):
    delta = 1.0/(n+1)
    h = np.arange(n) * delta
    if output == 'list':
        colours = [ hls2hex(x,l,s) for x in h ]
    else:
        colours = {}
        for i in range(n):
            colours[f'colour_{i+1}_of_{n}'] = hls2hex(h[i],l,s)
    return colours
   
    
    
colours_bright = equidistant_colours(n=20, output='dict')

#plot_colortable(colours_bright, sort_colors=False)
#plt.show()

colours_light = equidistant_colours(n=20, l=0.7, output='dict')

#plot_colortable(colours_light, sort_colors=False)
#plt.show()

colours_dark = equidistant_colours(n=20, l=0.3, output='dict')

#plot_colortable(colours_dark, sort_colors=False)
#plt.show()



# ==========================================
#

def get_polygon_coords(poly):
    xy = gpd.GeoSeries(poly).get_coordinates()
    xx = np.array(xy.x)
    yy = np.array(xy.y)
    return xx,yy

def get_polygon_bbox(poly):
    x,y = get_polygon_coords(poly)
    x1 = x.min()
    x2 = x.max()
    y1 = y.min()
    y2 = y.max()
    return x1,x2,y1,y2

# warning, this doesn't consider degrees the change in longitude as a func of lat.
def get_polygon_squarebox(poly, margin_percent=0): 
    x1,x2,y1,y2 = get_polygon_bbox(poly)
    x0 = 0.5*(x1+x2)
    y0 = 0.5*(y1+y2)
    dx = x2-x1
    dy = y2-y1
    delta = np.max([dx,dy])
    x1 = x0 + 0.5*(delta*(1+margin_percent/100))
    x2 = x0 - 0.5*(delta*(1+margin_percent/100))
    y1 = y0 + 0.5*(delta*(1+margin_percent/100))
    y2 = y0 - 0.5*(delta*(1+margin_percent/100))
    return x1,x2,y1,y2



# ==========================================
#
def pil_imsave(img, jpgfile='/tmp/tmp.jpg'):
    data = Image.fromarray(img)
    #data = ImageOps.flip(data)
    data.save(jpgfile)

def jpeg2bytes(jpgfile):
    im_blob1 = open(jpgfile, "rb").read()
    im_blob2 = base64.b64encode(im_blob1)
    return im_blob2

def bytes2bytestring(bytestype):
    return bytestype.decode()

def bytestring2bytes(bytestring):
    return bytestring.encode()

def np2bytes(np_arr):
    jpgfile='/tmp/tmp.jpg'
    pil_imsave(np_arr, jpgfile)
    im_blob2 = jpeg2bytes(jpgfile)
    return im_blob2

def np2bytestring(np_arr):
    im_blob2 = np2bytes(np_arr)
    im_blob3 = bytes2bytestring(im_blob2)
    return im_blob3

def bytes2jpeg(bytestype, jpgfile='/tmp/tmp.jpg'):
    im_blob1 = base64.b64decode(bytestype)
    with open(jpgfile, 'wb') as f:
        f.write(im_blob1)
    
def bytes2np(bytestype):
    jpgfile='/tmp/tmp.jpg'
    bytes2jpeg(bytestype, jpgfile)
    pil_im = Image.open(jpgfile)
    numpy_array = np.array(pil_im)
    return numpy_array
    
def bytestring2np(bytestring):
    bytestype = bytestring2bytes(bytestring)
    return bytes2np(bytestype)



# ==========================================
#
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

# ==========================================
#            
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

# ==========================================
#
def plot_tall_gdf(tall_gdf, labelcol, colourcol=None,textcol=None,opacitycol=None,plottheme='none'):

    preferred_colours = ['deeppink','magenta','blueviolet','slateblue','cornflowerblue',
                         'deepskyblue','cyan','teal','lime','green','gold','darkorange','red','maroon']
    random.shuffle(preferred_colours)
    if colourcol is None:
        labels = tall_gdf[labelcol].unique()
        color_mapping = {}
        for i in range(len(labels)):
            color_mapping[labels[i]] = preferred_colours[i]
        colourcol = 'plot_colour'
        tall_gdf[colourcol] = tall_gdf[labelcol].map(color_mapping)
    
    if textcol is None:
        textcol = 'plot_label'
        tall_gdf[textcol] = ''
    
    if opacitycol is None:
        opacitycol = 'plot_opacity'
        tall_gdf[opacitycol] = 0.5
    
    
    # set up multiple traces
    traces = []
    for ind, row in tall_gdf.iterrows():
        x, y = row.geometry.exterior.coords.xy
        x = np.array(x)
        y = np.array(y)
        my_trace = go.Scatter(
                x=x,
                y=y,
                opacity=row[opacitycol],
                fill='toself',
                mode='lines',
                name=row[labelcol],
                line_color=row[colourcol],
                text=row[textcol]
            )
        
        traces += [my_trace]
    
    # set up the buttons:
    buttons = []
    my_dataset = ['all'] + list(tall_gdf.dataset.unique())
    for dd in my_dataset:
        if dd == 'all':
            my_args2 = [{'visible':True}]
            my_args1 = [{'visible':'legendonly'}]
        else:
            my_args2 = [{'visible':True},[i for i,x in enumerate(traces) if dd in x.name]]
            my_args1 = [{'visible':'legendonly'},[i for i,x in enumerate(traces) if dd in x.name]]
        
        my_button = dict(method='restyle',
                                label=dd,
                                visible=True,
                                args=my_args1,
                                args2=my_args2,
                                )
        buttons += [my_button]
    
    # create the layout 
    layout = go.Layout(
        updatemenus=[
            dict(
                type='buttons',
                direction='right',
                x=0.5,
                y=1.05,
                showactive=True,
                buttons=buttons
            )
        ],
        title=dict(text='Toggle Layers by dataset:',x=0.4,y=0.99),
        showlegend=True
    )
    
    ouput_fig = go.Figure(data=traces,layout=layout)

    ouput_fig.update_layout(
        autosize=False,
        template=plottheme
    )
    return ouput_fig

# ==============================================================================
#
def pq2xy(pp,qq,hdr):
    pp = np.array(pp)
    qq = np.array(qq)
    xx = hdr['x0'] + (pp - hdr['p0'])/hdr['dpdx']
    yy = hdr['y0'] + (qq - hdr['q0'])/hdr['dqdy']
    yy = hdr['ny'] - yy
    return xx,yy   

def xy2pq(xx,yy,hdr):
    xx = np.array(xx)
    yy = hdr['ny'] - np.array(yy)
    pp = hdr['p0'] + (xx - hdr['x0']) * hdr['dpdx'] 
    qq = hdr['q0'] + (yy - hdr['y0']) * hdr['dqdy'] 
    return pp,qq 

def clip_image_with_hdr(im,hdr,x1,x2,y1,y2):
    im2 = im[y1:y2,x1:x2,:].copy()
    
    x0 = 0.5 * (x1 + x2)
    y0 = 0.5 * (y1 + y2)
    p0,q0 = xy2pq(x0,y0,hdr)
    ny,nx,nz = im2.shape
    
    hdr['x0'] = 0.5*nx
    hdr['y0'] = 0.5*ny
    hdr['nx'] = nx
    hdr['ny'] = ny
    hdr['p0'] = p0
    hdr['q0'] = q0
    
    return im2,hdr
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

def dedupe_tall_gdf(tall_gdf, extra_dedup_cols=[]):
    tall_gdf['row_gid'] = np.arange(len(tall_gdf))
    tall_gdf['geom_area'] = tall_gdf.geometry.area
    xy = tall_gdf.geometry.centroid.get_coordinates()
    tall_gdf['geom_xc'] = xy.x
    tall_gdf['geom_yc'] = xy.y
    dedup_cols = ['geom_area','geom_xc','geom_yc'] + extra_dedup_cols
    select_rows = tall_gdf.groupby(dedup_cols)['row_gid'].min().reset_index()
    result = select_rows[['row_gid']].merge(tall_gdf, how='left', on='row_gid')
    return result

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
