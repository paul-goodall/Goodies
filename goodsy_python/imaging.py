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

import shapely
from shapely import wkt

from matplotlib import colors as mcolors
import plotly.graph_objects as go
import random

import base64

from shapely.validation import make_valid
from shapely.geometry import Point, Polygon
from shapely.ops import transform
import geopandas as gpd

import math
from colorsys import hls_to_rgb
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

#
# ==============================================================================
#
# print some useful information about a numpy array:
def np_info(nda):
    print("type\t: ", nda.dtype,"\nshape\t: ", nda.shape, "\nmin\t: ", nda.min(), "\nmax\t: ", nda.max(),"\n")

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
            wTXT(com, comfile)
            print(f'command written to {comfile}')



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

#
# ==============================================================================
#

def hls2rgb(h,l=0.5,s=1.0):
    return hls_to_rgb(h, l, s)

#
# ==============================================================================
#

def rgb2hex(r, g, b):
    if (0.0 <= r <= 1.0) and (0.0 <= g <= 1.0) and (0.0 <= b <= 1.0):
        r = int(r*255)
        g = int(g*255)
        b = int(b*255)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

#
# ==============================================================================
#

def hls2hex(h,l=0.5,s=1.0):
    r,g,b = hls_to_rgb(h, l, s)
    return rgb2hex(r, g, b)

#
# ==============================================================================
#

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

#
# ==============================================================================
#

colours_bright = equidistant_colours(n=20, output='dict')

#plot_colortable(colours_bright, sort_colors=False)
#plt.show()

colours_light = equidistant_colours(n=20, l=0.7, output='dict')

#plot_colortable(colours_light, sort_colors=False)
#plt.show()

colours_dark = equidistant_colours(n=20, l=0.3, output='dict')

#plot_colortable(colours_dark, sort_colors=False)
#plt.show()

#
# ==============================================================================
#

def get_polygon_coords(poly):
    xy = gpd.GeoSeries(poly).get_coordinates()
    xx = np.array(xy.x)
    yy = np.array(xy.y)
    return xx,yy

#
# ==============================================================================
#

def get_polygon_bbox(poly):
    x,y = get_polygon_coords(poly)
    x1 = x.min()
    x2 = x.max()
    y1 = y.min()
    y2 = y.max()
    return x1,x2,y1,y2

#
# ==============================================================================
#

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



#
# ==============================================================================
#

def pil_imsave(img, jpgfile='/tmp/tmp.jpg'):
    data = Image.fromarray(img)
    #data = ImageOps.flip(data)
    data.save(jpgfile)

#
# ==============================================================================
#

def jpeg2bytes(jpgfile):
    im_blob1 = open(jpgfile, "rb").read()
    im_blob2 = base64.b64encode(im_blob1)
    return im_blob2

#
# ==============================================================================
#

def bytes2bytestring(bytestype):
    return bytestype.decode()

#
# ==============================================================================
#

def bytestring2bytes(bytestring):
    return bytestring.encode()

#
# ==============================================================================
#

def np2bytes(np_arr):
    jpgfile='/tmp/tmp.jpg'
    pil_imsave(np_arr, jpgfile)
    im_blob2 = jpeg2bytes(jpgfile)
    return im_blob2

#
# ==============================================================================
#

def np2bytestring(np_arr):
    im_blob2 = np2bytes(np_arr)
    im_blob3 = bytes2bytestring(im_blob2)
    return im_blob3

#
# ==============================================================================
#

def bytes2jpeg(bytestype, jpgfile='/tmp/tmp.jpg'):
    im_blob1 = base64.b64decode(bytestype)
    with open(jpgfile, 'wb') as f:
        f.write(im_blob1)

#
# ==============================================================================
#

def bytes2np(bytestype):
    jpgfile='/tmp/tmp.jpg'
    bytes2jpeg(bytestype, jpgfile)
    pil_im = Image.open(jpgfile)
    numpy_array = np.array(pil_im)
    return numpy_array

#
# ==============================================================================
#

def bytestring2np(bytestring):
    bytestype = bytestring2bytes(bytestring)
    return bytes2np(bytestype)

# ==========================================
#
# write a pickle:


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
def gdf2labelme(geo_df_xy,label_col,im_in,json_out,im_blob=False):

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




def write_png(x, filename, switch_rgb=False, depth=16):
    if switch_rgb:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    x = np.flipud(x)
    x = (x*65535).astype(np.uint16)
    cv2.imwrite(filename, x)


# Create meaningful image headers from image bounds:
def create_im_hdr(im, p1, p2, q1, q2):

    naxis2,naxis1,naxis3 = im.size

    # Let's reference everything to the centre of the image:
    cx0 = naxis1 / 2
    cy0 = naxis2 / 2
    cz0 = naxis3 / 2
    cp0 = 0.5*(p1 + p2)
    cq0 = 0.5*(q1 + q2)

    cdelt1 = (p2-p1)/naxis1
    cdelt2 = (q2-q1)/naxis2

    h = {}
    h['naxis1'] = naxis1
    h['naxis2'] = naxis2
    h['naxis3'] = naxis3
    h['cdelt1'] = cdelt1
    h['cdelt2'] = cdelt2
    h['cdelt3'] = 1
    h['crpix1'] = cx0
    h['crpix2'] = cy0
    h['crpix3'] = cz0
    h['crval1'] = cp0
    h['crval2'] = cq0
    h['crval3'] = 0
    h['crota1'] = 0
    h['crota2'] = 0
    h['crota3'] = 0

    return h
#
# ==============================================================================
#
