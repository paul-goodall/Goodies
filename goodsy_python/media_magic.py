import os
import sys
import pandas as pd
from glob import glob
import argparse
from mutagen.mp3 import MP3
import subprocess
import json

# ========

def copy_album(orig_album, new_album):
    com = f'''
    mkdir -p {new_album};
    cp -rf {orig_album}/* {new_album}/.
    '''
    if not os.path.exists(new_album):
        print(com)
        os.system(com)
    else:
        print(f'Album already exists: {new_album}')


# ========

def exif_set_CreateDate(fname,create_date='2024:01:01 12:34:56'):
    com = f'exiftool -overwrite_original -CreateDate="{create_date}" {fname}'
    print(com)
    os.system(com)

# ========

def get_youtube_audio(yt_url, outname='soundfile', aformat='mp3'):
    com = f'yt-dlp -x --audio-format {aformat} --audio-quality 0 -o {outname} {yt_url}'
    if not os.path.exists(f'{outname}.{aformat}'):
        print(com)
        os.system(com)

# ========

def video_replace_audio(orig_vid, new_audio, new_vid):
    com = f'ffmpeg -i {orig_vid} -i {new_audio} -c:v copy -map 0:v:0 -map 1:a:0 {new_vid}'
    print(com)
    os.system(com)

# ========

def convert_jpg_folder_to_mp4(album_dir, framerate, outfile):
    com = f'''
    ffmpeg -framerate {framerate} -pattern_type glob -i '{album_dir}/*.jpg' -c:v libx264 -vsync 2 -r {framerate} -pix_fmt yuv420p {outfile}
    '''
    print(com)
    os.system(com)

# ========

def trim_audio(input,output,t1,dt):
    com = f'''
    ffmpeg -ss {t1} -i {input} -t {dt} -c copy {ouput}
    '''
    print(com)
    os.system(com)

# ========

def change_wav_speed(orig_file, new_file, speedfac=0.9):
    x = speedfac
    y = 1/x
    com = f'''
    ffmpeg -y -i {orig_file} -af atempo={x} {new_file}
    '''
    print(com)

    os.system(com)


# ========

def get_image_exif(f0):
    res = subprocess.check_output(['exiftool','-json',f'{f0}'])
    res = json.loads(res)[0]

    for key in res:
        res[key] = str(res[key])

    try:
      df = pd.DataFrame(res)
    except:
      df = pd.DataFrame(res, index=[0])

    return df

# ========

def get_album_metadata(album_dir,album_meta=None,verbose=1):

    if album_meta is None:
        album_meta = f'{album_dir}/metadata.csv'

    album_name = album_dir.split('/')[-1]

    ff = glob(f'{album_dir}/*')
    ff.sort()

    df = None

    if not os.path.exists(album_meta):
        for f in ff:
            if f[-4:] == '.csv':
                next

            if verbose == 2:
                print(f)
            elif verbose == 1:
                print('.', end='')

            d = get_image_exif(f)
            d['origfile'] = f
            suff = f.split('.')[-1].lower().replace('jpeg','jpg')
            d['suffix']  = suff
            df = pd.concat([df,d])

        df = df.reset_index(drop=True)

        df['album_name'] = album_name
        df.to_csv(album_meta, index=False)
    else:
        df = pd.read_csv(album_meta)

    return df

# ========

def process_metadata_dates(df):

# Take either CreateDate or DateTimeOriginal, otherwise assign the mean album date
    cols = df.columns
    if 'CreateDate' in cols:
        df['CreateDate']       = pd.to_datetime(df['CreateDate'],       format="%Y:%m:%d %H:%M:%S",   utc=True)
    if 'DateTimeOriginal' in cols:
        df['DateTimeOriginal'] = pd.to_datetime(df['DateTimeOriginal'], format="%Y:%m:%d %H:%M:%S",   utc=True)

    df['BestDate'] = df['CreateDate']
    ii = df['BestDate'].isna()
    if 'DateTimeOriginal' in cols:
        df.loc[ii,'BestDate'] = df['DateTimeOriginal'][ii]

    mean_datetime = df.CreateDate.pipe(lambda d: (lambda m: m + (d - m).mean())(d.min())).round('s').to_pydatetime()
    ii = df['BestDate'].isna()
    if 'FileModifyDate' in cols:
        df.loc[ii,'BestDate'] = mean_datetime

    df['BestDate'] = pd.to_datetime(df['BestDate'],       format="%Y:%m:%d %H:%M:%S",   utc=True)

    df = df.sort_values(['BestDate','origfile'])
    df = df.reset_index(drop=True)
    df['album_item'] = range(1,len(df)+1)
    album_dir = os.path.split(df['SourceFile'][0])[0]
    df['newfile'] = [ f'{album_dir}/{r.album_name}_{r.album_item:04}.{r.suffix}' for i,r in df.iterrows() ]

    return df

# ========

def rename_files(df,verbose=1):
    for i,r in df.iterrows():
        f1 = r.origfile
        f2 = r.newfile
        if not os.path.exists(f2):
            com = f'mv {f1} {f2}'
            if verbose == 2:
                print(com)
            elif verbose == 1:
                print('.', end='')
            os.system(com)

# ========

def heic2jpg(df):

    album_dir = os.path.split(df['newfile'][0])[0]
    ff = glob(f'{album_dir}/*.heic')

    if(len(ff) > 0):
        com = f'''
        magick mogrify -format jpg {album_dir}/*.heic;
        rm {album_dir}/*.heic;
        '''
        print(com)
        os.system(com)
        df['newfile'] = [  x.replace('.heic','.jpg') for x in df['newfile'].tolist() ]

    return df

# ========

def mov2mp4(movfile,reencode=False,crf=23):

    # If re-encoding:
    # CRF values range from 0 to 51
    # 0 = best
    # 51 = worst
    # 23 = about same file size as input MOV
    # < 20 I can't really see any inmprovements over the original,
    # but file size is massively inflated

    mp4file = movfile.replace('.mov','.mp4')

    quiet = '-hide_banner -loglevel error'

    if reencode:
        vid_com = f'ffmpeg {quiet} -i {movfile} -crf {crf} -pix_fmt yuv420p {mp4file}'
    else:
        vid_com = f'ffmpeg {quiet} -i {movfile} -c copy -tag:v hvc1 {mp4file}'

    os.system(vid_com)

    return mp4file

# ========

def df_png2jpg(df):
    print('need to do this')

    return df

# ========

def pics2jpg(df):
    # Will need to add something for every other format,
    # e.g. png, tiff, etc etc

    df = df_png2jpg(df)

    return df

# ========

def df_mov2mp4(df):

    for i,r in df.iterrows():
        f = r.newfile
        suff = f.split('.')[-1]
        if suff == 'mov':
            if os.path.exists(f):
                newf = mov2mp4(f)
                com = f'rm {f}'
                os.system(com)

    df['newfile'] = [ x.replace('.mov','.mp4') for x in df['newfile'].tolist() ]

    return df

# ========

def movies2mp4(df):

    # Will need to add something for every other format,
    # e.g. mpg, avi, 3gp, etc etc

    df = df_mov2mp4(df)

    return df

# ========

def auto_orient_pics(df):
    album_dir = os.path.split(df['newfile'][0])[0]
    com = f'magick mogrify -auto-orient {album_dir}/*jpg;'
    os.system(com)

# ========


def create_prepped_album(album_dir,new_album):

    com = f'''
    mkdir -p {new_album};

    cp {album_dir}/*jpg {new_album}/.;

    magick mogrify -auto-orient {new_album}/*jpg;
    magick mogrify -resize '{width}x>' {new_album}/*jpg;
    magick mogrify -resize '>x{height}' {new_album}/*jpg;
    magick mogrify -background black -gravity center -extent '{width}x<' {new_album}/*jpg;
    magick mogrify -background black -gravity center -extent '>x{height}' {new_album}/*jpg;
    '''

    if not os.path.exists(new_album):
        print(com)
        os.system(com)

# ========
