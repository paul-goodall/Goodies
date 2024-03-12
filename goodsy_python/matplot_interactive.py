import base64
import json
import sys
import labelme
import matplotlib.pyplot as plt
import numpy as np
import glob

screen_x = int(sys.argv[1])
screen_y = int(sys.argv[2])
json_filelist = sys.argv[3:]


def wjson(dict,fname):
    with open(fname, 'w') as f:
        f.write(json.dumps(dict))

def rjson(fname):
    with open(fname, 'r') as f:
        return json.load(f)


def put_plotinfo(data):
    return wjson(data,'docs/plotinfo.json')

def get_plotinfo():
    data = rjson('docs/plotinfo.json')
    return data

def increment_framenum():
    data = get_plotinfo()
    val = data['framenum']
    val += 1
    data['framenum'] = val
    put_plotinfo(data)
    return data

def decrement_framenum():
    data = get_plotinfo()
    val = data['framenum']
    val -= 1
    data['framenum'] = val
    put_plotinfo(data)
    return data

def toggle_layers():
    data = get_plotinfo()
    val = data['labels_on']
    val *= -1
    data['labels_on'] = val
    put_plotinfo(data)
    return data

def onclick(event):
    plot_info = get_plotinfo()
    print('key: ' + str(event.key))
    if event.key == 'right':
        plot_info = increment_framenum()
    if event.key == 'left':
        plot_info = decrement_framenum()
    if event.key == 'tab':
        plot_info = toggle_layers()

    view_filenum(plot_info, json_filelist)





def view_filenum(plot_info,json_filelist):

    fig=plt.gcf()
    ax=plt.gca()
    ax.cla()

    fn = plot_info['framenum']
    labels_on = plot_info['labels_on']
    nfiles = len(json_filelist)
    if fn > (nfiles-1):
        fn = 0
    if fn < 0:
        fn = (nfiles-1)

    plot_info['framenum'] = fn
    #print('Looking at frame: ' + str(fn))
    put_plotinfo(plot_info)

    json_file = json_filelist[fn]
    with open(json_file, "r") as f:
        data = json.load(f)

    url = data['shapes'][0]['description']
    print(url)
    # load image from data["imageData"]
    image = labelme.utils.img_b64_to_arr(data["imageData"])
    #print("image:", image.shape, image.dtype)

    # load label_names, label, label_points_xy from data["shapes"]
    unique_label_names = ["_background_"] + sorted(
        set([shape["label"] for shape in data["shapes"]])
    )
    label = np.zeros(image.shape[:2], dtype=np.int32)
    label_names = []
    label_points_xy = []
    for shape in data["shapes"]:
        label_id = unique_label_names.index(shape["label"])

        label_names.append(shape["label"])

        mask = labelme.utils.shape_to_mask(
            img_shape=image.shape[:2],
            points=shape["points"],
            shape_type=shape["shape_type"],
        )
        label[mask] = label_id
        label_points_xy.append(shape["points"])
    #print("label:", label.shape, label.dtype)
    #print("label_names:", label_names)


    colors = "rgbymc"
    ax.imshow(image)
    if labels_on == 1:
        for i, (label_name, label_points_xy_i) in enumerate(zip(label_names, label_points_xy)):
            label_id = unique_label_names.index(label_name)
            label_points_xy_i = np.array(label_points_xy_i)
            ax.plot(
                label_points_xy_i[:, 0],
                label_points_xy_i[:, 1],
                marker='',
                color=colors[label_id % len(colors)],
                label=label_name if label_name not in label_names[:i] else None,
            )
        ax.legend()

    plt.tight_layout()
    plt.suptitle(json_file)

    plt.draw()

    return True




plot_info = {}
plot_info['framenum']  = 0
plot_info['labels_on'] = 1
put_plotinfo(plot_info)

fig, ax = plt.subplots(1, 1, figsize=(screen_x, screen_y))
view_filenum(plot_info,json_filelist)
cid = fig.canvas.mpl_connect('key_press_event', onclick)

plt.show()
