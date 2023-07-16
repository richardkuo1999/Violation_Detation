import cv2
import json
import math
import numpy as np
from pathlib import Path

def get_degrees(diff):
    degrees = math.degrees(math.atan2(diff[0], diff[1]))
    degrees += 360 if degrees < 0 else 0
    return degrees

def write_log(results_file, msg):
    with open(results_file, 'a') as f:
        print(msg)
        f.write(msg+'\n')



def loadGPS(source):
    GPSList = [] 
    if Path(source+'/mapData.json').exists():
        with open((source+'/mapData.json'), 'r') as r:
            GPSList = json.load(r)['data']
        
    else:
        print('NO GPS data')
    # if Path(source+'/GPS.txt').exists():
    #     with open((source+'/GPS.txt'), 'r') as r:
    #         for line in r.readlines():
    #             line = list(map(float,line.strip('\n').split(',')))
    #             GPSList.append(line)
    # else:
    #     print('NO GPS data')
    return GPSList

def fixBBox(box, orig_shape):
    box = list(map(int, box.tolist()))
    for i in range(len(box)):
            if box[i] < 0:
                box[i] = 0
    if box[0] > orig_shape[1]:
        box[0] = orig_shape[1]
    if box[2] > orig_shape[1]:
        box[2] = orig_shape[1]
        
    if box[1] > orig_shape[0]:
        box[1] = orig_shape[0]
    if box[3] > orig_shape[0]:
        box[3] = orig_shape[0]
    return box

def save2json(data, save_path):
    json_object = json.dumps(data)
    json_path =  save_path / 'parking.json'
    with open(json_path, "w") as outfile:
        outfile.write(json_object)

def xywhn2xyxy(x, shape = (640,640)):
    h, w = shape
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y=x.clone().tolist()
    y[0] = int(w * (x[0] - x[2] / 2))  # top left x
    y[0] = 0 if y[0] < 0 else y[0]

    y[1] = int(h * (x[1] - x[3] / 2))  # top left y
    y[1] = 0 if y[1] < 0 else y[1]
    
    y[2] = int(w * (x[0] + x[2] / 2))  # bottom right x
    y[2] = w if y[2] > w else y[2]

    y[3] = int(h * (x[1] + x[3] / 2))  # bottom right y
    y[3] = h if y[3] > h else y[3]

    return y

def letterbox(combination, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    """Resize the input image and automatically padding to suitable shape :https://zhuanlan.zhihu.com/p/172121380"""
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    img, seg_label, lane_label = combination
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # height, width ratios
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dh, dw = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[0], new_shape[1])
        ratio = new_shape[0] / shape[0], new_shape[1] / shape[1]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad[::-1], interpolation=cv2.INTER_LINEAR)
        seg_label = cv2.resize(seg_label, new_unpad[::-1], interpolation=cv2.INTER_NEAREST)
        lane_label = cv2.resize(lane_label, new_unpad[::-1], interpolation=cv2.INTER_NEAREST)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    seg_label = cv2.copyMakeBorder(seg_label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))  # add border
    lane_label = cv2.copyMakeBorder(lane_label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))  # add border
    # print(img.shape)
    
    combination = (img, seg_label, lane_label)
    return combination, ratio, (dw, dh)