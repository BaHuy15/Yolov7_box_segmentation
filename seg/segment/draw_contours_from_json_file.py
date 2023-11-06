import numpy as np
import cv2
import os
import glob
import argparse
import json
from yolov7_mask.seg.segment.utils_draw import read_file

class CFG:
    num_path=8
    index=1
    file_path='/home/tonyhuy/TOMO/Instance_segmentation/Dataset/data_blister_10_11/train/'
    json_file=sorted(glob.glob(f'{file_path}/*.*.json'))
    image_file=sorted(glob.glob(f'{file_path}/*.bmp'))
    color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))

json_obj,image=read_file(CFG.image_file,CFG.json_file,'both',CFG.num_path,CFG.index)
for n in range(6):
    region_X=json_obj['regions'][f'{n}']['List_X']
    region_Y=json_obj['regions'][f'{n}']['List_Y']
    i,j=0,0
    left=0
    m,k=[],[]
    while left<len(region_X):
        if(i==j):
            k.append(region_X[i])
            k.append(region_Y[j])
            m.append(k)
            k=[]
            i+=1
            j+=1
        left+=1
        if left >len(region_X):
            break
    pts = np.array([m], np.int32)
    final= cv2.polylines(image, [pts], True, CFG.color,3)
    if n==5:
        cv2.imwrite(f'/home/tonyhuy/TOMO/Instance_segmentation/yolov7_mask/seg/segment/draw_img/segment_mask_{n}.jpg',final)
