import numpy as np
import cv2
import os
import glob
import argparse
import json
import torch
from torch.utils.data import DataLoader,Dataset
# from yolov7_mask.seg.segment.utils_draw import read_file

class extract_mask(Dataset):
    def __init__(self,file_path):
        super(extract_mask,self).__init__()
        self.file_path= file_path
        self.json_file=sorted(glob.glob(f'{file_path}/*.*.json'))
        self.image_file=sorted(glob.glob(f'{file_path}/*.bmp'))
        self.color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
    def __getitem__(self, index):
        json_path= self.json_file[index]
        image=cv2.imread(self.image_file[index])
        image= np.zeros((image.shape[0],image.shape[1]))
        with open(json_path, 'r') as f:
            json_object= json.load(f)
        for n in range(len(json_object['regions'])):
            region_X=json_object['regions'][f'{n}']['List_X']
            region_Y=json_object['regions'][f'{n}']['List_Y']
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
            final= cv2.polylines(image, [pts], True, self.color,3)            
        return final
    
    def __len__(self):
        return len(self.json_file)

file_path='/home/tonyhuy/Yolov7_box_segmentation/Box_data/Filling_2023_05_16/train/'
final= extract_mask(file_path)  
mask=final.__getitem__(3)
# mask =cv2.resize(mask,(320,320),cv2.INTER_LINEAR)
cv2.imwrite(f'/home/tonyhuy/Yolov7_box_segmentation/result_mask/img.jpg',mask)

exit()
def choose_npath(file,num_path):
    if(isinstance(file,list)): 
        return file[:num_path]

def read_file(img_file,json_file,mode,num_path,index):
    """
    Read json file
    """
    assert mode in ['image','json','both']
    if mode=='json':
        file_list=choose_npath(json_file,num_path)[index]
        with open(file_list, 'r') as f:
            json_object= json.load(f)
        return json_object
    elif(mode=='image'):
        file_list=choose_npath(img_file,num_path)[index]
        image=cv2.imread(image_obj)
        return image
    elif mode=='both':
        file_list=choose_npath(json_file,num_path)[index]
        img_list=choose_npath(img_file,num_path)[index]
        with open(file_list, 'r') as f:
            json_object= json.load(f)
        image_obj=cv2.imread(img_list)
        return json_object,image_obj,img_list

class CFG:
    num_path=8
    index=1
    file_path='/home/tonyhuy/Yolov7_box_segmentation/Box_data/Filling_2023_05_16/train/'
    json_file=sorted(glob.glob(f'{file_path}/*.*.json'))
    image_file=sorted(glob.glob(f'{file_path}/*.bmp'))
    color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))

# json_obj,image=read_file(CFG.image_file,CFG.json_file,'both',CFG.num_path,CFG.index)

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
