import numpy as np
import cv2
import os
import glob
import argparse
import json
import torch
import random
from torch.utils.data import DataLoader,Dataset
# from yolov7_mask.seg.segment.utils_draw import read_file

class extract_mask(Dataset):
    def __init__(self,file_path):
        super(extract_mask,self).__init__()
        self.file_path= file_path
        self.json_file=sorted(glob.glob(f'{file_path}/*.*.json'))
        self.image_file=sorted(glob.glob(f'{file_path}/*.bmp'))
        self.background_folder = sorted(glob.glob('/home/tonyhuy/bottle_classification/data_bottle_detection/coco/images/*'))
        self.color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
        
    def __getitem__(self, index):
        json_path= self.json_file[index]
        image=cv2.imread(self.image_file[index])
        bg_img=cv2.imread(self.background_folder[index])
        # # Pad to height
        # pad_height = image.shape[0]-bg_img.shape[0]
        # # Pad to width
        # pad_width = image.shape[1]-bg_img.shape[1]
        # # Pad to top
        # top = pad_height // 2
        # # Pad to bottom
        # bottom = pad_height - top
        # # Pad to left
        # left = pad_width // 2
        # # Pad to right
        # right = pad_width - left
        # bg_img = cv2.copyMakeBorder(bg_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # print(bg_img.shape, image.shape)
         # if background > image ==> crop
        if bg_img.shape[0] > image.shape[0] and bg_img.shape[1] > image.shape[1]:
            x = random.randint(0, bg_img.shape[1] - image.shape[1] -1)
            y = random.randint(0, bg_img.shape[0] - image.shape[0] -1)
            bg_img = bg_img[y:y + image.shape[0], x: x + image.shape[1]]
        else:
            bg_img = cv2.resize(bg_img, (image.shape[1], image.shape[0]))       
        # Create zero mask
        mask= np.zeros((image.shape[0],image.shape[1]))
        with open(json_path, 'r') as f:
            json_object= json.load(f)
        for n in range(len(json_object['regions'])):
            region_X=json_object['regions'][f'{n}']['List_X']
            region_Y=json_object['regions'][f'{n}']['List_Y']
            xmin,xmax,ymin,ymax=int(min(region_X)),int(max(region_X)),int(min(region_Y)),int(max(region_Y))
            bg_img[ymin:ymax,xmin:xmax]=image[ymin:ymax,xmin:xmax]
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
            mask= cv2.polylines(mask, [pts], True, self.color,3)
            # bg_img= cv2.polylines(bg_img, [pts], True, (112,244,124),3)            

        return bg_img,mask
    
    def __len__(self):
        return len(self.json_file)

file_path='/home/tonyhuy/Yolov7_box_segmentation/Box_data/Filling_2023_05_16/train/'
save_dir='/home/tonyhuy/Yolov7_box_segmentation/result_mask/change_bg'

dataset= extract_mask(file_path)  
for i,(image,mask) in enumerate(dataset):
    cv2.imwrite(os.path.join(save_dir,'images',f'image_{i}.jpg'),image)
    cv2.imwrite(os.path.join(save_dir,'masks',f'mask_{i}.jpg'),mask)

    # cv2.imwrite(f'/home/tonyhuy/Yolov7_box_segmentation/result_mask/images/img_{i}.jpg',image)
    # cv2.imwrite(f'/home/tonyhuy/Yolov7_box_segmentation/result_mask/masks/mask_{i}.jpg',mask)
    if i==len(dataset)-1:
        break