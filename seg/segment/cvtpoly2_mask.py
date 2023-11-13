import numpy as np
import cv2
import os
import glob
import argparse
import json
import torch
import random
import argparse
from torch.utils.data import DataLoader,Dataset
import pandas as pd
# from yolov7_mask.seg.segment.utils_draw import read_file



def extract_roi(image,roi):
    # https://stackoverflow.com/questions/15341538/numpy-opencv-2-how-do-i-crop-non-rectangular-region
    mask = np.zeros(image.shape, dtype=np.uint8)
    # roi_corners = np.array([[(10,10), (300,300), (10,300)]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    roi_corners=np.array([roi], np.int32)
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    # apply the mask
    masked = cv2.bitwise_and(image, mask)
    masked = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
    masked[masked>0] = 255
    return masked

class extract_mask(Dataset):
    def __init__(self,img_file_path,label_file_path,background_path):
        super(extract_mask,self).__init__()
        # self.json_file=sorted(glob.glob(f'{label_file_path}/*.*.json'))
        self.file_path=sorted(glob.glob(f'{img_file_path}/*'))
        self.background_folder = sorted(glob.glob(f'{background_path}/*'))
        self.label_file_path=label_file_path
        self.color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
        
    def __getitem__(self, index):
        # json_path= self.json_file[index]
        file_path=self.file_path[index]
        idx_name= file_path.split("/")[-1]
        json_path=os.path.join(self.label_file_path,f'{idx_name}.json')
        json_name=f'{idx_name}.json'
        image=cv2.imread(file_path)
        with open(json_path, 'r') as f:
            json_object= json.load(f)
        for n in range(len(json_object['regions'])):
            region_X=json_object['regions'][f'{n}']['List_X']
            region_Y=json_object['regions'][f'{n}']['List_Y']
            # Combine X and Y to create list Z
            Z = list(zip(region_X, region_Y))

            # If you need Z as a list of lists:
            roi = list([list(t) for t in Z])
            mask=extract_roi(image,roi)
            # bg_img= cv2.polylines(bg_img, [pts], True, (112,244,124),3)            

        return image,mask,idx_name,roi,region_X,region_Y,json_name
    
    def __len__(self):
        return len(self.json_file)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', nargs='+', type=str, default='/home/tonyhuy/Yolov7_box_segmentation/Box_data/Filling_2023_05_16/train/', help='model path(s)')
    parser.add_argument('--save_dir', type=str, default='/home/tonyhuy/Yolov7_box_segmentation/result_mask', help='save directory')
    parser.add_argument('--file_names', type=str, default='images', help='save directory')
    opt = parser.parse_args()
    return opt

# Path to image dir
img_file_path='/home/tonyhuy/Yolov7_box_segmentation/Box_data/dataset1/separate/images'
label_file_path='/home/tonyhuy/Yolov7_box_segmentation/Box_data/dataset1/separate/labels'

save_dir='/home/tonyhuy/Yolov7_box_segmentation/result_mask'
file_names=['images','masks']
background_path='/home/tonyhuy/bottle_classification/data_bottle_detection/coco/images'
dataset= extract_mask(img_file_path,label_file_path,background_path)
data=[]
# Function to add data to the list
def add_data(idx_name, roi,region_X,region_Y,json_name):
    data.append({"idx_name": idx_name,"json_path":json_name, "roi": roi,"X_region":region_X,"Y_region":region_Y})
for i,(image,mask,idx_name,roi,region_X,region_Y,json_name) in enumerate(dataset):
    add_data(idx_name, roi,region_X,region_Y,json_name)
    cv2.imwrite(os.path.join(save_dir,file_names[0],f'image_{idx_name}.jpg'),image)
    cv2.imwrite(os.path.join(save_dir,file_names[1],f'mask_{idx_name}.jpg'),mask)
    # cv2.imwrite(f'{save_dir}/{file_names[0]}/img_{i}.jpg',image)
    # cv2.imwrite(f'{save_dir}/{file_names[1]}/mask_{i}.jpg',mask)
    # if i==len(dataset)-1:
    #     break
# Create a pandas DataFrame
df = pd.DataFrame(data)
# Specify the name of the CSV file
csv_file = "/home/tonyhuy/Yolov7_box_segmentation/result_mask/output_with_idx_and_roi.csv"

# Save the DataFrame to the CSV file
df.to_csv(csv_file, index=False)