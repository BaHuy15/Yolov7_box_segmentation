import cv2
import numpy as np
import random
import os
import glob
import random

def replace_background(image, mask, background_path):
    background = cv2.imread(background_path)
    # if background > image ==> crop
    if background.shape[0] > image.shape[0] and background.shape[1] > image.shape[1]:
        x = random.randint(0, background.shape[1] - image.shape[1] -1)
        y = random.randint(0, background.shape[0] - image.shape[0] -1)
        background = background[y:y + image.shape[0], x: x + image.shape[1]]
    else:
        background = cv2.resize(background, (image.shape[1], image.shape[0]))
    # Create a binary mask for the object(s)
    mask[mask>0] = 255
    object_mask = mask
    # Ensure the mask is binary (0 and 255 values)
    # object_mask = cv2.threshold(object_mask, 1, 255, cv2.THRESH_BINARY)[1]

    # Use cv2.bitwise_and to extract the object region
    object_region = cv2.bitwise_and(image, image, mask=object_mask)
    # Invert the binary mask to get the background mask
    background_mask = cv2.bitwise_not(object_mask)
    # Extract the object(s) from the image
    # object_region = cv2.bitwise_or(image, image, mask=object_mask)
    # Extract the background region from the background image
    background_region = cv2.bitwise_and(background, background, mask=background_mask)
    #Test 
    result = cv2.add(object_region, background_region)
    return result, object_mask

# Example usage:
image_folder = '/home/tonyhuy/Yolov7_box_segmentation/result_mask/images'
mask_folder = '/home/tonyhuy/Yolov7_box_segmentation/result_mask/masks'  # binary mask folder
background_folder = '/home/tonyhuy/bottle_classification/data_bottle_detection/coco/images'
save_dir='/home/tonyhuy/Yolov7_box_segmentation/result_mask'


for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    mask_path = os.path.join(mask_folder, image_name.replace('image','mask'))
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    background_path = random.choice(glob.glob(os.path.join(background_folder, "*")))
    result, object_mask = replace_background(image, mask, background_path)
    cv2.imwrite(os.path.join(save_dir, 'change_bg/images',"changed_background_" + image_name), result)
    cv2.imwrite(os.path.join(save_dir, 'change_bg/masks',"changed_background_" + image_name.replace("image","mask")), object_mask)
