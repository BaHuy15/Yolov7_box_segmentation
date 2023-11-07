import os
import cv2
import numpy as np

def convert_to_yolo_data(images_folder, masks_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = os.listdir(images_folder)
    
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        mask_path = os.path.join(masks_folder, image_file.replace('image', 'mask'))  # Điều chỉnh phần mở rộng tệp mask

        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask >0] = 255
        
        image_height, image_width, _ = image.shape
        
        label_file = os.path.join(output_folder, image_file.replace('.jpg', '.txt'))
        with open(label_file, 'w') as label_f:
            # calculate
            contours, _ = cv2.findContours((mask == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contour = contours[0]
                
                x, y, w, h = cv2.boundingRect(contour)
                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                width = w / image_width
                height = h / image_height
            
                # write information to file
                label_f.write(f"{0} {x_center} {y_center} {width} {height}\n")

images_folder = '/home/tonyhuy/Yolov7_box_segmentation/result_mask/change_bg/images'
masks_folder = '/home/tonyhuy/Yolov7_box_segmentation/result_mask/change_bg/masks'
output_folder = '/home/tonyhuy/Yolov7_box_segmentation/result_mask/convert2yolo'

convert_to_yolo_data(images_folder, masks_folder, output_folder)