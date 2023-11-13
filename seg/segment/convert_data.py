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
        # print(image_path)
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask >0] = 255
        
        image_height, image_width, _ = image.shape
        # label_file = os.path.join(output_folder, image_file.replace('.jpg', '.txt'))
        demo_label_file=os.path.join('/home/tonyhuy/Yolov7_box_segmentation/result_mask/demo',image_file.replace('.jpg', '.txt'))
        with open(demo_label_file, 'w') as label_f:
            # calculate
            contours, _ = cv2.findContours((mask == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Create an empty list to store the results
            # results = []
            # Iterate through each contour and get x1y1,x2y2,x3y3...
            # https://github.com/muhemuhe427/jsonlabel2yolo/blob/main/json2yolo
            # https://github.com/ultralytics/yolov5/issues/9816
            label_pts = [0]
            contour = contours[0]
            for contour in contours:
                for t in contour:
                    label_pts.append(t[0][0]/image_width)
                    label_pts.append(t[0][1]/image_height)
            for t in label_pts:
                label_f.write(str(t))
                label_f.write(' ')
                # Find bounding box coordinates
            # x, y, w, h = cv2.boundingRect(contour)
            # # Normalize coordinates
            # center_x = (x + w / 2) / image_width
            # center_y = (y + h / 2) / image_height
            # width = w / image_width
            # height = h / image_height
            # box_coordinate=[0, center_x, center_y, width, height]
            # while len(box_coordinate) < 8:
            #     box_coordinate.append(0)
            # results.append(box_coordinate)
            #     # label_f.write(line + '\n')
            # for result in results:
            #     line = ' '.join(map(str, result))
            #     label_f.write(line + '\n')

                # label_f.write(line + '\n')
                # file.write(line + '\n')
            # if len(contours) > 0:
            #     contour = contours[0]
            #     x, y, w, h = cv2.boundingRect(contour)
            #     x_center = (x + w / 2) / image_width
            #     y_center = (y + h / 2) / image_height
            #     width = w / image_width
            #     height = h / image_height
            
                # write information to file
                # label_f.write(f"{0} {x_center} {y_center} {width} {height}\n")

images_folder = '/home/tonyhuy/Yolov7_box_segmentation/result_mask/change_bg/images'
masks_folder = '/home/tonyhuy/Yolov7_box_segmentation/result_mask/change_bg/masks'
output_folder = '/home/tonyhuy/Yolov7_box_segmentation/result_mask/convert2yolo'

convert_to_yolo_data(images_folder, masks_folder, output_folder)