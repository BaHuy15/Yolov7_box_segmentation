import os
import shutil
import sys
import glob
import json
import yaml
import time
# import argparse

# support function
def convert_Glabeller_to_COCO(file_path, json_path, class_names):
    coco_format_data = ""
    
    if os.path.exists(json_path):
        # load json file and get data
        with open(json_path, 'r') as label_file:
            label_datas = json.load(label_file)
                
        # get data and convert to COCO format
        img_width, img_height = float(label_datas['width']), float(label_datas['height'])
        
        polygon_data = label_datas["regions"]
        for label_index, data in polygon_data.items():
            label_index = int(label_index)
            label_class = label_datas['classId'][label_index]
            if label_class in class_names:
                index = 0
                for i, class_name in enumerate(class_names):
                    if label_class == class_name:
                        index = i
                        break
                coco_format_data += str(index) + ' '
            else:
                print("Label name is not in class_names. Please check class_names and labels")
                sys.exit()

            # coco_format_data += "0" + " "
            poly_len = len(data["List_X"])
            for i in range(poly_len):
                point_X = float(data["List_X"][i])
                point_Y = float(data["List_Y"][i])
                coco_format_data += str(point_X/img_width) + " " + str(point_Y/img_height) + " "
                
            coco_format_data += "\n"
            
    with open(file_path, 'w') as f:
        f.write(coco_format_data)

def convert_format_train_data(source, dst, class_names):
    # source = opt['data_path']
    # assert os.path.exists(source), f'{source} doesn\'t not exist!!!'
    
    # dst = opt['save_dir']
    # assert os.path.exists(dst), f'{dst} doesn\'t not exist!!!'
    
    # assert os.path.exists(opt[class_file), f'{opt.class_file} doesn\'t not exist!!!'
    # with open(opt.class_file, mode='r') as file:
    #     class_names = yaml.safe_load(file)['names']
    # class_names = opt['class_names']
    
    # get subdirectories in source
    print('Converting data...')
    t0 = time.time()
    subfol_list = [name for name in os.listdir(source)
                if os.path.isdir(os.path.join(source, name)) and (name in ['train', 'val'])]
    print(subfol_list)

    for subfol in subfol_list:
        source_set_path = os.path.join(source, subfol)
        target_images_path = os.path.join(*[dst, "images", subfol])
        target_labels_path = os.path.join(*[dst, "labels", subfol])
        
        # check if target directories exist and create them if not
        if not os.path.exists(target_images_path):
            os.makedirs(target_images_path)
        if not os.path.exists(target_labels_path):
            os.makedirs(target_labels_path)
        
        # get all images in source_set_path
        labels_list_path = []
        images_list_path = glob.glob(os.path.join(source_set_path, "*.bmp")) + glob.glob(os.path.join(source_set_path, "*.jpg"))\
                           + glob.glob(os.path.join(source_set_path, "*.png"))
        for path in images_list_path:
            path = path + ".json"
            labels_list_path.append(path)

        # labels_list_path = [path + ".json" for path in images_list_path]
        
        # copy images to target directories
        for image_path, label_path in zip(images_list_path, labels_list_path):
            shutil.copy(image_path, target_images_path)
            new_label_path = os.path.join(target_labels_path, os.path.basename(image_path).split('.')[0] + ".txt")
            convert_Glabeller_to_COCO(new_label_path, label_path, class_names)


    print(f"Convert done!!! total_time={round(time.time() - t0, 2)}s")
    print()
    
def convert_format_test_data(source, dst, class_names):
    print('Converting data...')
    t0 = time.time()

    target_images_path = os.path.join(*[dst, "images"])
    target_labels_path = os.path.join(*[dst, "labels"])
    
    # check if target directories exist and create them if not
    if not os.path.exists(target_images_path):
        os.makedirs(target_images_path)
    if not os.path.exists(target_labels_path):
        os.makedirs(target_labels_path)
    
    # get all images in source_set_path
    labels_list_path = []
    images_list_path = glob.glob(os.path.join(source, "*.bmp")) + glob.glob(os.path.join(source, "*.jpg"))
    for path in images_list_path:
        path = path + ".json"
        labels_list_path.append(path)

    # labels_list_path = [path + ".json" for path in images_list_path]
    
    # copy images to target directories
    for image_path, label_path in zip(images_list_path, labels_list_path):
        shutil.copy(image_path, target_images_path)
        new_label_path = os.path.join(target_labels_path, os.path.basename(image_path).split('.')[0] + ".txt")
        convert_Glabeller_to_COCO(new_label_path, label_path, class_names)

    print(f"Convert done!!! total_time={round(time.time() - t0, 2)}s")
    print()
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-path', type=str, default='', help='Source data path')
#     parser.add_argument('--save-dir', type=str, default='', help='Destination data path')
#     parser.add_argument('--class-file', type=str, default='', help='.yaml file contains class names')
#     return parser.parse_args()

# if __name__ == '__main__':
#     opt = parse_opt()
    
#     run(opt)