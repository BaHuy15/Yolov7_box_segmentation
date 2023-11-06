# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  path/                           # directory
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg.xml                # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import shutil
import yaml

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, letterbox
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, 
                           xyxy2xywh, xywhn2xyxy, xyn2xy)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode
from utils.convert_kla_format_multiclass import convert_format_test_data, convert_Glabeller_to_COCO
import numpy as np


import math
import pandas as pd
import itertools

def calculate_area(box):
    """
        Input:
            box - [x_min, y_min, x_max, y_max]
        Output:
            area = (x_max - x_min) * (y_max - y_min)
    """
    return (box[2] - box[0]) * (box[3] - box[1])

def get_process_info(im_shape, new_shape=(640, 640), auto=True, scaleFill=False, scaleup=True, stride=32):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / im_shape[0], new_shape[1] / im_shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(im_shape[1] * r)), int(round(im_shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / im_shape[1], new_shape[0] / im_shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    return ratio, (dw, dh)

def get_centroid_from_mask(mask):
    if mask.shape[-1] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key = cv2.contourArea)
    M = cv2.moments(contour)
    center_x = np.rint(M["m10"] / M["m00"]).astype(np.int32)
    center_y = np.rint(M["m01"] / M["m00"]).astype(np.int32)

    return center_x, center_y

def get_centroid_from_box(box):
    return np.int32([(box[2] + box[0])/2, (box[3] + box[1])/2])

def get_distance(x ,y):
    return math.sqrt((y[0]-x[0])**2 + (y[1]-x[1])**2)


def mask_iou(mask1: torch.Tensor,mask2: torch.Tensor):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return intersection / union

def binary_mask_to_box(mask):
    """
        Input:
            mask - tensor, shape=(H, W)
        Output:
            bbox - tensor, format=[x_min, y_min, x_max, y_max]
    """
    
    # Find the non-zero pixels
    try:
        coords = torch.nonzero(mask)

        # Get the minimum and maximum coordinates for the bounding box
        top_left = coords.min(0)[0]
        bottom_right = coords.max(0)[0]
        # Convert the coordinates to a box representation (top-left, bottom-right)
        box = torch.concat([top_left[::-1], bottom_right[::-1]], dim=0)
        return box
    except:
        return None

def calculate_iou(box1, box2):
    """
        Inputs:
            box1 - format = [x_min, y_min, x_max, y_max]
            box2 - format = [x_min, y_min, x_max, y_max]
        Output:
            iou = inter_area / union_area
    """
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3]) 
    
    inter_area = calculate_area((x_min, y_min, x_max, y_max))
    union_area = calculate_area(box1) + calculate_area(box2) - inter_area
    # Calculate the IoU
    iou = inter_area / union_area
    return iou

def expand_box(box, expansion):
    """
        Inputs:
            box - format = [x_min, y_min, x_max, y_max]
            expansion - value to expand the box
    """
    # Subtract the expansion value from the top-left coordinate
    top_left = box[:2] - expansion
    # Add the expansion value to the bottom-right coordinate
    bottom_right = box[2:] + expansion
    # Combine the top-left and bottom-right coordinates into a box
    expanded_box = torch.concat([top_left, bottom_right], dim=0)
    return expanded_box

def create_mask(x1, y1, x2, y2, im_shape):
    """
        Inputs:
            (x1, y1), (x2, y2) - (x_min, y_min), (x_max, y_max), int type
            im_shape - (H, W), shape of the image
        Output:
            mask - shape = (H, W), drawed a rectangle object
    """
    mask = np.zeros(im_shape, dtype=np.uint8)
    cv2.rectangle(mask, [x1, y1], [x2, y2], 255, -1)
    return mask

def masks_from_label(shape, scale, pad, txt, gt_is_box):
    """
        Inputs:
            shape - (H, W), current image shape
            scale - (S_W, S_H), origin image shape
            ratio - min(shape[0] / shape0[0], shape[1] / shape0[0])
            pad - (padw, padh), padding size
            txt - Path, text file
            gt_is_box - bool
        Outputs:
            boxes - 
            masks - np.array, shape = (num_gt, H, W)
            class_ids - np.array, shape = (num_gt,)
    """
    
    # read file
    contents = open(txt).read()
    if contents is '':
        return [], [], []
    labels = [content.strip().split(' ') for content in contents.strip().split('\n')]
    
    # compute scale and pad size
    scaled_w, scaled_h = scale
    padw, padh = pad
    
    class_ids = [label[0] for label in labels]
    label_points = [np.array(label[1:], dtype=float) for label in labels]
    masks = []
    boxes = []
    
    if gt_is_box:
        boxes = np.round(xywhn2xyxy(x=label_points, w=scaled_w, h=scaled_h, padw=padw, padh=padh))         
        masks = np.array([create_mask(x1, y1, x2, y2, shape) for (x1, y1, x2, y2) in np.int32(boxes)])
        print(masks.shape)
    else:
        for points in label_points:
            vertices = np.round(xyn2xy(x=points.reshape(-1, 2), w=scaled_w, h=scaled_h, padw=padw, padh=padh))
            mask_gt = np.zeros(shape, np.uint8)
            
            # Don't draw mask
            mask_gt = cv2.fillPoly(mask_gt, pts=[np.int32(vertices)], color=255)
            boxes.append(np.concatenate([vertices.min(axis=0), vertices.max(axis=0)]))
            masks.append(mask_gt)
            
        boxes = np.array(boxes)
        masks = np.array(masks)
        
    return boxes, masks, class_ids

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        source_label='',
        area_thres=9,
        expansion=4,
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'seg/runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference

        # source_label = None,
        UK_OK = False,
        finger = False, # plot finger from Long's code
        gt_is_box = False,
):
    
    # get class names of the new dataset
    with open(str(data), mode='r') as stream:
        class_names = yaml.safe_load(stream)['names']
    source = Path(source)
    
    check_coco_format = True
    try:
        
        # ======================================= PREPARE DATA ================================================
        # case 1: source is a directory, source_label is None
        # case 2: source is a image file, source_label is a label file
        # convert to coco format if it is in GLabbler format
        if source.is_dir():
            if 'images' not in os.listdir(source):
                # new structure:
                # test
                # |------- image files and their label files
                # |------- test_coco_format
                #          |------- images - image files
                #          |------- labels - label files with coco format
                check_coco_format = False
                dst_path = source / 'test_coco_format'
                if not dst_path.exists():
                    dst_path.mkdir()
                convert_format_test_data(source=str(source), dst=str(dst_path), class_names=class_names)
                source_label = source #/ 'test_coco_format' / 'labels'
                source = source #/ 'test_coco_format' / 'images'
            else:
                source_label = source / 'labels'
                source = source / 'images'
        else:
            # format source label to coco format
            source_label = Path(source_label)
            if source_label.suffix == '.json':
                check_coco_format = True
                new_label_path = str(source_label).split('.')[0] + '.txt'
                convert_Glabeller_to_COCO(new_label_path, source_label, class_names)
                source_label = Path(new_label_path)
        # =====================================================================================================
        
        # =================================== INITIALIZE PROJECT ==============================================
        project = Path(project)
        
        # check source label is file or not
        if source_label.suffix == '.txt':
            if source_label.stem != source.stem:
                raise 'label and img must have same name, different suffix'
            # declare source_label to dir parent
            source_label = source_label.parent

        source = str(source)
        print(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download
        
        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        # make subdirs
        # project_name / predict_UK_OK_seg
        # |------------ correct_images
        # |------------ incorrect_images
        (save_dir / 'correct_images').mkdir()
        (save_dir / 'incorrect_images').mkdir()
        
        # ==========================  
        
        # ================================= STATICS ================================================
        img_names = []
        UK_per_img = []
        OK_per_img = []
        dists_per_img = []
        
        total_finger = 0
        # ==========================================================================================

        # ================================== LOAD MODEL AND DATA LOADER =============================================
        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        # ==================================================================================================
        
        # ================================== RUN INFERENCE =================================================
        for path, im, im0s, vid_cap, s in dataset:
            # Transform image data
            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred, out = model(im, augment=augment, visualize=visualize)
                proto = out[1]

            # NMS
            with dt[2]:
                # format predictions: [batch_size, num_det, x_min, y_min, x_max, y_max, p_obj, class_id, ...]
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
    
            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
            
            # Process predictions
            ratio, pad = get_process_info(im_shape=im0s.shape[:2], new_shape=imgsz, auto=pt, stride=stride)
            scale = (im0s.shape[1] * ratio[0], im0s.shape[0] * ratio[1])
            
            # Process predictions on per image
            for i, det in enumerate(pred):
                # det format = [num_det, x_min, y_min, x_max, y_max, p_obj, class_id, ...] -----------------------------
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                # prepare path of image
                p = Path(p)
                save_path = str(save_dir / p.name)  # im.jpg

                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop

                # make Anotators: preds and gts
                annotator = Annotator(im0.copy(), line_width=line_thickness, example=str(names))
                gt_annotator = Annotator(im0.copy(), line_width=line_thickness, example=str(class_names))
                
                # Process detections: get masks and scale bounding boxes
                # masks = tensors if len(det) > 0 else []
                if len(det):
                    # masks's shape = (num_det, H, W), coords's shape = (num_det, x_min, y_min, x_max, y_max)
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    
                    boxes = det[:, :4].clone().round()
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Mask plotting ----------------------------------------------------------------------------------------
                    mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                    class_ids = [cls for cls in det[:, 5]]
                    #Predict masks gt : gt_mask
                    im_masks = plot_masks(im[i], masks, mcolors, show_mask = True)  # image with masks shape(imh,imw,3)
                    # annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                    # Mask plotting ----------------------------------------------------------------------------------------
                   
                   

                    # Write results
                    for *xyxy, conf, cls in reversed(det[:, :6]):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            if finger == True:
                                label=names[c]
                            else:
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True),finger=finger)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                else:
                    masks = []
                    boxes = []
                    
                # Calculate Under Kill and Over Kill -----------------------------------------------------------------
                if UK_OK:
                    txt = str(source_label / (p.stem + '.txt'))
                    
                    gt_boxes, gt_masks, gt_class_ids = masks_from_label(im.shape[2:], scale, pad, txt, gt_is_box)
                    gt_boxes = torch.from_numpy(gt_boxes).to(device) if not isinstance(gt_boxes, list) else gt_boxes
                    gt_masks = torch.from_numpy(gt_masks).to(device) if not isinstance(gt_masks, list) else gt_masks
                    total_finger += len(gt_masks)

                    UK = 0
                    OK = 0
                    dists = []
                    
                    len_masks = len(masks)
                    len_gt_masks = len(gt_masks)
                    if len_masks == 0 and len_gt_masks > 0:
                        UK = len_gt_masks
                    if len_masks > 0 and len_gt_masks == 0:
                        OK = len_masks
                    if len_masks > 0 and len_gt_masks > 0:                        
                        # get pairs (gt, pred) respectively
                        gt_id_per_mask = np.full(shape=(len_masks,), fill_value=-1, dtype=int)
                        for i in range(len_gt_masks):
                            gt_box = gt_boxes[i]
                            
                            if calculate_area(gt_box) < area_thres:
                                gt_box = expand_box(gt_box, expansion)
                            
                            gt_class_name = class_names[int(gt_class_ids[i])]

                            max_iou = 0
                            mask_id = -1
                            for j in range(len_masks):
                                
                                # if it had a gt, passing it
                                if gt_id_per_mask[j] != -1:
                                    continue
                                
                                predict_box = binary_mask_to_box(masks[j])
                                if predict_box is None:
                                    predict_box = boxes[j]
                                if calculate_area(predict_box) < area_thres:
                                    predict_box = expand_box(predict_box, expansion)
                                
                                # If object is finger, we have to expand the bbox
                                if finger:
                                    predict_box = expand_box(predict_box, 18)
                                
                                iou = calculate_iou(gt_box, predict_box)

                                if iou > max_iou and (gt_class_name == names[int(class_ids[j].item())]):
                                    max_iou = iou
                                    mask_id = j
                            
                            if mask_id != -1:
                                gt_id_per_mask[mask_id] = i

                        # compute Under Kill and Over Kill
                        n_pairs = sum(gt_id_per_mask != -1)
                        UK = len_gt_masks - n_pairs
                        OK = len_masks - n_pairs
                        
                        # compute distance of each pair:
                        for mask_id, gt_id in enumerate(gt_id_per_mask):
                            if gt_id == -1:
                                continue
                            
                            pred_mask = masks[mask_id]
                            check_mask = torch.sum(pred_mask).item() > 0
                            pred_centroid = get_centroid_from_mask(np.uint8(255 * pred_mask.to('cpu').numpy())) if check_mask else get_centroid_from_box(boxes[mask_id].to('cpu'))
                            
                            gt_mask = gt_masks[gt_id]
                            check_gt_mask = torch.sum(gt_mask).item() > 0
                            gt_centroid = get_centroid_from_mask(gt_mask.to('cpu').numpy()) if check_gt_mask else get_centroid_from_box(gt_boxes[gt_id].to('cpu'))
                            
                            dists.append(get_distance(pred_centroid, gt_centroid))
                   
                    
                    # update statics
                    img_names.append(p.stem)
                    UK_per_img.append(UK)
                    OK_per_img.append(OK)
                    dists_per_img.append(dists)
                    
                    # get save path
                    save_path = (save_dir / os.path.join('incorrect_images' if UK > 0 or OK > 0 else 'correct_images', p.name))
                    # print(f'finger ne {finger}')
                    # Ground truth Mask plotting----------------------------------------------------------------------------
                    if len_gt_masks > 0:
                        mcolors = [colors(int(cls), True) for cls in gt_class_ids]
                        #______________________ Scale mask [0-1]________________________#
                        gt_masks=gt_masks.type(torch.cuda.FloatTensor)
                        gt_masks=torch.div(gt_masks,255)
                        # print(f'ground truth mask *****{gt_masks} with shape {gt_masks.shape}')
                        #_______________________________________________________________#
                        im_masks = plot_masks(im[0], gt_masks, mcolors, show_mask = True)
                        # print(f'ground truth mask {gt_masks} and shape {gt_masks.shape} and unique values {np.unique(gt_masks.detach().cpu().numpy())}')
                        # cv2.imwrite('ground_truth.jpg',im_masks)                    
                        # gt_annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                        gt_boxes = scale_coords(im.shape[2:], gt_boxes, im0.shape)
                        for c, gt_box in zip(gt_class_ids, gt_boxes):
                            label = class_names[int(c)]
                            gt_annotator.box_label(gt_box, label, color=colors(c, True),finger=finger)
                    
                    # Draw UK and OK result
                    color = (0, 0, 255) if (UK + OK) == 0 else (0, 0, 255)  #(0, 255, 0)
                    cv2.putText(img=gt_annotator.im, 
                                text=f'UK: {UK}   OK: {OK}', 
                                org=(500,45),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale=1.5,
                                color=color, thickness=2, lineType=cv2.LINE_AA)     
                    # -------------------------------------------------------------------------------------------------------
                # -----------------------------------------------------------------------------------------------------------
                
                # Stream results --------------------------------------------------------------------------------------------
                im0 = np.concatenate([gt_annotator.result(), annotator.result()])
                #gt_annotator.result()--- gray_img

                # im0=np.concatenate([annotator.result(),im0])

                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        
        # ======================================= POOLING RESULTS ====================================================
        if UK_OK == True:
            total_UK = np.sum(UK_per_img)
            total_OK = np.sum(OK_per_img)
            
            dists = list(itertools.chain(*dists_per_img))
            mean_dists = np.mean(dists)
            std_dists = np.std(dists, ddof=1)
            
            print("=============================================================")
            print(f" ***    Underkills: {total_UK}, Overkills: {total_OK}, Average distance: {mean_dists}  *** ")
            print(f"Total: {total_finger}")
            print("=============================================================")
            
            # Save results
            results = pd.DataFrame({'image_name': img_names, 'UK': UK_per_img, 'OK': OK_per_img, 'dists': dists_per_img})
            results = results.combine_first(pd.DataFrame({'image_name': [''], 'UK': [total_UK], 'OK': [total_OK], 'dists': [(mean_dists, std_dists)]}, index=['statics']))
            results.to_csv(save_dir / 'results.csv')
        # ============================================================================================================
            
        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"==========* Results saved to {colorstr('bold', save_dir)}{s} *============")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    except:
        raise
    finally:
            if not check_coco_format:
                if os.path.isdir(str(source)):
                    shutil.rmtree(path=dst_path, ignore_errors=True)
                else:
                    os.remove(path=source_label)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source-label', type=str, default='', help='file/dir/URL/glob, 0 for webcam')
    ''''''
    parser.add_argument('--UK-OK', action='store_true', help='get Underkill and Overkill')
    parser.add_argument('--finger', action='store_true', help='Plot finger from Long\'s code')
    parser.add_argument('--area-thres', type=int, default=9, help='confidence threshold')
    parser.add_argument('--expansion', type=int, default=4, help='confidence threshold')
    ''''''
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--gt-is-box', action='store_true' , help='is your groundtruth the boxes')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

