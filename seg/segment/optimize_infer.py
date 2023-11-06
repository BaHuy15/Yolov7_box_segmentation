import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks,plot_contours,plot_only_masks
from utils.torch_utils import select_device, smart_inference_mode
import yaml
import numpy as np
import glob
from utils.augmentations import letterbox
# Input
#==========================Load image in image folder(json+image)=======================#
def load_image_source(path):
    '''
    input(str): path to image files
    Example: path='/home/tonyhuy/bottle_classification/data_bottle_detection/home/pevis/TOMO_detection/data_bottle_detection/test'
    output(list) list of image path
    Example :
    ['/home/tonyhuy/bottle_classification/data_bottle_detection/home/pevis/TOMO_detection/data_bottle_detection/test/9220.png',
    '/home/tonyhuy/bottle_classification/data_bottle_detection/home/pevis/TOMO_detection/data_bottle_detection/test/9221.png'
    ]
    '''
    files = []
    for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
        p = str(Path(p).resolve())
        if '*' in p:
            files.extend(sorted(glob.glob(p, recursive=True)))  # glob
        elif os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
        elif os.path.isfile(p):
            files.append(p)  # files
        else:
            raise FileNotFoundError(f'{p} does not exist')
    return files,p


#==========================Load config model=======================#

def model_params(device,weights,imgsz,dnn,data,half):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    return model,imgsz,stride,names,pt,device




class Load_image:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None):
        files,p=load_image_source(path)
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        # Read image
        self.count += 1
        im0 = cv2.imread(path)  # BGR
        assert im0 is not None, f'Image Not Found {path}'
        s = f'image {self.count}/{self.nf} {path}: '

        if self.transforms:
            im = self.transforms(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/tonyhuy/bottle_classification/seg/runs/train-seg/exp7/weights/epoch_144_best.pt' , help='model path(s)')
    parser.add_argument('--source', type=str, default='/home/tonyhuy/bottle_classification/data_bottle_detection/home/pevis/TOMO_detection/data_bottle_detection/test', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='/home/tonyhuy/bottle_classification/seg/data/coco.yaml', help='(optional) dataset.yaml path')
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
    parser.add_argument('--finger', action='store_true', help='Plot finger from Long\'s code')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt



class Configuration:
    def __init__(self,opt):
        self.weights=opt.weights
        self.data=opt.data
        self.device=opt.device
        
        self.imgsz=opt.imgsz
        self.dnn=opt.dnn
        self.half=opt.half
        self.source=opt.source
        self.model,self.imgsz,self.stride,self.names,self.pt,self.devices=self.init_model()[0],\
                                       
        self.visualize=False
        self.bs=1
        self.augment=opt.augment
        self.conf_thres, self.iou_thres,self.max_det, self.classes, self.agnostic_nms=opt.conf_thres,\
                                                                                    opt.iou_thres,\
                                                                                    opt.max_det,\
                                                                                    opt.classes,\
                                                                                    opt.agnostic_nms\
        # Directories
        self.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (self.save_dir / 'labels' if opt.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
    def init_model(self):
        model,imgsz,stride,names,pt,device = model_params(self.device,self.weights,self.imgsz,self.dnn,self.data,self.half)
        return model,imgsz,stride,names,pt,device

    def init_dataset(self):
        dataset = Load_image(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        return dataset


    def run_inference(self):
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        dataset=self.init_dataset()
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))
        for path, im, im0s, vid_cap, s in dataset:
            im_origin = im
            with dt[0]:
                im = torch.from_numpy(im).to(self.devices)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            # Inference
            with dt[1]:
                # visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                pred, out = self.model(im, augment=self.augment) #visualize=visualize
                proto = out[1]
             # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)
        return pred




    
def init_project():
    #==============================Configuration for model and data======================#
    dnn=False
    half=False
    source='/home/tonyhuy/bottle_classification/data_bottle_detection/home/pevis/TOMO_detection/data_bottle_detection/test'
    weights='/home/tonyhuy/bottle_classification/seg/runs/train-seg/exp7/weights/epoch_144_best.pt' 
    data='/home/tonyhuy/bottle_classification/seg/data/coco.yaml'
    device='3'
    imgsz=640
    #==================================Init model and their parameters=====================#
    model,imgsz,stride,names,pt = model_params(device,weights,imgsz,dnn,data,half)
    #==================================Init dataset===================================#
    dataset = Load_image(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())



def main():
    opt = parse_opt()
    init=Configuration(opt)
    pred=init.run_inference()
    print(pred.shape)





if __name__ == "__main__":
    main()
    





