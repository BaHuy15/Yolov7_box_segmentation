# Yolov7_box_segmentation
# Detect box location using YOLOv7

In this project, we applied yolov7 to localize bounding box of boxes and segment them. To illustrate this, see figure below.
<div align="center">
    <a href="./">
        <img alt="figure 1: Detect box location and segment them" src="./Figure/image.png" width="100%"/>
    </a>

</div>

<div align="center">
    <a href="./">
        <img alt="figure 2: Original image"src="./Figure/image_fig.png" width="100%"/>
    </a>

</div>


## Table of Contents

- [Table of Contents](#table-of-contents)
- [Install Requirements](#Install-Requirements)
- [Create and activate virtual environment](#Create-and-activate-virtual-environment)
- [Data Format](#Data-Format)
- [Generate auto-augmented data](#Generate-augmentation-data)
- [Download Yolov7 Weights](#Download-Yolov7-Weights)
- [Evaluation](#Evaluation )
- [Training](#Training)
- [Result](#Result)
- [Citation](#Citation)
- [Acknowledgement](#Acknowledgements)

## Install Requirements
------------------------ --------------------
<details><summary> <b>Expand</b> </summary>                  
                                                                  
| Package | Version  |  
| :-- | :-: |
|absl-py|**1.4.0**|   
|albumentations |**1.3.0**|
|appdirs |  **1.4.4** |               
|attrs |  **22.2.0**  |
| backcall| **0.2.0**  |                 
| cachetools|  **5.3.0**  |               
| certifi| **2022.12.7**|                   
| chardet| **4.0.0** |
|charset-normalizer| **3.1.0** | 
|clearml| **1.10.3** |
| click| **8.1.3**|
| clip | **0.2.0**   |                                      
| cycler|**0.11.0** |                             
| decorator|**5.1.1**|            
| docker-pycreds|**0.4.0**  |                      
| fonttools| **4.38.0** |                              
| furl |**2.1.3**   |        
| gitdb | **4.0.10** |            
| GitPython |**3.1.31** |             
| google-auth| **2.17.2**|              
| google-auth-oauthlib| **0.4.6** |               
| grpcio | **1.53.0**  |               
| idna   | **3.4** |                                       
| imageio | **2.27.0** |                 
| imgaug | **0.4.0** |                         
| importlib-metadata| **6.3.0** |                      
| importlib-resources| **5.12.0** |                       
| ipython| **7.34.0**|          
| jedi | **0.18.2**  |                     
| joblib | **1.2.0**  |                
| jsonschema| **4.17.3**|                
| kiwisolver| **1.4.4** |                    
| Markdown | **3.4.3** |             
| MarkupSafe| **2.1.2** |                                     
| matplotlib | **3.5.3** |                   
| matplotlib-inline| **0.1.6** |                      
| networkx | **2.6.3** |                        
| numpy | **1.21.6**|                         
| nvidia-cublas-cu11 | **11.10.3.66**|                                   
| nvidia-cuda-nvrtc-cu11| **11.7.99** |                                                      
| nvidia-cuda-runtime-cu11| **11.7.99** |                                       
| nvidia-cudnn-cu11| **8.5.0.96** |                             
| oauthlib| **3.2.2** |                                            
| opencv-python |**4.7.0.72** |                            
| opencv-python-headless| **4.7.0.72** | 
| orderedmultidict| **1.0.1** |                     
| packaging | **23.0** |                    
| pandas |**1.3.5**|                  
| parso |**0.8.3**|                                        
| pathlib2 |**2.3.7.post1**|                        
| pathtools|**0.1.2** |     
| pexpect |**4.8.0**|                              
| pickleshare |**0.7.5**|           
| Pillow|**9.5.0**|                  
| pip|**23.0.1** |                
| pkgutil_resolve_name| **1.3.10**|                  
| prompt-toolkit|**3.0.38**|                                                     
| protobuf|**3.20.1**|                         
| psutil|**5.9.4**|                            
| ptyprocess|**0.7.0**|                                
| pyasn1|**0.4.8**|                 
| pyasn1-modules|**0.2.8** |           
| pycocotools|**2.0.6** |                
| Pygments|**2.14.0**|               
| PyJWT |**2.4.0**|               
| pyparsing|**3.0.9**|               
| pyrsistent|**0.19.3**|             
| python-dateutil|**2.8.2** |         
| python-dotenv|**0.21.1**|           
| pytz |**2023.3**|               
| PyWavelets|**1.3.0**|                              
| PyYAML|**6.0** |                
| qudida|**0.0.4**|            
| requests|**2.28.2**|              
| requests-oauthlib|**1.3.1**|           
| requests-toolbelt|**0.10.1**|          
| roboflow|**1.0.3**|       
| rsa|**4.9**|                       
| scikit-image |**0.19.3**|                                                      
| scikit-learn |**1.0.2**|                        
| scipy|**1.7.3**|                    
| seaborn|**0.12.2**|                
| sentry-sdk|**1.19.1**|                  
| setproctitle|**1.3.2**|                                                                          
| setuptools|**47.1.0**|                
| shapely|**2.0.1** |               
| six|**1.16.0**|                
| smmap|**5.0.0**|              
| tensorboard|**2.11.2**|             
| tensorboard-data-server|**0.6.1** |          
| tensorboard-plugin-wit|**1.8.1** |           
| thop |**0.1.1.post2209072238**|                                        
| threadpoolctl |**3.1.0**|                              
| tifffile |**2021.11.2** |                        
| torch  |**1.10.1+cu102** |                              
| torch-tb-profiler|**0.4.1** |                                                   
| torchaudio |**0.10.1+cu102** |                                           
| torchvision |**0.11.2+cu102**|                               
| tqdm   |**4.65.0** |                                 
| traitlets|**5.9.0**|                                       
| typing_extensions |**4.5.0**|                        
| ultralytics |**8.0.110**|                              
| urllib3 |**1.26.15** |                         
| wandb|**0.14.2** |                                                       
| wcwidth |**0.2.6**|                                      
| Werkzeug |**2.2.3** |                                        
| wget  |**3.2**  |                   
| wheel |**0.40.0**|                        
| zipp |**3.15.0**| 
   
</details>  

## Create virtual environment
<details><summary> <b>Expand</b> </summary> 

``` shell 

# Create virtual environment
python3 -m venv venv_yolo                   

# Activate virtual environment
source path/to/ven_yolo/bin/activate      

``` 

</details> 

## Data Format
<details><summary> <b>Expand</b> </summary> 

``` shell 
Yolov7_box_segmentation
    |                            
    |
    |____Box_data                                                                             
    |        |
    |        |________Filling_2023_05_16                                       
    |                       |
    |                       |______Test                             
    |                       |       |________.png/.jpg
    |                       |
    |                       |
    |                       |______Train                      
    |                       |        |                                        
    |                       |        |_________.bmg
    |                       |        |
    |                       |        |_________.json 
    |                       |                      
    |                       |_______Val   # Test image                           
    |                                |                                        
    |                                |_________.bmg
    |                                |
    |                                |_________.json                     
    |             
    |
    |______command                    
    |        |                                        
    |        |_________predict_UK_OK.sh
    |        |
    |        |_________predict.sh              
    |        |
    |        |_________train.sh 
    |
    |______seg
            |
            |_____data
            |       |
            |       |_______hyps       
            |       |        |_______blister.yaml                          
            |       |        |_______coco.yaml                      
            |       |        |_______hyp.scratch.custom.yaml                         
            |       |        |_______hyp.scratch.p5.yaml                        
            |       |
            |       |_______scripts   
            |       |        |    
            |       |        |_______get_coco.sh 
            |       |        |                       
            |       |        |_______get_imagenet.sh
            |       |                                             
            |       |
            |       |_______coco.yaml               
            |                   
            |
            |______models                                         
            |
            |______runs                 
            |
            |______segment                       
            |
            |______tools             
            |
            |______utils                       
            |
            |______wandb                                 
            |       
            |______export.py # run inference file                                     
            |
            |______hubconf.py                    
            |
            |______requirements.txt # Evaluate on new data                     
            |
            |______yolov5m-seg.pt   # Test                   
            |
            |______yolov5s-seg.pt # Train         


```                                             

</details>  

## Add segment data to background

<details><summary> <b>Expand</b> </summary> 

``` shell 

# Convert polygon read from json file to mask png/jpg file                             
python3 cvtpoly2_mask.py # (this code will save mask in folder ./result_mask) (1)                 
           
# Read background image and mask.jpg created from script (1) to replace background                
python3 replace_background.py # (this script will replace image with background) (2)                     
                            
# Read background image and mask.jpg created from script (1) to replace background
python3 convert_data.py # (this script will convert images, label, masks to coco format to train) (3)                               

``` 
</details>

## Download Yolov7 Weights                                                        
[`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) [`yolov7x.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) [`yolov7-w6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) [`yolov7-e6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) [`yolov7-d6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) [`yolov7-e6e.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)           

## Evaluation   
``` shell
# python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
#If inference grayscale
python inference_blisters.py --gray_scale --weights /home/tonyhuy/yolov7/runs/train/yolov749/weights/last.pt --image_size 640  --device '2' --path /home/tonyhuy/yolov7_blister/blister_data/crop_img --conf_thresh 0.5 --batch_size 60  --class_name /home/tonyhuy/yolov7_blister/data/blister.yaml
#If you use inference_blisters.py file to infer video, set --batch_size 1 and add argument --run_video

#Detect code
python detect.py --weights /home/tonyhuy/yolov7/runs/train/yolov725/weights/last.pt --device 1 --conf 0.3 --iou-thres 0.4 --img-size 320 --source /home/tonyhuy/yolov7/blister_data/images/test

#Run prediction for augmented data(grayscale image)
python yolov7_pred.py --weights /home/tonyhuy/yolov7/runs/train/yolov749/weights/last.pt --gray_scale --device 1 --conf 0.3 --iou-thres 0.4 --img-size 640 --source /home/tonyhuy/yolov7_blister/augmented_data

#Run prediction for augmented data(RGB image)
python yolov7_pred.py --weights /home/tonyhuy/yolov7/runs/train/yolov749/weights/last.pt --device 1 --conf 0.3 --iou-thres 0.4 --img-size 640 --source /home/tonyhuy/yolov7_blister/augmented_data

```
After running inference_blister.py,the results will be saved here:

<div align="center">
    <a href="./">
        <img alt="figure 1" src="./figure/result.jpg" width="70%"/>
    </a>
</div>

## Generate augmentation data
Run this script to generate augmented images.
``` shell
# If save augmented data,use this command
python hand_augment.py  --path /home/tonyhuy/yolov7_blister/blister_data/images/test/ --save --save_dir /home/tonyhuy/yolov7_blister/augmented_data --augment_yaml data/augment_list.yaml 


```

## Training

``` shell
#____________________________________________ Run training file___________________________________________________#
# Image size 320
python train.py --workers 8 --device 3  --batch-size 8 --data data/blister.yaml --img 320  --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# Image size 640
python train.py --workers 8 --device 1  --batch-size 8 --data data/blister.yaml --img 640  --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7 --hyp data/hyp.scratch.p5.yaml
```

## Result                                                      
[Weights and Bias running result](https://wandb.ai/huynguyen15/YOLOR/runs/6cj3l4xu?workspace=user-huynguyen15)  

<div align="center">
    <a href="./">
        <img src="./figure/img.png" width="100%"/>
    </a>
</div>

## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

```
@article{wang2022designing,
  title={Designing Network Design Strategies Through Gradient Path Analysis},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Yeh, I-Hau},
  journal={arXiv preprint arXiv:2211.04800},
  year={2022}
}
```
## Web Demo

<!-- - Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces/akhaliq/yolov7) using Gradio. Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov7-trainable-bag-of-freebies-sets-new/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=yolov7-trainable-bag-of-freebies-sets-new)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)
<a href="https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)  -->

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>

