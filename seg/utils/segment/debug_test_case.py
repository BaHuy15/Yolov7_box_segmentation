# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders
"""

import os
import random

import cv2
import numpy as np
import glob
import torch
from torch.utils.data import DataLoader, distributed
import random

from ..augmentations import augment_hsv, copy_paste, letterbox
from ..aug_dataloaders import InfiniteDataLoader, LoadImagesAndLabels, seed_worker
from ..general import LOGGER, xyn2xy, xywhn2xyxy, xyxy2xywhn
from ..torch_utils import torch_distributed_zero_first
from .augmentations import mixup, random_perspective,custom_random_perspective


def create_dataloader(path,
                      imgsz,second_model,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False,
                      mask_downsample_ratio=1,
                      overlap_mask=False):
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabelsAndMasks(
            path,second_model,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            downsample_ratio=mask_downsample_ratio,overlap=overlap_mask)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    # generator = torch.Generator()
    # generator.manual_seed(0)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabelsAndMasks.collate_fn4 if quad else LoadImagesAndLabelsAndMasks.collate_fn,
        worker_init_fn=seed_worker,
        # generator=generator,
    ), dataset
    


class LoadImagesAndLabelsAndMasks(LoadImagesAndLabels):  # for training/testing

    def __init__(
        self,
        path,second_model,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0,
        prefix="",
        downsample_ratio=1,
        overlap=False,
    ):
        super().__init__(path, second_model,img_size, batch_size, augment, hyp, rect, image_weights, cache_images, single_cls,
                         stride, pad, prefix)
        self.downsample_ratio = downsample_ratio
        self.overlap = overlap
        self.bg_file= '/home/tonyhuy/bottle_classification/data_bottle_detection/coco/images'
        self.back_ground=os.listdir('/home/tonyhuy/bottle_classification/data_bottle_detection/coco/images')
        self.back_ground_img_list= [os.path.join(self.bg_file,x) for x in self.back_ground]
        self.second_model= second_model
    def __getitem__(self, index):
        global bg_id
        bg_id=index
        index = self.indices[index]  # linear, shuffled, or image_weights 
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        masks = []
        if mosaic:
            # Load mosaic
            # image:(640,640,3)            0            1          2          3
            # labels: (n,5):    class     x1          y1          x2          y2
            #                [ [  3      291.46      216.42      304.21       234.4]
            #                  [  0      206.76      108.53      221.85      121.87]
            #                  [  3      46.671      344.03      61.172      366.07]
            #                  [  1      228.23      549.96      260.13      573.15]]
            # Segments: (n,1000,2)
            img, labels, segments = self.load_mosaic(index) #[[x1,y1]
                                                            # [x2,y2]
                                                            # [x3,y3]
                                                            # ...
                                                            # [xn,yn]  ]
            # MixUp augmentation
            # if random.random() < hyp["mixup"]:
            #     img, labels, segments = mixup(img, labels, segments, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image and resize
            img, (h0, w0), (h, w)= self.load_image(index)
            # Letterbox
            shape=self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            # shape = [384, 672]

            # print(f'Con day la dau vao: {img.shape} và shape {shape}')

            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            # [array, array, ....], array.shape=(num_points, 2), xyxyxyxy
            segments = self.segments[index].copy()
            if len(segments):
                for i_s in range(len(segments)):
                    segments[i_s] = xyn2xy(
                        segments[i_s],
                        ratio[0] * w,
                        ratio[1] * h,
                        padw=pad[0],
                        padh=pad[1],
                    )
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
            # segments=np.concatenate(segments,0)
            #=============================load background image and resize==================================#
            # bg_img=cv2.imread(self.back_ground_img_list[index])
            # interp = cv2.INTER_LINEAR if (self.augment) else cv2.INTER_AREA
            # bg_img = cv2.resize(bg_img, (640,640), interpolation=interp)
            # new_img=img.copy()
            #=================================================================================================#
            if self.augment:
                img, labels, segments = random_perspective(
                    img,
                    labels,
                    segments=segments,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                    # return_seg=True,
                )

        nl = len(labels)  # number of labels
        # print(f'this is segment: {segments.shape}')
        if nl: # Có nhảy vào đây

            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
            if self.overlap:
                
                #=============================== Custom ==============================================#

                # test_mask = []

                # idx=3
                # bg_img=cv2.imread(self.back_ground_img_list[bg_id])
                # # print(f'here are lists: {self.back_ground_img_list[indx]}')
                # interp = cv2.INTER_LINEAR if (self.augment) else cv2.INTER_AREA
                # bg_img = cv2.resize(bg_img, (img.shape[0],img.shape[1]), interpolation=interp)

                # # for idx in range(len(segments)):
                # polygons=[segments[idx].reshape(-1)]
                # mask = np.zeros(img.shape[:2], dtype=np.uint8) #(H,W)
                # polygons = np.asarray(polygons) # polyps --> array--int32
                # polygons = polygons.astype(np.int32)
                # shape = polygons.shape
                # polygons = polygons.reshape(shape[0], -1, 2) # N,poly_point,2
                # cv2.fillPoly(mask, polygons, color=(125,125,0))
                # cv2.fillPoly(img, polygons, color=(125,125,0))
                # cv2.imwrite(f'/home/tonyhuy/bottle_classification/seg/runs/train-seg/exp5/{random.randint(0,1000)}.jpg',bg_img)
                
                
                # nh, nw = (img.shape[0] // self.downsample_ratio, img.shape[1] // self.downsample_ratio)
                # # NOTE: fillPoly firstly then resize is trying the keep the same way
                # # of loss calculation when mask-ratio=1.
                # mask = cv2.resize(mask, (nw, nh)) #(W,H)
                # test_mask.append(mask)
                # test_mask=np.array(test_mask)
                # exit()
                #=====================================================================================#

                
                masks, sorted_idx = polygons2masks_overlap(img.shape[:2],
                                                           segments, #(n, 1000, 2)
                                                           downsample_ratio=self.downsample_ratio)
                
                masks = masks[None]  # (640, 640) -> (1, 640, 640)

                # sorted_idx [ 8  9 11  5 14 10  6  7  3  0 12  4  1  2 13] [0 2 1 3]
                # print(f'sorted_id and label:{sorted_idx} and {labels}')
                # m = np.array([17, 15 ,25 ,11 , 4, 27 ,10  ,6 ,12,1 ,21 ,26 , 2 ,22  ,3 ,16 ,24 ,13 , 8 , 5 ,23 , 7 ,14 ,18 ,20 ,19 , 0 , 9])
                # x=labels[m]
                # print('gia trị m',m)
                labels = labels[sorted_idx] # [[class,x1,y1,x2,y2]...[class,x1,y1,x2,y2]] 
                # print(f'label ne:{labels}')
                # [[         3     0.78513      0.7292    0.043827    0.059326]
                # [          0     0.74264      0.9315    0.051844    0.045965]]
                
            else:
                masks = polygons2masks(img.shape[:2], segments, color=1, downsample_ratio=self.downsample_ratio)
                # cv2.imwrite(f'/home/tonyhuy/bottle_classification/seg/runs/train-seg/exp5/{index}_mask.jpg',masks)

        masks = (torch.from_numpy(masks) if len(masks) else torch.zeros(1 if self.overlap else nl, img.shape[0] //
                                                                        self.downsample_ratio, img.shape[1] //
                                                                        self.downsample_ratio))
        # TODO: albumentations support
        if self.augment:
            # Albumentations
            # there are some augmentation that won't change boxes and masks,
            # so just be it for now.
            #=================== Thêm=======================#
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
                    masks = torch.flip(masks, dims=[1])

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
                    masks = torch.flip(masks, dims=[2])

            # Cutouts  # labels = cutout(img, labels, p=0.5)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)


        # Save image
        # cv2.imwrite(os.path.join('/home/charleschinh/TOMO/AugImages', str(uuid.uuid4()) + '.png'), np.uint8(img))
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # Image RGB: [C,H,W] labels : [n,6] and masks: [16,W,H]
        return (torch.from_numpy(img), labels_out, self.im_files[index], shapes, masks)
    
    def bbox_ioa(self,box1, box2, eps=1e-7):
        """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
        box1:       np.array of shape(4)
        box2:       np.array of shape(nx4)
        returns:    np.array of shape(n)
        """

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                    (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

        # Intersection over box2 area
        return inter_area / box2_area
    
    def segment_one_mask(self,im,bg_img,random_index,labels,segments,p=0.5):
        n = len(segments)
        j=random_index
        if p and n:
            h, w, c = im.shape  # height, width, channels
            im_new = np.zeros(im.shape, np.uint8)
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = self.bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

            result = cv2.bitwise_and(src1=im, src2=im_new)
            result = cv2.flip(result, 1)  # augment segments (flip left-right)
            i = result > 0  # pixels to replace
            # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
            im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug
            #================================== Debug ====================================#
            # class_id,x1,y1,x2,y2=labels[j]
            # bg_img[int(y1)-10:int(y2)+40,int(x1)-10:int(x2)+10]=im[int(y1)-10:int(y2)+40,int(x1)-10:int(x2)+10]
            # cv2.fillPoly(bg_img, pts=[segments[j].astype(np.int32)], color=(255, 0, 0))
            # cv2.rectangle(bg_img, (int(x1),int(y1)), (int(x2),int(y2)),(0,125,255),3) 
            # cv2.imwrite(f'/home/tonyhuy/bottle_classification/seg/runs/train-seg/exp5/{random.randint(0,1000)}.jpg',bg_img)
            #=============================================================================#
        return im, labels, segments #bg_img


    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4, = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        # 3 additional image indices
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w)= self.load_image(index) #(360,640,3)
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            
            #============================================#
            # print('this is length label',len(labels5),'\n')
            # print('this is length segments',len(segments5),'\n')
            #============================================#
            labels4.append(labels)
            segments4.extend(segments)
        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        random_index= random.randint(0,len(labels4)-1)

        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate
        random_number= 0.5 # random.uniform(0,1)
        if random_number == 0.3:
            # empty_mask= np.zeros((img4.shape[0],img4.shape[1]))
            # empty_mask= np.stack([empty_mask,empty_mask,empty_mask],axis=2)
            # for indx in range(len(self.back_ground_img_list)):

            # Before image (1280,1280,3)| label length : n | segment length :n|

            img4, labels4, segments4 = self.segment_one_mask(img4,bg_img,random_index,labels4,segments4,p=0.5)
            # print('before',img4.shape,len(labels4),len(segments4))

            # Augment
            # img4, labels4, segments4 = copy_paste(bg_img, labels4, segments4, p=self.hyp["copy_paste"])

            # After image (1280,1280,3) | label length : n | segment length :n |
            img4, labels4, segments4 = custom_random_perspective(img4,random_index,bg_img,
                                                        labels4,
                                                        segments4,
                                                        degrees=self.hyp["degrees"],
                                                        translate=self.hyp["translate"],
                                                        scale=self.hyp["scale"],
                                                        shear=self.hyp["shear"],
                                                        perspective=self.hyp["perspective"],
                                                        border=self.mosaic_border)  # border to remove
            # print('After',img4.shape,len(labels4),len(segments4))
        else:
            img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
            img4, labels4, segments4 = random_perspective(img4,
                                                        labels4,
                                                        segments4,
                                                        degrees=self.hyp["degrees"],
                                                        translate=self.hyp["translate"],
                                                        scale=self.hyp["scale"],
                                                        shear=self.hyp["shear"],
                                                        perspective=self.hyp["perspective"],
                                                        border=self.mosaic_border)  # border to remove

        # print(f'Check dau ra:{img4.shape}') 
        return img4, labels4, segments4 #img :#(640,640,3)

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, masks = zip(*batch)  # transposed
        batched_masks = torch.cat(masks, 0)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, batched_masks

def image_polygon(image,polygons,color=1):
    polygons = np.asarray(polygons) # polyps --> array--int32
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(1, -1, 2) # N,poly_point,2 #shape[0]
    cv2.fillPoly(image, polygons, color=color)
    return image


def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
            First create zero mask and fill polygons into zero mask,zero mask is a list,so
            we need to convert list to array. To fill polygons, we reshape (N, num_point,2)
    """
    mask = np.zeros(img_size, dtype=np.uint8) #(H,W)
    polygons = np.asarray(polygons) # polyps --> array--int32
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2) # N,poly_point,2
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh)) #(W,H)

    # cv2.imwrite(f'/home/tonyhuy/bottle_classification/seg/runs/train-seg/exp5/{random.randint(0,1000)}_mask.jpg',mask)

    return mask


def polygons2masks(img_size, polygons, color, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M],
            N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(img_size, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((img_size[0] // downsample_ratio, img_size[1] // downsample_ratio), dtype=np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        # mask array
        mask = polygon2mask(
            img_size,
            [segments[si].reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
    ms.append(mask)
    areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index
