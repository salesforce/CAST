import os
import random
import sys
import albumentations as alb
from PIL import Image
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from albumentations.pytorch.transforms import ToTensorV2
import scipy
from skimage.transform import resize
import time
from moco.augmentations import transforms as T
from loguru import logger


class SaliencyConstrainedRandomCropping(Dataset):
    def __init__(self, main_dir, mask_dir, split, second_constraint, output_mask_region, output_mask_size, min_areacover=0.2, mask_threshold=128, create_pretty=False):
        self.main_dir = os.path.join(main_dir, split)
        self.mask_dir = os.path.join(mask_dir, split)
        self.output_mask_size = output_mask_size
        self.split = split
        self.mask_threshold = mask_threshold
        self.create_pretty = create_pretty
        self.min_areacover = min_areacover
        
        all_imgs = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.main_dir)) for f in fn]
        random.shuffle(all_imgs)

        self.total_imgs = all_imgs
        
        self.rr_crop = T.MaskConstraintRandomResizedPairCrop(
        height=224, width=224, min_areacover=self.min_areacover, second_constraint=second_constraint, output_mask_region=output_mask_region
        )

        # whether to use crops that are easy to see (only for visualization purposes)
        if not self.create_pretty:
            self.transform = alb.Compose([
                alb.HorizontalFlip(p=0.5),
                alb.ToGray(p=0.2),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=0.5),
                alb.Normalize(mean=T.IMAGENET_COLOR_MEAN, std=T.IMAGENET_COLOR_STD, p=1.0),
                ToTensorV2(),
            ])
        else:
            self.transform = alb.Compose([
                alb.HorizontalFlip(p=0.5),
                alb.ToGray(p=0.2),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0, p=0.5),
                alb.Normalize(mean=T.IMAGENET_COLOR_MEAN, std=T.IMAGENET_COLOR_STD, p=1.0),
                ToTensorV2(),
            ])
        logger.info("COCO - Initializing custom dataset for MOCOv1 {}...".format(self.split))
        logger.info("Number of images in COCO {}: {}".format(self.split, len(self.total_imgs)))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        folder_name =  os.path.basename(os.path.dirname(self.total_imgs[idx]))
        file_name = os.path.basename(self.total_imgs[idx])
        mask_loc = os.path.join(os.path.join(self.mask_dir, folder_name), file_name).replace('.jpg','.png')

        # load image
        image = np.array(Image.open(img_loc).convert('RGB'))

        # load saliency maps
        try:
            gt_mask = np.array(Image.open(mask_loc).convert('L'))
            if gt_mask.sum() == 0:
                new_idx = np.random.randint(0, len(self.total_imgs))
                return self.__getitem__(new_idx)

            # binarize mask
            gt_mask[gt_mask <= self.mask_threshold] = 0      # Black
            gt_mask[gt_mask > self.mask_threshold] = 1     # White
            
        except:
            new_idx = np.random.randint(0, len(self.total_imgs))
            return self.__getitem__(new_idx)
            
        try:
            query_crop, key_crop, mask_query, mask_key = self.rr_crop(image=image, reference_mask=gt_mask)['image']
        except Exception as e:
            # account for failed attempts
            new_idx = np.random.randint(0,len(self.total_imgs))
            return self.__getitem__(new_idx)
        
        # now that we have query crops, key crops and saliency maps corresponding to the query and key crops, apply other transforms for each of the crops independently
        query_output = self.transform(image=query_crop, mask=mask_query)
        query_crop, mask_query = query_output["image"], query_output["mask"]
        key_output = self.transform(image=key_crop, mask=mask_key)
        key_crop, mask_key = key_output["image"], key_output["mask"]

        
        mask_query = mask_query.unsqueeze(0).unsqueeze(0)
        mask_key = mask_key.unsqueeze(0).unsqueeze(0)
        mask_query_raw = mask_query.squeeze(0)
        mask_key_raw = mask_key.squeeze(0)

        # downsample for speed (note that gradcam on the last layer of resnet is 7x7), so this resolution is usually sufficient for the loss
        mask_query_interp = torch.nn.functional.interpolate(mask_query.type(torch.FloatTensor), self.output_mask_size, mode='bilinear', align_corners=True).squeeze(0)
        mask_key_interp = torch.nn.functional.interpolate(mask_key.type(torch.FloatTensor), self.output_mask_size, mode='bilinear', align_corners=True).squeeze(0)

        # ensure that downsizing doesn't lead to empty maps
        if mask_query_interp.sum()==0 or mask_key_interp.sum() == 0:
            new_idx = np.random.randint(0,len(self.total_imgs))
            return self.__getitem__(new_idx)
        
         # query and key crops
        images = [query_crop, key_crop]
        
        # create masked key from key and key-based saliency map
        masked_key = images[1] * mask_key.squeeze(1)  

        # return query crop, key crop, mask corresponding to query, masked key crop
        return images, img_loc, mask_query_interp, masked_key

