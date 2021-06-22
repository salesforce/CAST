# CASTing Your Model: Learning to Localize Improves Self-supervised Representations

This is a PyTorch implementation of our CVPR'21 paper

The code is built on top of the MoCo Framework

```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```


## Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).


## Dataset setup

The code requires you to have a folder of train and val images under <ImageFolder> and precomputed saliency maps <MaskFolder>. We use the Salency maps from DeepUSPS (code found [here](https://drive.google.com/file/d/10GlmenXR7nEJyRlmPHouvHP-g9KfUW1F/view)). We provide saliency maps computed for COCO [here](https://console.cloud.google.com/storage/browser/sfr-cast-data-research/saliency_data)
 
## Pre-trained models

The CAST pretrained models trained on COCO can be found [here](https://console.cloud.google.com/storage/browser/sfr-cast-data-research/Models)

## Unsupervised Training

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_cast.py -a resnet50 --cos  --lr 0.5   --batch-size 256   --dist-url 'tcp://localhost:10001' <ImageFolder> --mask-dir <MaskFolder>  --crit-gcam cosine --alpha-masked 3 --second-constraint "ref" --output-mask-region "ref" --num-gpus-per-machine 8  --print-freq 10 --workers 8
```

## Imagenet Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_200.pth \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```


## Code organization

`main_cast.py` contains the main code for our approach CAST. 
`grad_cam.py` contains functions that compute Grad-CAM maps for a specified layer. 
`moco/datasets.py` contains our Saliency Constrained Random Cropping data augmentation procedure. This uses functions from `moco/augumentations/transforms.py` and `moco/augmentations/functional.py`
`main_lincls.py` contains code to evaluate our self-trained model on the downstream task of imagenet linear classification.


## Citations
- If you find this codebase useful, please cite our paper:
```
@inproceedings{cast2021,
    title = {CASTing Your Model: Learning to Localize Improves Self-Supervised Representations},
    author = {Ramprasaath R. Selvaraju, Karan Desai, Justin Johnson, Nikhil Naik},
    booktitle = {CVPR},
    year = {2021}
}
```

## Contact
- Please send an email to rselvaraju@salesforce.com if you have questions.
