
from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot[:,0]=1
        return one_hot

    def forward(self, image_q, image_k, add_to_queue):
        self.image_shape = image_q.shape[2:]
        
        self.logits, self.target , self.intermediate= self.model(image_q, image_k, add_to_queue=add_to_queue, return_intermediate_outputs=True)

        return self.logits, self.target

    def backward(self, ids):
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list
   
    def backward(self, ids, target_layer):
        """
        Class-specific backpropagation
        """
        # one hot encode the location of the correct key (0)
        one_hot = self._encode_one_hot(ids)
        fmaps = self.intermediate['layer4']

        grad_wrt_act = torch.autograd.grad(outputs=self.logits, inputs=fmaps, grad_outputs=one_hot, create_graph=True)[0]

        return grad_wrt_act

    def generate(self, target_layer, grads):
        
        fmaps = self.intermediate['layer4']
        weights = F.adaptive_avg_pool2d(grads, 1) 

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        B, C, H, W = gcam.shape

        gcam_raw = gcam
        gcam_raw = gcam_raw.view(B, -1)
        gcam_raw -= gcam_raw.min(dim=1, keepdim=True)[0]
        gcam_raw /= (gcam_raw.max(dim=1, keepdim=True)[0]+0.0000001)
        gcam_raw = gcam_raw.view(B, C, H, W)

        gcam = F.relu(gcam)

        # uncomment to scale gradcam to image size
        # gcam = F.interpolate(
        #     gcam, self.image_shape, mode="bilinear", align_corners=False
        # )

        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= (gcam.max(dim=1, keepdim=True)[0]+0.0000001)
        gcam = gcam.view(B, C, H, W)

        return gcam
