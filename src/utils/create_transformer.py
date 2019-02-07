from __future__ import print_function
import os
import sys
import argparse
import numbers
import collections
import pickle
import numpy as np
import PIL
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

class Resize(object):
    """
    Modified transforms.Resize to allow side selection
    
    If size is an int, use resize_large_size makes larger edge of the image to size.
    ----------------------------------------------------------
    Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR, resize_larger_edge=False):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.resize_larger = resize_larger_edge

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        
        if isinstance(self.size, int) and self.resize_larger:
            old_sz = img.size
            ratio = float(self.size)/max(old_sz)
            new_sz = tuple([int(x*ratio) for x in old_sz])
            sz = new_sz[::-1]
        else:
            sz = self.size

        return F.resize(img, sz, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class Pad(object):
    """
    Modified transforms.Pad to allow padding to a given size
    
    If single int is provided and it is greater than 100, image will be padded to (int,int)
    
    ----------------------------------------------------------
    Pad the given PIL Image on all sides with the given "pad" value.

    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            constant: pads with a constant value, this value is specified with fill
            edge: pads with the last value at the edge of the image
            reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        if self.padding > 100:
            sz = img.size
            delta_w = self.padding - sz[0]
            delta_h = self.padding - sz[1]
            
            t, b = delta_h//2, delta_h-(delta_h//2)
            l, r = delta_w//2, delta_w-(delta_w//2)
            pad = (l,t,r,b)
        else:
            pad = self.padding      
        
        return F.pad(img, pad, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)

def createTransformer(augment=False):
  if not augment:
    return transforms.Compose([
        Resize(224,resize_larger_edge=True), # Resize larger edge to 224
        Pad(224), # Pad smaller edge to 224
        transforms.ToTensor(), # Convert to tensor
        transforms.Normalize((0.485, 0.456, 0.406), # Normalize
                           (0.229, 0.224, 0.225))
    ])
  else:
    return transforms.Compose([
        transforms.resize(224), # Resize smaller edge to 256
        transforms.RandomCrop(224), # Get 224 x 224 crop from random location
        transforms.RandomHorizontalFlip(), # Random horizontal flips
        transforms.ToTensor(), # Convert to tensor
        transforms.Normalize((0.485, 0.456, 0.406), # Normalize
                           (0.229, 0.224, 0.225))
      ])

def createYoloTransformer(dim):
    return transforms.Compose([
        Resize(dim,resize_larger_edge=True), # Resize larger edge to dim
        Pad(dim, fill=(128,128,128)), # Pad smaller edge to dim
        transforms.ToTensor() # Convert to tensor
    ])