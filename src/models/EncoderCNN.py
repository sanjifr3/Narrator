# -*- coding: utf-8 -*-
"""
A PyTorch CNN model wrapper.

Wraps a pre-trained CNN network
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    A PyTorch CNN model wrapper.

    This class inherits form the torch.nn.Module class
    """

    base_model_options = ['resnet18', 'resnet50', 'resnet152',
                          'vgg11', 'vgg11_bn', 'vgg16', 'vgg16_bn',
                          'vgg19', 'vgg19_bn', 'squeezenet0', 'squeezenet1',
                          'densenet121', 'densenet201', 'inception']

    def __init__(self, base_model='vgg'):
        """
        Construct the EncoderCNN class.

        Args:
            base_model: Base CNN model to use
        Returns;
            A PyTorch network model

        """
        super(EncoderCNN, self).__init__()

        try:
            assert base_model in self.base_model_options
        except AssertionError:
            print('Invalid base model: %s'.format(base_model))
            print(' -- Valid types: ', self.base_model_options)
            return

        # Load selected base model with pre-trained weights
        if base_model == 'resnet18':
            self.bm = models.resnet18(pretrained=True)
        elif base_model == 'resnet50':
            self.bm = models.resnet50(pretrained=True)
        elif base_model == 'resnet152':
            self.bm = models.resnet152(pretrained=True)
        elif base_model == 'vgg11':
            self.bm = models.vgg11(pretrained=True)
        elif base_model == 'vgg11_bn':
            self.bm = models.vgg11_bn(pretrained=True)
        elif base_model == 'vgg16':
            self.bm = models.vgg16(pretrained=True)
        elif base_model == 'vgg16_bn':
            self.bm = models.vgg16_bn(pretrained=True)
        elif base_model == 'vgg19':
            self.bm = models.vgg19(pretrained=True)
        elif base_model == 'vgg19_bn':
            self.bm = models.vgg19_bn(pretrained=True)
        elif base_model == 'squeezenet0':
            self.bm = models.squeezenet1_0(pretrained=True)
        elif base_model == 'squeezenet1':
            self.bm = models.squeezenet1_1(pretrained=True)
        elif base_model == 'densenet121':
            self.bm = models.densenet121(pretrained=True)
        elif base_model == 'densenet201':
            self.bm = models.densenet201(pretrained=True)
        elif base_model == 'inception':
            self.bm = models.inception_v3(pretrained=True)

        # Freeze layers
        for param in self.bm.parameters():
            param.requires_grad = False

        modules = list(self.bm.children())[:-1]
        # num_features = self.bm.fc.in_features

        self.bm = nn.Sequential(*modules)

        return

    def forward(self, images):
        """
        Compute the forward pass of the network.

        Args:
            images: Image Tensor

        Returns:
            Image embeddings from second last layer of base network

        """
        with torch.no_grad():
            features = self.bm(images)
        return features.view(features.size(0), -1)
