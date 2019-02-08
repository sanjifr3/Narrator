"""
Simple wrapper for the pytorch implementation from:
https://github.com/ayooshkathuria/pytorch-yolo-v3
"""
from __future__ import division, print_function
import sys
import os
import pickle
import numpy as np
import cv2

import torch
from torch.autograd import Variable

DIR_NAME = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR_NAME + '/pytorch-yolo-v3')
from util import load_classes, write_results
from darknet import Darknet


class YOLOObjectDetector(object):
    """
    Simple wrapper class for pytorch yolov3 implementation from:
      https://github.com/ayooshkathuria/pytorch-yolo-v3
    """

    def __init__(self, darknet_path=os.environ['HOME'] + '/programs/darknet/',
                 model_name='yolov3', dataset='coco', dim=480):
        """
        Constructs the YOLOObjectDetector class

        Args:
            darknet_path: Path to darknet
            model_name: Model to use
            dataset: Dataset for model
            dim: Input dimension of YOLO network

        """
        # YOLO file paths
        cfg_file = '%s/cfg/%s.cfg' % (darknet_path, model_name)
        weights_file = '%s/weights/%s.weights' % (darknet_path, model_name)
        classes_file = '%s/data/%s.names' % (darknet_path, dataset)

        # Liad classes
        self.classes = load_classes(classes_file)
        self.np_classes = np.array(self.classes)
        self.colors = pickle.load(
            open(
                DIR_NAME +
                "/pytorch-yolo-v3/pallete",
                "rb"))

        # Load model
        print('Loading ' + model_name + ' network')
        self.model = Darknet(cfg_file)
        self.model.load_weights(weights_file)
        print(model_name + ' network loaded successfully')

        # Check dimension to see if its valid
        self.dim = dim
        try:
            assert self.dim % 32 == 0 and self.dim > 32
        except AssertionError as e:
            print(e)
            print("Invalid model dimension -- must be multiple of 32")
            self.dim = int(self.dim / 32) * 32
            print("Moded dimension changed to {}".format(self.dim))

        # Change input size
        self.model.net_info['height'] = self.dim

        # Move to gpu if available
        if torch.cuda.is_available():
            self.model.cuda()

        # Do one pass and set to evaluation mode
        self.model(self.get_test_input(), torch.cuda.is_available())
        self.model.eval()

        return

    def detect(self, im, thresh=0.3, nms_thresh=0.45,
               max_objects=4, draw=False):
        """
        Detects objects in given image

        Args:
            im: Image
            thresh: Detection threshold
            nms: Detection non-max suppression threshold
            max_objects: Maximum number of objects to return
            draw: Option to draw results on image
        Returns:
            object classes plus details and optionally an
            image with results drawn
        """
        # Preprocess image
        im, orig_im, dim = self.prep_image(im)
        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        # Move to GPU if available
        if torch.cuda.is_available():
            im_dim = im_dim.cuda()
            im = im.cuda()

        # Compute output
        with torch.no_grad():
            output = self.model(Variable(im), torch.cuda.is_available())

        # Process results
        output = write_results(
            output, thresh, len(
                self.classes), nms=True, nms_conf=nms_thresh)

        classes = []
        if not isinstance(output, int):
            # Rescale output to original image size
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(self.dim / im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (self.dim - scaling_factor *
                                  im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (self.dim - scaling_factor *
                                  im_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(
                    output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(
                    output[i, [2, 4]], 0.0, im_dim[i, 1])

            # Optionally draw on image
            if draw:
                list(map(lambda x: self.draw(x, orig_im), output))

            # Extract classes, locations, propabilities from output
            if output.size()[0] > 0:
                if max_objects is not None and output.size()[0] > max_objects:
                    output = output[-max_objects:]

                # Reverse ordering of output
                reversed_indx = np.arange(output.size()[0])[::-1].tolist()
                output = output[reversed_indx, :]

                classes = self.np_classes[
                    output[:, -1].int().cpu().numpy().tolist()].tolist()
                output = output[:, 1:7]
                output[:, 2] -= output[:, 0]  # Compute and store width
                output[:, 3] -= output[:, 1]  # Compute and store height
                output[:, 0] += 0.5 * output[:, 2]  # Compute and store xc
                output[:, 1] += 0.5 * output[:, 3]  # Compute and store yc
                output[:, [0, 2]] /= dim[0]  # Get percentage by im width
                output[:, [1, 3]] /= dim[1]  # Get percentage by im height

                # Change format to: obj_prob, class_prob, xc,yc,w,h
                output = output[:, [-2, -1, 0, 1, 2, 3]]

                # Pad to max objects size
                if max_objects is not None and output.size()[0] < max_objects:
                    sz = (4 - output.size()[0], output.size()[1])
                    output = torch.cat((output, torch.zeros(sz).cuda()), 0)
                    classes = classes + ['<pad>'] * sz[0]

        if draw:
            return classes, output, orig_im

        return classes, output

    def draw(self, x, im):
        '''Draw object on given image'''
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        cls = int(x[-1])
        label = "{0}".format(self.classes[cls])

        color = self.colors[np.random.randint(len(self.colors))]
        cv2.rectangle(im, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(im, c1, c2, color, -1)
        cv2.putText(im, label, (c1[0], c1[1] + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    def prep_image(self, im):
        '''Prepare image for inputting into neural network'''
        orig_im = im
        dim = orig_im.shape[1], orig_im.shape[0]
        im = (self.letterbox_image(orig_im))
        im_ = im[:, :, ::-1].transpose((2, 0, 1)).copy()
        im_ = torch.from_numpy(im_).float().div(255.0).unsqueeze(0)
        return im_, orig_im, dim

    def letterbox_image(self, im):
        '''resize image with unchanged aspect ratio using padding'''
        img_w, img_h = im.shape[1], im.shape[0]
        w, h = self.dim, self.dim
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))
        resized_image = cv2.resize(
            im, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        canvas = np.full((self.dim, self.dim, 3), 128)
        canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) //
               2:(w - new_w) // 2 + new_w, :] = resized_image
        return canvas

    def get_test_input(self):
        '''Get sample test input for network'''
        im = cv2.imread(DIR_NAME + '/pytorch-yolo-v3/dog-cycle-car.png')
        im = cv2.resize(im, (self.dim, self.dim))
        im_ = im[:, :, ::-1].transpose((2, 0, 1))
        im_ = im_[np.newaxis, :, :, :] / 255.0
        im_ = torch.from_numpy(im_).float()
        im_ = Variable(im_)

        if torch.cuda.is_available():
            im_ = im_.cuda()

        return im_
