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
import cv2
import torch

DIR_NAME = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR_NAME + '/../')
from models.YOLOObjectDetector import YOLOObjectDetector

def main(args):
    yolo = YOLOObjectDetector(model_name=args.model)
    
    sets = ['train2014','val2014','train2017','val2017']

    for i, st in enumerate(sets):
        print ("Preprocessing " + st + " images with " + args.model)
        # Get list of images
        images = [f.split('.')[0] for f in os.listdir(args.dir+st) if '.jpg' in f]
        if args.continue_preprocessing:
          converted = [f.split('.')[0] for f in os.listdir(args.dir+st) if args.model + '.pkl' in f]
          images = [f for f in images if f not in converted]
        
        for j, im_file in enumerate(images):
            if j % 100 == 0:
                print("[{}/{}] Progress: {}%\r".format(i, len(sets), round(j/float(len(images))*100.0),2),end="")
            try:
              im = cv2.imread(args.dir + st + '/' + im_file + '.jpg')
              results = {'classes': None, 'output': None}
              results['classes'], results['output'] = yolo.detect(im, max_objects=args.max_objects)

              with open(args.dir + st + '/' + im_file + '_' + args.model + '.pkl', 'wb') as f:
                  pickle.dump(results, f)
            except OSError as e:
              print ("Could not process " + args.dir + st + '/' + im_file + '.jpg')

        sys.stdout.flush()
        print('Done preprocessing ' + st + ' images')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, help='YOLO model',
                      default='yolov3')
  parser.add_argument('--dir', type=str, help='Directory containing videos to encode',
                      default=os.environ['HOME'] + '/Database/coco/images/')
  parser.add_argument('--continue_preprocessing', type=bool, help='Continue preprocessing or start from scratch',
                      default=False)
  parser.add_argument('--max_objects', type=int, help="Top K classes to keep from yolo per frame",
                      default=4)
  args = parser.parse_args()
  main(args)