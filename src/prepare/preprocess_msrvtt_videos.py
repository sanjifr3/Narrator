import cv2
import numpy as np
import json
import os
from pprint import pprint
import sys
import argparse
import torch
import os
import pickle
import PIL

DIR_NAME = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR_NAME + '/../')

from utils.create_transformer import create_transformer
from models.EncoderCNN import EncoderCNN

transformer = create_transformer()

def getVidArray(video_path, res=224, num_frames=40, frames_int=1):
  """
  Read in video and create numpy volume of shape 
    (num_samples, num_frames, 3, res, res)

  """
  try:
    cap = cv2.VideoCapture(video_path)
  except:
    print("Can not open %s" % (video_path))
    return None

  if frames_int is None:
    frames_int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / float(num_frames+1))

  vid_array = torch.zeros(num_frames,3, res, res)

  if torch.cuda.is_available():
    vid_array = vid_array.cuda() 

  frame_idx = 0

  while True:
    ret,frame = cap.read()

    if not ret or frame_idx == num_frames:
      break

    try:
      frame = PIL.Image.fromarray(frame).convert('RGB')

      if torch.cuda.is_available():
        frame = transformer(frame).cuda().unsqueeze(0)
      else:
        frame = transformer(frame).unsqueeze(0)

      vid_array[frame_idx] = frame
      frame_idx += 1
    except OSError as e:
      print("Could not process frame in " + video_path)

  cap.release()
  return vid_array

def main(args):
  if args.dir[-1] != '/':
    args.dir += '/'

  encoder = EncoderCNN(args.model)
  if torch.cuda.is_available():
    encoder.cuda()

  encoder.eval()
  
  video_files = [f.split('.')[0] for f in os.listdir(args.dir) if '.mp4' in f]
  if args.continue_preprocessing:
    done_files = [f.split('.')[0] for f in os.listdir(args.dir) if args.model + '.pkl' in f]
    video_files = [f for f in video_files if f not in done_files]

  for i,vid in enumerate(video_files):
    if i % 100 == 0:
      print("Embedding Videos: {}%\r".format(round(i/float(len(video_files))*100.0),2),end='')

    vid_array = getVidArray(args.dir + vid + '.mp4', args.resolution, args.num_frames, args.frames_interval)
    vid_array = encoder(vid_array)

    with open(args.dir + vid + '_' + args.model + '.pkl','wb') as f:
      pickle.dump(vid_array, f)

  sys.stdout.flush()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--resolution', type=int, help='CNN input resolution',
                      default=224)
  parser.add_argument('--num_frames', type=int, help='Number of frames to include in video',
                      default=40)
  parser.add_argument('--frames_interval', type=int, help='Interval between frames',
                      default=1)
  parser.add_argument('--embedding_size', type=int, help='Size of embedding',
                      default=2048)
  parser.add_argument('--model', type=str, help='Base CNN encoding model',
                      default='resnet152')
  parser.add_argument('--dir', type=str, help='Directory containing videos to encode',
                      default=os.environ['HOME'] + '/Database/MSR-VTT/train-video/')
  parser.add_argument('--continue_preprocessing', type=bool, help='Continue preprocessing or start from scratch',
                      default=True)
  args = parser.parse_args()
  main(args)