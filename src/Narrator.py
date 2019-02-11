from __future__ import print_function
import sys
import time
import os
import pickle
import cv2
import PIL
import nltk
import numpy as np
import pandas as pd
import skimage.io as io
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager

from scenedetect.detectors.content_detector import ContentDetector
from scenedetect.detectors.threshold_detector import ThresholdDetector

# pickle needs to be able to find Vocabulary to load vocab
DIR_NAME = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR_NAME)
sys.path.append(DIR_NAME + '/utils')
sys.path.append(DIR_NAME + '/models')

try:
    from models.ImageCaptioner import ImageCaptioner
    from models.VideoCaptioner import VideoCaptioner
    from models.EncoderCNN import EncoderCNN
except ImportError:
    from ImageCaptioner import ImageCaptioner
    from VideoCaptioner import VideoCaptioner
    from EncoderCNN import EncoderCNN

try:
    from utils.create_transformer import create_transformer
    from utils.Vocabulary import Vocabulary
    from utils.TTS import TTS
except ImportError:
    from create_transformer import create_transformer
    from Vocabulary import Vocabulary
    from TTS import TTS

class Narrator(object):
    img_extensions = ['jpg','png','jpeg']
    vid_extensions = ['mp4','avi']

    def __init__(self, root_path = '../',
                       coco_vocab_path = 'data/processed/coco_vocab.pkl',
                       msrvtt_vocab_path = 'data/processed/msrvtt_vocab.pkl',
                       base_model = 'resnet152',
                       ic_model_path = 'models/image_caption-model3-25-0.1895-4.7424.pkl',
                       vc_model_path = 'models/video_caption-model4-480-0.3936-5.0.pkl',
                       im_embedding_size = 2048,
                       vid_embedding_size = 2048,
                       embed_size = 256,
                       hidden_size = 512,
                       num_frames = 40,
                       max_caption_length = 35,
                       im_res = 224):

        self.num_frames = num_frames
        with open(root_path + msrvtt_vocab_path, 'rb') as f:
            self.msrvtt_vocab = pickle.load(f)
        with open(root_path + coco_vocab_path, 'rb') as f:
            self.coco_vocab = pickle.load(f)
       
        self.transformer = create_transformer()
        self.encoder = EncoderCNN(base_model)

        self.image_captioner = ImageCaptioner(
            im_embedding_size,
            embed_size,
            hidden_size,
            len(self.coco_vocab),
            start_id = self.coco_vocab.word2idx[self.coco_vocab.start_word],
            end_id = self.coco_vocab.word2idx[self.coco_vocab.end_word]
        )

        ic_checkpoint = torch.load(root_path + ic_model_path)
        self.image_captioner.load_state_dict(ic_checkpoint['params'])
        
        self.video_captioner = VideoCaptioner(
            vid_embedding_size,
            embed_size,
            hidden_size,
            len(self.msrvtt_vocab),
            start_id = self.msrvtt_vocab.word2idx[self.msrvtt_vocab.start_word],
            end_id = self.msrvtt_vocab.word2idx[self.msrvtt_vocab.end_word]
        )

        vc_checkpoint = torch.load(root_path + vc_model_path)
        self.video_captioner.load_state_dict(vc_checkpoint['params'])
        self.vid_embedding_size = vid_embedding_size

        self.tts = TTS()

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.image_captioner.cuda()
            self.video_captioner.cuda()

        self.encoder.eval()
        self.image_captioner.eval()
        self.video_captioner.eval()

    def genCaption(self, f, beam_size=5, as_string=False, by_scene=False):
        ext = f.split('.')[-1]
        if ext in self.img_extensions:
            return self.genImCaption(f, beam_size, as_string)
        elif ext in self.vid_extensions:
            return self.genVidCaption(f, beam_size, as_string, by_scene)
        else:
            return "ERROR: Invalid file type: " + ext

    def genVidCaption(self, f, beam_size=5, as_string=False, by_scene=False):
        if not os.path.exists(f):
            return "ERROR: File does not exist!"

        cap = cv2.VideoCapture(f)

        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        num_frames = self.num_frames

        if by_scene:
            scenes = self.findSceneChanges(f)
            scene_change_timecodes = [scene[0].get_timecode()[:-4] for scene in scenes]
            scene_change_idxs = [scene[0].get_frames() for scene in scenes]
        else:
            scene_change_timecodes = ['00:00:00']
            scene_change_idxs = [0]

        vid_embeddings = torch.zeros(len(scene_change_idxs),num_frames,self.vid_embedding_size) 
        if torch.cuda.is_available():
            vid_embeddings = vid_embeddings.cuda()
       
        last_frame = scene_change_idxs[-1] + num_frames + 1

        frame_idx = 0
        cap_start_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret or frame_idx == last_frame:
                break

            if frame_idx in scene_change_idxs:
                cap_start_idx = frame_idx
                vid_array = torch.zeros(num_frames, 3, 224, 224)

            if frame_idx - cap_start_idx < num_frames:
                try:          
                    frame = PIL.Image.fromarray(frame).convert('RGB')

                    if torch.cuda.is_available():
                        frame = self.transformer(frame).cuda().unsqueeze(0)
                    else:
                        frame = self.transformer(frame).unsqueeze(0)

                    vid_array[frame_idx - cap_start_idx] = frame
            
                except OSError as e:
                    print("Could not process frame in " + f)

            if frame_idx - cap_start_idx == num_frames:
                if torch.cuda.is_available():
                    vid_array = vid_array.cuda()
                vid_embeddings[scene_change_idxs.index(cap_start_idx)] = self.encoder(vid_array)

            frame_idx += 1

        cap.release()

        encoded_captions = self.video_captioner.predict(vid_embeddings, beam_size=beam_size).cpu().numpy().astype(int)

        captions = []
        for caption in encoded_captions:
            captions.append(self.msrvtt_vocab.decode(caption, clean=True, join=as_string))

        if not by_scene:
            return captions[0]
        
        return captions, scene_change_timecodes       

    def genImCaption(self, f, beam_size=5, as_string=False):
        if os.path.exists(f):
            im = PIL.Image.open(f).convert('RGB')
        else:
            try:
                im = PIL.Image.fromarray(io.imread(f)).convert('RGB')
            except:
               return "ERROR: File doesn't exist"

        im = self.transformer(im).cuda().unsqueeze(0)
        im = self.encoder(im)
        caption = self.image_captioner.predict(im, beam_size=beam_size)[0].cpu().numpy().astype(int)
        caption = self.coco_vocab.decode(caption, clean=True, join=as_string)

        return caption

    def getNearestMatch(self, caption, gts, dataset='coco', as_string=True):

        try:
            assert(dataset == 'coco' or dataset == 'msr-vtt')
        except AssertionError as e:
            print (e)
            return caption, gts

        if dataset == 'coco':
            vocab = self.coco_vocab
        elif dataset == 'msr-vtt':
            vocab = self.msrvtt_vocab

        gts = gts.apply(lambda x: vocab.encode(x, self.max_caption_length + 1))
        gts = gts.apply(lambda x: vocab.encode(x, clean=True))

        nearest_gt = None
        best_score = 0.0

        for gt in gts:
            bleu = vocab.evaluate([gt], caption)
            if bleu > best_score:
                best_score = bleu
                nearest_gt = gt

        if as_string:
            gt = ' '.join(nearest_gt).capitalize()
            caption = ' '.join(caption).capitalize()
        else:
            gt = nearest_gt

        return

    def genAudioFile(self, text, path):
        self.tts.generateAudio(text, path)
        return

    def findSceneChanges(self, video_path, method='threshold', new_stat_file=True):
        # type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
    
        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()
        
        # Construct our SceneManager and pass it our StatsManager.
        scene_manager = SceneManager(stats_manager)

        # Add ContentDetector algorithm (each detector's constructor
        # takes detector options, e.g. thresholsd).
        if method == 'content':
            scene_manager.add_detector(ContentDetector(threshold=30, min_scene_len=40))
        else:
            scene_manager.add_detector(ThresholdDetector(min_scene_len=40, threshold=125, min_percent=0.5))
            
        base_timecode = video_manager.get_base_timecode()

        # We save our stats file to {VIDEO_PATH}.{CONTENT}.stats.csv.
        stats_file_path = '%s.%s.stats.csv' % (video_path, method)

        scene_list = []

        try:
            # If stats file exists, load it.
            if not new_stat_file and os.path.exists(stats_file_path):
                # Read stats from CSV file opened in read mode:
                with open(stats_file_path, 'r') as stats_file:
                    stats_manager.load_from_csv(stats_file, base_timecode)

            # Set downscale factor to improve processing speed.
            video_manager.set_downscale_factor(2)

            # Start video_manager.
            video_manager.start()

            # Perform scene detection on video_manager.
            scene_manager.detect_scenes(frame_source=video_manager)

            # Obtain list of detected scenes.
            scene_list = scene_manager.get_scene_list(base_timecode)
            # Each scene is a tuple of (start, end) FrameTimecodes.

            # We only write to the stats file if a save is required:
            if stats_manager.is_save_required():
                with open(stats_file_path, 'w') as stats_file:
                    stats_manager.save_to_csv(stats_file, base_timecode)

        finally:
            video_manager.release()

        return scene_list

