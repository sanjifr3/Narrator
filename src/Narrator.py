# -*- coding: utf-8 -*-
"""
The Narrator class.

This class serves the models and frameworks within narrator in a single object.
"""
from __future__ import print_function
import sys
import os
import pickle
import PIL
import skimage.io as io
import torch

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager

from scenedetect.detectors.content_detector import ContentDetector
from scenedetect.detectors.threshold_detector import ThresholdDetector

import cv2

# pickle needs to be able to find Vocabulary to load vocab
DIR_NAME = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR_NAME)
sys.path.append(DIR_NAME + '/utils')
sys.path.append(DIR_NAME + '/models')

# To account for path errors
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
    """
    The Narrator class.

    This class serves the models and frameworks within narrator in
    a single object.
    """
    img_extensions = ['jpg', 'png', 'jpeg']
    vid_extensions = ['mp4', 'avi']

    def __init__(self,
                 root_path='../',
                 coco_vocab_path='data/processed/coco_vocab.pkl',
                 msrvtt_vocab_path='data/processed/msrvtt_vocab.pkl',
                 base_model='resnet152',
                 ic_model_path='models/image_caption-model3-25-0.1895-4.7424.pkl',
                 vc_model_path='models/video_caption-model11-160-0.3501-5.0.pkl',
                 im_embedding_size=2048,
                 vid_embedding_size=2048,
                 embed_size=256,
                 hidden_size=512,
                 num_frames=40,
                 max_caption_length=35,
                 ic_rnn_type='lstm',
                 vc_rnn_type='gru',
                 im_res=224):
        """
        Construct the Narrator class.

        Args:
            root_path: Path to narrator git root
            coco_vocab_path: Path to the COCO vocab pickle
            msrvtt_vocab_path: Path to the MSR-VTT vocab pickle
            base_model: Base model for the CNN encoder
            ic_model_path: Path to the image captioning model
            vc_model_path: Path to the video captioning model
            im_embedding_size: Size of image embedding
            vid_embedding_size: Size of video embedding
            embed_size; Size of image/word embedding
            hidden_size: Size of the hidden embedding
            num_frames: Number of frames
            max_caption_length: Maximum number of captions
            im_res: Resolution of input image/video)

        """

        # Store class variables
        self.num_frames = num_frames
        self.vid_embedding_size = vid_embedding_size
        self.max_caption_length = max_caption_length

        # Load vocabularies
        with open(root_path + msrvtt_vocab_path, 'rb') as f:
            self.msrvtt_vocab = pickle.load(f)
        with open(root_path + coco_vocab_path, 'rb') as f:
            self.coco_vocab = pickle.load(f)

        # Load transformer and image encoder
        self.transformer = create_transformer()
        self.encoder = EncoderCNN(base_model)

        # Create image captioner and load weights
        self.image_captioner = ImageCaptioner(
            im_embedding_size,
            embed_size,
            hidden_size,
            len(self.coco_vocab),
            rnn_type=ic_rnn_type,
            start_id=self.coco_vocab.word2idx[self.coco_vocab.start_word],
            end_id=self.coco_vocab.word2idx[self.coco_vocab.end_word]
        )

        ic_checkpoint = torch.load(root_path + ic_model_path)
        self.image_captioner.load_state_dict(ic_checkpoint['params'])

        # Create video captioner and load weights
        self.video_captioner = VideoCaptioner(
            vid_embedding_size,
            embed_size,
            hidden_size,
            len(self.msrvtt_vocab),
            rnn_type=vc_rnn_type,
            start_id=self.msrvtt_vocab.word2idx[self.msrvtt_vocab.start_word],
            end_id=self.msrvtt_vocab.word2idx[self.msrvtt_vocab.end_word]
        )

        vc_checkpoint = torch.load(root_path + vc_model_path)
        self.video_captioner.load_state_dict(vc_checkpoint['params'])

        # Construct TTS
        self.tts = TTS()

        # Push all torch models to GPU / set on eval mode
        if torch.cuda.is_available():
            self.encoder.cuda()
            self.image_captioner.cuda()
            self.video_captioner.cuda()

        self.encoder.eval()
        self.image_captioner.eval()
        self.video_captioner.eval()

    def gen_caption(self, f, beam_size=5, as_string=False, by_scene=False):
        """
        Generate a caption for given image/video file.

        Args:
            f: Path to file
            beam_size: Beam size for beam search
            as_string: Return caption as string or list
            by_scene: Caption by scene or not

        Return:
            A list or string caption.

        """

        # Use extention to determine what type of file and call relevant method
        ext = f.split('.')[-1]
        if ext in self.img_extensions:
            return self.gen_im_caption(f, beam_size, as_string)
        elif ext in self.vid_extensions:
            return self.gen_vid_caption(f, beam_size, as_string, by_scene)

        return 'ERROR: Invalid file type: ' + ext

    def gen_vid_caption(self, f, beam_size=5, as_string=False, by_scene=False):
        """
        Generates a caption for a given video file.

        Args:
            f: Path to file
            beam_size: Beam size for beam search
            as_string: Return caption as string or list
            by_scene: Caption by scene or not

        Return:
            A list or string caption.

        """
        if not os.path.exists(f):
            return 'ERROR: File does not exist!'

        cap = cv2.VideoCapture(f)

        # total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        num_frames = self.num_frames

        # Find scene splits if requested or just treat the first 40 frames
        # as one sample
        if by_scene:
            scenes = self.find_scene_changes(f)
            scene_change_timecodes = [
                scene[0].get_timecode()[:-4] for scene in scenes]
            scene_change_idxs = [scene[0].get_frames() for scene in scenes]
            
            if len(scene_change_idxs) == 0:
                scene_change_timecodes = ['00:00:00']
                scene_change_idxs = [0]
        else:
            scene_change_timecodes = ['00:00:00']
            scene_change_idxs = [0]

        # Empty torch tensor's to store values
        vid_embeddings = torch.zeros(
            len(scene_change_idxs), num_frames, self.vid_embedding_size)
        if torch.cuda.is_available():
            vid_embeddings = vid_embeddings.cuda()

        # Determine last frame to analyze
        last_frame = scene_change_idxs[-1] + num_frames + 1

        frame_idx = 0
        cap_start_idx = 0

        # Loop through and store relevant frames
        while True:
            ret, frame = cap.read()

            if not ret or frame_idx == last_frame:
                break

            # Start storing frames
            if frame_idx in scene_change_idxs:
                cap_start_idx = frame_idx
                vid_array = torch.zeros(num_frames, 3, 224, 224)

            # Transform, and store
            if frame_idx - cap_start_idx < num_frames:
                try:
                    frame = PIL.Image.fromarray(frame).convert('RGB')

                    if torch.cuda.is_available():
                        frame = self.transformer(frame).cuda().unsqueeze(0)
                    else:
                        frame = self.transformer(frame).unsqueeze(0)

                    vid_array[frame_idx - cap_start_idx] = frame

                except OSError as e:
                    print(e + " could not process frame in " + f)

            # If at scene ending frame, encode the collected scene
            if frame_idx - cap_start_idx == num_frames:
                if torch.cuda.is_available():
                    vid_array = vid_array.cuda()
                vid_embeddings[scene_change_idxs.index(
                    cap_start_idx)] = self.encoder(vid_array)

            frame_idx += 1

        cap.release()

        # Predict captions using the video embeddings
        encoded_captions = self.video_captioner.predict(
            vid_embeddings, beam_size=beam_size).cpu().numpy().astype(int)

        # Convert word ids to word tags
        captions = []
        for caption in encoded_captions:
            captions.append(self.msrvtt_vocab.decode(
                caption, clean=True, join=as_string))

        # Return scene or optionally scene with timecodes
        if not by_scene:
            return captions[0]

        return captions, scene_change_timecodes

    def gen_im_caption(self, f, beam_size=5, as_string=False):
        """
        Generates a caption for a given image file or link.

        Args:
            f: Path to file or link
            beam_size: Beam size for beam search
            as_string: Return caption as string or list

        Return:
            A list or string caption.

        """
        # Load data
        if os.path.exists(f):
            im = PIL.Image.open(f).convert('RGB')
        else:
            try:
                im = PIL.Image.fromarray(io.imread(f)).convert('RGB')
            except:
                return "ERROR: File doesn't exist"

        # transform, convert to tensor and encode
        im = self.transformer(im).cuda().unsqueeze(0)
        im = self.encoder(im)

        # Make prediction with modified tensor
        caption = self.image_captioner.predict(im, beam_size=beam_size)[
            0].cpu().numpy().astype(int)
        caption = self.coco_vocab.decode(caption, clean=True, join=as_string)

        return caption

    def get_nearest_match(self, caption, gts, dataset='coco', as_string=True):
        """
        Finds nearest gt in given set of gts with caption.
        """

        try:
            assert(dataset == 'coco' or dataset == 'msr-vtt')
        except AssertionError as e:
            print(e)
            return caption, gts

        # use the correct vocabulary
        if dataset == 'coco':
            vocab = self.coco_vocab
        elif dataset == 'msr-vtt':
            vocab = self.msrvtt_vocab

        # get caption tags decoded and encoded using the vocabulary
        gts = gts.apply(lambda x: vocab.encode(x, self.max_caption_length + 1))
        gts = gts.apply(lambda x: vocab.decode(x, clean=True))

        # Determine the nearest best batch
        nearest_gt = None
        best_score = 0.0
        for gt in gts:
            bleu = vocab.evaluate([gt], caption)
            if bleu > best_score:
                best_score = bleu
                nearest_gt = gt

        # Convert to string if requested
        if as_string:
            gt = ' '.join(nearest_gt).capitalize()
            caption = ' '.join(caption).capitalize()
        else:
            gt = nearest_gt

        return caption, gt

    def gen_audio_file(self, text, path):
        """Generate speech audio for given text at path."""
        self.tts.generate_audio(text, path)
        return

    def find_scene_changes(self, video_path, method='threshold',
                           new_stat_file=True):
        """
        Detect scene changes in given video.

        Args:
            video_path: Path to video to analyze
            method: Method for detecting scene changes
            new_stat_file: Option to save results

        Returns:
            Scene changes + their corresponding time codes

        """
        # type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]

        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()

        # Construct our SceneManager and pass it our StatsManager.
        scene_manager = SceneManager(stats_manager)

        # Add ContentDetector algorithm (each detector's constructor
        # takes detector options, e.g. thresholsd).
        if method == 'content':
            scene_manager.add_detector(
                ContentDetector(threshold=30, min_scene_len=40))
        else:
            scene_manager.add_detector(ThresholdDetector(
                min_scene_len=40, threshold=125, min_percent=0.5))

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
