# -*- coding: utf-8 -*-
"""
MSR-VTT Dataset Torch Dataloader and Dataset implemention.

get_video_dataloader creates a dataloader with a
new MSR-VTT dataset with the specified parameters
"""
from __future__ import print_function
import os
import sys
import ast
import pickle
import numpy as np
import pandas as pd
import PIL
import cv2
import torch
from torch.utils.data import Dataset, sampler, DataLoader

DIR_NAME = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/../')
sys.path.append(DIR_NAME)

from create_transformer import create_transformer
from Vocabulary import Vocabulary

from models.EncoderCNN import EncoderCNN


def get_video_dataloader(mode='train',
                         videos_path=os.environ['HOME'] +
                         '/Database/MSR-VTT/train-video/',
                         vocab_path='data/processed/msrvtt_vocab.pkl',
                         captions_path='data/processed/msrvtt_captions.csv',
                         batch_size=32,
                         num_frames=40,
                         max_len=30,
                         embedding_size=2048,
                         num_captions=20,
                         load_features=False,
                         load_captions=False,
                         preload=False,
                         model='resnet152',
                         num_workers=0):
    """
    Generate a dataloader with the specified parameters.

    Args:
        mode: Dataset type to load
        videos_path: Path to MSR-VTT videos dataset
        vocab_path: Path to MSR-VTT vocab file
        caption_size: Path to captions vocab file
        batch_size: Batch size for Dataloader
        num_frames: Number of frames per video to process
        max_len: Max caption length
        embedding_size: Size of image embedding
        num_captions: Number of captions per image in dataset
        load_features: Boolean for creating or loading image features
        load_captions: Boolean for creating or loading image captions
        preload: Boolean for either preloading data
           into RAM during construction
        model: base model for encoderCNN
        num_workers: Dataloader parameter

    Return:
        data_loader: A torch dataloader for the MSR-VTT dataset

    """
    # Ensure specified mode is validate
    try:
        assert mode in ['train', 'dev', 'test']
    except AssertionError:
        print('Invalid mode specified: {}'.format(mode))
        print(' Defaulting to dev mode')
        mode = 'dev'

    # Build dataset
    data = VideoDataset(mode, videos_path, vocab_path, captions_path,
                        batch_size, num_frames, max_len,
                        embedding_size, num_captions, load_features,
                        load_captions, preload, model)

    if mode == 'train':
        # Get all possible video indices
        indices = data.get_indices()

        # Initialize a sampler for the indices
        init_sampler = sampler.SubsetRandomSampler(indices=indices)

        # Create data loader with dataset and sampler
        data_loader = DataLoader(dataset=data,
                                 num_workers=num_workers,
                                 batch_sampler=sampler.BatchSampler(
                                     sampler=init_sampler,
                                     batch_size=batch_size,
                                     drop_last=False))
    else:
        data_loader = DataLoader(dataset=data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
    return data_loader


def reset_dataloader(data_loader):
    """Reset sampler for dataloader."""
    indices = data_loader.dataset.get_indices()
    new_sampler = sampler.SubsetRandomSampler(indices=indices)
    data_loader.batch_sampler.sampler = new_sampler


class VideoDataset(Dataset):
    """MSR-VTT Torch Dataset (inherits from torch.utils.data.Dataset)."""

    def get_vocab_size(self):
        """Returns the size of the attached vocabulary."""
        return len(self.vocab)

    def get_vocab(self):
        """Returns the vocab idx to word dictionary."""
        return self.vocab.idx2word

    def get_idx(self):
        """Returns the word to idx dictionary."""
        return self.vocab.word2idx

    def get_seq_len(self):
        """
        Determines and returns the total number of batches per epoch.

        Returns:
            The number of batches per epoch.
        """
        num_batches = int(np.floor(len(self.files) / float(self.batch_size)))
        if len(self.files) % self.batch_size != 0:
            return num_batches + 1

        return num_batches

    def get_indices(self):
        """Returns idxs for all video files."""
        return np.arange(0, len(self.files)).tolist()

    def get_references(self, ids):
        """Get all captions for given ids."""
        return [self.df[self.df['vid_id'] == idx]
                ['decoded_caption'].values.tolist() for idx in ids]

    def __init__(self,
                 mode='train',
                 videos_path=os.environ['HOME'] +
                 '/Database/MSR-VTT/train-video/',
                 vocab_path='data/processed/msrvtt_vocab.pkl',
                 captions_path='data/processed/msrvtt_captions.csv',
                 batch_size=32,
                 num_frames=40,
                 max_len=30,
                 embedding_size=2048,
                 num_captions=20,
                 load_features=True,
                 load_captions=True,
                 preload=False,
                 model='resnet152'):
        """
        Construct the VideoDataset class.

        Args:
            mode: Dataset type to load
            videos_path: Path to MSR-VTT videos dataset
            vocab_path: Path to MSR-VTT vocab file
            caption_size: Path to captions vocab file
            batch_size: Batch size for Dataloader
            num_frames: Number of frames per video to process
            max_len: Max caption length
            embedding_size: Size of image embedding
            num_captions: Number of captions per image in dataset
            load_features: Boolean for creating or loading image features
            load_captions: Boolean for creating or loading image captions
            preload: Boolean for either preloading data
               into RAM during construction
            model: base model for encoderCNN

        """
        super(VideoDataset, self).__init__()

        try:
            assert(mode in ['train', 'dev', 'val', 'test'])
        except:
            print("Invalid mode specified: {}".format(mode))
            print("Defaulting to train mode")
            mode = 'train'

        # Make val synonymous with dev
        if mode == 'val':
            mode = 'dev'

        # Declare class variables
        self.mode = mode
        self.num_frames = num_frames
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_captions = num_captions
        self.videos_path = videos_path
        self.load_features = load_features
        self.preload = preload
        self.model = model

        if not self.load_features:
            self.transformer = create_transformer()
            self.encoder = EncoderCNN(model)

            # Move to gpu if available
            if torch.cuda.is_available():
                self.encoder.cuda()

            # Set encoder in evaluation mode
            self.encoder.eval()

        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # Read in captions dataframe
        self.df = pd.read_csv(captions_path)
        self.df = self.df[self.df['set'] == mode]

        # Load or encode captions into fixed length embeddings, drop
        # any captions with length greater than max_len
        if (not load_captions or
                'embedded_caption' not in self.df.columns.values):
            self.df['embedded_caption'] = self.df['caption'].apply(
                lambda x: self.vocab.encode(x, max_len + 1))
            self.df = self.df[self.df['embedded_caption'].apply(
                lambda x: x[-1]) == self.vocab(self.vocab.pad_word)]
            self.df['embedded_caption'] = self.df[
                'embedded_caption'].apply(lambda x: x[:-1])
            self.df['decoded_caption'] = self.df['embedded_caption'].apply(
                lambda x: self.vocab.decode(x, clean=True))
        else:
            self.df['embedded_caption'] = self.df[
                'embedded_caption'].apply(ast.literal_eval)
            self.df['decoded_caption'] = self.df[
                'decoded_caption'].apply(ast.literal_eval)
            self.df = self.df[
                self.df['embedded_caption'].apply(
                    lambda x: x[max_len]) == self.vocab(
                        self.vocab.pad_word)]
            self.df['embedded_caption'] = self.df[
                'embedded_caption'].apply(lambda x: x[:max_len])

        self.files = self.df['vid_id'].unique()

        # Preload features
        if self.preload and self.load_features:
            # Create empty tensors to fill
            self.vid_embeddings = torch.empty(
                len(self.files), num_frames, embedding_size)
            self.cap_embeddings = torch.empty(
                len(self.files), num_captions, max_len)

            # Loop through unique video ids
            for i, vid_id in enumerate(self.files):

                # Load an store video feature
                with open(self.videos_path + vid_id + '_' +
                          model + '.pkl', 'rb') as f:
                    self.vid_embeddings[i] = pickle.load(f)

                # Get captions for video
                cap_embeddings = self.df[self.df['vid_id'] == vid_id][
                    'embedded_caption'].values.tolist()
                # Randomly sampole or crop to get num_caption captions
                while len(cap_embeddings) < num_captions:
                    cap_embeddings.append(
                        cap_embeddings[
                            np.random.randint(
                                0, len(cap_embeddings))])
                if len(cap_embeddings) > num_captions:
                    cap_embeddings = cap_embeddings[:num_captions]

                # Append to torch tensor
                self.cap_embeddings[i] = torch.Tensor(
                    np.vstack(cap_embeddings)).long()
        else:
            self.preload = False  # Prevent preloading if not loading features

    def __getitem__(self, ix):
        """
        Returns video id, video embedding, and captions for given \
            index.

        If in training mode, return a random caption sample.
        Otherwise, return all captions for a given ix.

        Args:
            ix: Batch index
        """
        vid_id = self.files[ix]

        # Load preprocessed videos/captions from memory
        if self.preload:
            # Select random caption index
            cap_ix = np.random.randint(0, self.num_captions)
            if self.mode == 'train':
                return vid_id, self.vid_embeddings[
                    ix], self.cap_embeddings[ix, cap_ix].long()
            return vid_id, self.vid_embeddings[
                ix], self.cap_embeddings[ix].long()

        # Load features from file
        if self.load_features:
            with open(self.videos_path +
                      vid_id + '_' + self.model + '.pkl', 'rb') as f:
                vid_array = pickle.load(f)

        # Generate features from raw video
        else:
            vid_array = self.get_vid_array(
                self.videos_path + vid_id + '.mp4')
            vid_array = self.encoder(vid_array)

        # Grab captions related to video from dataframe
        captions = self.df[self.df['vid_id'] == vid_id][
            'embedded_caption'].values

        if self.mode == 'train':
            # Randomly select caption
            cap_ix = np.random.randint(0, len(captions))
            return (vid_id, vid_array,
                    torch.Tensor(captions[cap_ix]).long())

        # Select all captions for video and randomly sample
        # to fixed length
        captions = captions.tolist()
        while len(captions) < self.num_captions:
            captions.append(
                captions[
                    np.random.randint(
                        0, len(captions))])
        if len(captions) > self.num_captions:
            captions = captions[:self.num_captions]

        return vid_id, vid_array, torch.Tensor(
            np.vstack(captions)).long()

    def __len__(self):
        """Get number of videos."""
        return len(self.files)

    def get_vid_array(self, video_name):
        """
        Read in video and create a torch array from \
            (num_frames, 3, 224, 224).

        Args:
            video_name: Path to video

        Returns:
            A torch tensor of frame encodings
        """
        try:
            cap = cv2.VideoCapture(video_name)
        except:
            print('Could not open %s' % (video_name))
            return None

        # Make empty arrays to store results in
        vid_array = torch.zeros(self.num_frames, 3, 224, 224)
        if torch.cuda.is_available():
            vid_array = vid_array.cuda()

        frame_idx = 0

        # Loop through and append frames to torch array
        while True:
            ret, frame = cap.read()

            if not ret or frame_idx == self.num_frames:
                break

            try:
                frame = PIL.Image.fromarray(frame).convert('RGB')

                if torch.cuda.is_available():
                    frame = self.transformer(frame).cuda().unsqueeze(0)
                else:
                    frame = self.transformer(frame).unsqueeze(0)

                vid_array[frame_idx] = frame
                frame_idx += 1
            except OSError as e:
                print(e + ' Could not process frame in ' + video_name)

        cap.release()
        return vid_array
