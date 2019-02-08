import os
import sys
import ast
import pickle
from random import shuffle
import numpy as np
import pandas as pd
import cv2
import PIL
import nltk
import torch
from torch.utils.data import Dataset, sampler, DataLoader

from create_transformer import create_transformer
from Vocabulary import Vocabulary

DIR_NAME = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/../')

from models.EncoderCNN import EncoderCNN

def get_video_dataloader(mode = 'train',
                       videos_path = os.environ['HOME'] + '/Database/MSR-VTT/train-video/',
                       vocab_path = 'data/processed/msrvtt_vocab.pkl',
                       captions_path = 'data/processed/msrvtt_captions.csv',
                       batch_size = 32,
                       num_frames = 40,
                       max_len = 30,
                       embedding_size = 2048,
                       num_captions = 20,
                       load_features = False,
                       load_captions = False,
                       preload = False,
                       model = 'resnet152',
                       num_workers=0):

    try:
        assert(mode in ['train','dev','test'])
    except AssertionError:
        print ("Invalid mode specified: {}".format(mode))

    data = VideoDataset(mode, videos_path, vocab_path, captions_path,
                        batch_size, num_frames, max_len,
                        embedding_size, num_captions, load_features, 
                        load_captions, preload, model)

    if mode == "train":
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
    indices = data_loader.dataset.get_indices()
    new_sampler = sampler.SubsetRandomSampler(indices=indices)
    data_loader.batch_sampler.sampler = new_sampler

class VideoDataset(Dataset):
    def get_vocab_size(self):
        return len(self.vocab)
    def get_vocab(self):
        return self.vocab.idx2word
    def get_idx(self):
        return self.vocab.word2idx
    def get_seq_len(self):
        num_batches = int(np.floor(len(self.files)/float(self.batch_size)))
        if len(self.files) % self.batch_size != 0:
            return num_batches + 1
        else:
            return num_batches
    def get_indices(self):
        return np.arange(0, len(self.files)).tolist()
    def get_references(self, ids):
        # return [self.df[self.df['id'] == idx]['embedded_caption'].apply(lambda x: self.vocab.decode(x, clean=True)).values.tolist() for idx in ids]
        return [self.df[self.df['vid_id'] == idx]['decoded_caption'].values.tolist() for idx in ids]
    def __init__(self,
                   mode = 'train',
                   videos_path = os.environ['HOME'] + '/Database/MSR-VTT/train-video/',
                   vocab_path = 'data/processed/msrvtt_vocab.pkl',
                   captions_path = 'data/processed/msrvtt_captions.csv',
                   batch_size = 32,
                   num_frames = 40,
                   max_len = 30,
                   embedding_size = 2048,
                   num_captions = 20,
                   load_features = True,
                   load_captions = True,
                   preload = False,
                   model = 'resnet152'):
        super(VideoDataset, self).__init__()

        try:
            assert(mode in ['train','dev','test'])
        except:
            print("Invalid mode specified: {}".format(mode))
            print("Defaulting to train mode")
            mode = 'train'

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
        #self.df = self.df[:1000]
        # print ('Samples in dataframe: ', len(self.df))

        # Load or encode captions
        if not load_captions or 'embedded_caption' not in self.df.columns.values:
            self.df['embedded_caption'] = self.df['caption'].apply(lambda x: self.vocab.encode(x, max_len+1))
            self.df = self.df[self.df['embedded_caption'].apply(lambda x: x[-1]) == self.vocab(self.vocab.pad_word)]
            self.df['embedded_caption'] = self.df['embedded_caption'].apply(lambda x: x[:-1])
            self.df['decoded_caption'] = self.df['embedded_caption'].apply(lambda x: self.vocab.decode(x, clean=True))
        else:
            self.df['embedded_caption'] = self.df['embedded_caption'].apply(ast.literal_eval)
            self.df['decoded_caption'] = self.df['decoded_caption'].apply(ast.literal_eval)
            self.df = self.df[self.df['embedded_caption'].apply(lambda x: x[max_len]) == self.vocab(self.vocab.pad_word)]
            self.df['embedded_caption'] = self.df['embedded_caption'].apply(lambda x: x[:max_len])
        
        self.files = self.df['vid_id'].unique()
        # print ('Unique videos: ', len(self.df))

        # Preload features
        if self.preload and self.load_features:
            # Create empty tensors to fill
            self.vid_embeddings = torch.empty(len(self.files), num_frames, embedding_size)
            self.cap_embeddings = torch.empty(len(self.files), num_captions, max_len)

            # Loop through unique video ids
            for i, vid_id in enumerate(self.files):

                # Load an store video feature
                with open(self.videos_path + vid_id + '_' + model + '.pkl','rb') as f:
                   self.vid_embeddings[i] = pickle.load(f)

                # Get captions for video
                cap_embeddings = self.df[self.df['vid_id'] == vid_id]['embedded_caption'].values.tolist()
                # Randomly sampole or crop to get num_caption captions
                while len(cap_embeddings) < num_captions:
                    cap_embeddings.append(cap_embeddings[np.random.randint(0,len(cap_embeddings))])
                if len(cap_embeddings) > num_captions:
                    cap_embeddings = cap_embeddings[:num_captions]

                # Append to torch tensor
                self.cap_embeddings[i] = torch.Tensor(np.vstack(cap_embeddings)).long()
        else:
            self.preload = False # Prevent preloading if not loading features  

    def __getitem__(self,ix):
        vid_id = self.files[ix]

        # Load, preprocess video, and extract features with CNN
        if self.preload:
            # Select random caption index
            cap_ix = np.random.randint(0, self.num_captions)
            if self.mode == 'train':
                return vid_id, self.vid_embeddings[ix], self.cap_embeddings[ix, cap_ix].long()
            else:
                return vid_id, self.vid_embeddings[ix], self.cap_embeddings[ix].long()
        else:
            if self.load_features: 
                with open(self.videos_path + vid_id + '_' + self.model + '.pkl','rb') as f:
                    vid_array = pickle.load(f)
            else:
                vid_array = self.getVidArray(self.videos_path + vid_id + '.mp4')
                vid_array = self.encoder(vid_array)

            captions = self.df[self.df['vid_id'] == vid_id]['embedded_caption'].values

            if self.mode == 'train':
                # Randomly select caption
                cap_ix = np.random.randint(0, len(captions))
                return vid_id, vid_array, torch.Tensor(captions[cap_ix]).long()
            else:
                captions = captions.tolist()
                while len(captions) < self.num_captions:
                    captions.append(captions[np.random.randint(0,len(captions))])
                if len(captions) > self.num_captions:
                    captions = captions[:self.num_captions] 

                return vid_id, vid_array, torch.Tensor(np.vstack(captions)).long()

    def __len__(self):
        return len(self.files)

    def getVidArray(self, video_name):
        try:
            cap = cv2.VideoCapture(video_name)
        except:
            print ("Could not open %s" % (video_name))
            return None

        vid_array = torch.zeros(self.num_frames, 3, 224, 224)
        if torch.cuda.is_available():
            vid_array = vid_array.cuda()

        frame_idx = 0

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
                print ("Could not process frame in " + video_name)

        cap.release()
        return vid_array    