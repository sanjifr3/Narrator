"""
COCO Dataset Torch Dataloader and Dataset implementions

get_image_dataloader creates a dataloader with a 
new coco dataset with the specified parameters
"""
from __future__ import print_function
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

def get_image_dataloader(mode='train',
                        coco_set=2014,
                        images_path=os.environ['HOME'] + '/Database/coco/images/',
                        vocab_path = 'data/processed/coco_vocab.pkl',
                        captions_path = 'data/processed/coco_captions.csv',
                        batch_size = 32,
                        max_len = 30, 
                        embedding_size = 2048, 
                        num_captions = 5,
                        load_features = False,
                        load_captions = False,
                        preload = False,
                        model = 'resnet152',
                        num_workers=0):
    """
    Generate a dataload with the specified parameters

    Params:
    @mode: Dataset type to load
    @coco_set: COCO dataset year to load
    @images_path: Path to COCO dataset images
    @vocab_path: Path to COCO vocab file
    @caption_size: Path to captions vocab file
    @batch_size: Batch size for Dataloader
    @max_len: Max caption length
    @embedding_size: Size of image embedding
    @num_captions: Number of captions per image in dataset
    @load_features: Boolean for creating or loading image features
    @load_captions: Boolean for creating or loading image captions
    @preload: Boolean for either preloading data into RAM during construction
    @model: base model for encoderCNN
    @num_workers: Dataloader parameter

    Return:
    @data_loader: A torch dataloader for the specified coco dataset
    """
    # Ensure that specified mode is valid
    try:
        assert(mode in ['train','val','test'])
        assert(coco_set in [2014,2017])
        assert(os.path.exists(images_path))
        assert(os.path.exists(vocab_path))
        assert(os.path.exists(caption_path))
    except AssertionError:
        # Defaulting conditions
        if mode not in ['train','val','test']:
            print ("Invalid mode specified: {}. Defaulting to val mode".format(mode))
            mode = 'val'
        if coco_set not in [2014,2017]:
            print ("Invalid coco year specified: {}. Defaulting to 2014".format(coco_set))
            year = 2014

        # Terminating conditions
        if not os.path.exists(images_path):
            print (images_path + " does not exist!")
            return None
        elif not os.path.exists(vocab_path):
            print (vocab_path + " does not exist!")
        elif not os.path.exists(caption_path):
            print (caption_path + " does not exist!")

    # Generate dataset
    data = ImageDataset(mode, coco_set, images_path, vocab_path, 
                        captions_path, batch_size, max_len, 
                        embedding_size,num_captions, load_features, 
                        load_captions, preload, model)
    
    # Create a dataloader -- only randomly sample when
    # training
    if mode == 'train':
        # Get all possible image indices
        indices = data.get_indices()

        # Initialize a sampler for the indices
        init_sampler = sampler.SubsetRandomSampler(
                                            indices=indices)

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
    """
    Resets sampler for dataloader

    Params:
    @dataloader: Makes dataloader

    Returns:
    @dataloader: Returns dataloader
    """
    indices = data_loader.dataset.get_indices()
    new_sampler = sampler.SubsetRandomSampler(indices=indices)
    data_loader.batch_sampler.sampler = new_sampler

class ImageDataset(Dataset):
    """
    COCO Torch Dataset (inherits from torch.utils.data.Dataset)
    """
    def get_vocab_size(self):
        """
        Returns the size of the attached vocabulary
        """
        return len(self.vocab)
    def get_vocab(self):
        """
        Returns the vocab idx to word dictionary
        """
        return self.vocab.idx2word
    def get_idx(self):
        """
        Returns the word to idx dictionary
        """
        return self.vocab.word2idx
    def get_seq_len(self):
        """
        Determines and returns the total number of batches
        per epoch

        Returns
        @num_batches: Number of batches per epoch
        """
        num_batches = int(np.floor(len(self.files)/float(self.batch_size)))
        if len(self.files) % self.batch_size != 0:
            return num_batches + 1
        else:
            return num_batches
    def get_indices(self):
        """
        Returns idxs of all image files

        Returns:
        @
        """
        return np.arange(0, len(self.files)).tolist()
    def get_references(self, ids):
        # return [self.df[self.df['id'] == idx]['embedded_caption'].apply(lambda x: self.vocab.decode(x, clean=True)).values.tolist() for idx in ids]
        return [self.df[self.df['id'] == idx]['decoded_caption'].values.tolist() for idx in ids]
    def __init__(self, 
                    mode = 'train',
                    coco_set = 2014,
                    images_path=os.environ['HOME'] + '/Database/coco/images/',
                    vocab_path = 'data/processed/coco_vocab.pkl',
                    captions_path = 'data/processed/coco_captions.csv',
                    batch_size = 32,
                    max_len = 35,
                    embedding_size = 2048,
                    num_captions = 5,
                    load_features = False,
                    load_captions = False,
                    preload = False,
                    model = 'resnet152'):
        """
        Returns idxs of all image files

        Params:
        @mode: Dataset type to load
        @coco_set: COCO dataset year to load
        @images_path: Path to COCO dataset images
        @vocab_path: Path to COCO vocab file
        @caption_size: Path to captions vocab file
        @batch_size: Batch size for Dataloader
        @max_len: Max caption length
        @embedding_size: Size of image embedding
        @num_captions: Number of captions per image in dataset
        @load_features: Boolean for creating or loading image features
        @load_captions: Boolean for creating or loading image captions
        @preload: Boolean for either preloading data into RAM during construction
        @model: base model for encoderCNN
        """
        
        super(ImageDataset, self).__init__()

        try:
            assert(mode in ['train','val','test'])
        except AssertionError:
            print ("Invalid mode specified: {}".format(mode))
        
        # Make class variables
        self.mode = mode
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_captions = num_captions
        self.images_path = images_path
        self.load_features = load_features
        self.preload = preload
        self.model = model

        if not load_features:
            self.transformer = create_transformer()
            self.encoder = EncoderCNN(model)
        
            # Move to gpu if available
            if torch.cuda.is_available():
                self.encoder.cuda()
            
            # Set encoder to evaluation mode
            self.encoder.eval()
        
        # Load in vocabulary
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # Read in captions and select relevant captions
        self.df = pd.read_csv(captions_path)
        self.df = self.df[self.df['set'] == mode + str(coco_set)]
        
        # Shuffle dataframe
        if self.mode == 'val':
            self.df = self.df.sample(frac=.5).reset_index(drop=True)
        
        # Encode captions into a fixed length embedding, drop any captions > max_len
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

        #self.df.to_csv(captions_path,index=False)
        if self.load_features:
            self.df['filename'] = self.df['filename'].apply(lambda x: x.replace('.jpg', '_' + model + '.pkl'))
        
        #print(len(self.df))
        #print
        self.df = self.df[self.df['filename'] != 'train2014/COCO_train2014_000000167126_' + model + '.pkl']
        #print(len(self.df))
        
        self.files = self.df['id'].unique()

        # Preload features
        if self.preload and self.load_features:
            self.im_embeddings = torch.empty(len(self.files), 1, embedding_size)
            self.cap_embeddings = torch.empty(len(self.files), num_captions, max_len)

            for i, im_id in enumerate(self.files):
                with open(self.images_path + self.df[self.df['id'] == im_id]['filename'].values[0],'rb') as f:
                    self.im_embeddings[i] = pickle.load(f)

                # Get captions for image
                cap_embeddings = self.df[self.df['id'] == im_id]['embedded_caption'].values.tolist()
                
                # Sample randomly or crop to get num_captions captions
                while len(cap_embeddings) < num_captions:
                    cap_embeddings.append(cap_embeddings[np.random.randint(0,len(cap_embeddings))])
                if len(cap_embeddings) > num_captions:
                    cap_embeddings = cap_embeddings[:num_captions]

                # Append to torch tensor
                self.cap_embeddings[i] = torch.Tensor(np.vstack(cap_embeddings)).long()
        else:
            self.preload = False # Prevent preloading if not loading features  

    def __getitem__(self, ix):
        im_id = self.files[ix]

        if self.preload:
            # Select random caption index
            cap_ix = np.random.randint(0,self.num_captions)
            if self.mode == 'train':
                return im_id, self.im_embeddings[ix], self.cap_embeddings[ix,cap_ix].long()
            else:
                return im_id, self.im_embeddings[ix], self.cap_embeddings[ix].long()
        else:
            # Load, preprocess image, and extract features with CNN
            if self.load_features:
                try:
                    with open(self.images_path + self.df[self.df['id'] == im_id]['filename'].values[0],'rb') as f:
                        im = pickle.load(f)
                except FileNotFoundError as e:
                    print (e)
                    print (im_id)
            else:
                # Load and process image
                im = PIL.Image.open(self.images_path + self.df[self.df['id'] == im_id]['filename'].values[0]).convert('RGB')
                if torch.cuda.is_available():
                    im = self.transformer(im).cuda().unsqueeze(0)
                else:
                    im = self.transformer(im).unsqueeze(0)
                im = self.encoder(im)

            captions = self.df[self.df['id'] == im_id]['embedded_caption'].values

            if self.mode == 'train':
                # Randomly select caption
                cap_ix = np.random.randint(0, len(captions))
                return im_id, im, torch.Tensor(captions[cap_ix]).long()

            else:
                captions = captions.tolist()
                while len(captions) < self.num_captions:
                    captions.append(captions[np.random.randint(0,len(captions))])
                if len(captions) > self.num_captions:
                    captions = captions[:self.num_captions] 

                return vid_id, vid_array, torch.Tensor(np.vstack(captions)).long()
        
    def __len__(self):
        return len(self.files)