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

        self.tts = TTS()

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.image_captioner.cuda()
            self.video_captioner.cuda()

        self.encoder.eval()
        self.image_captioner.eval()
        self.video_captioner.eval()

    def genCaption(self, f, beam_size=5, as_string=False):
        ext = f.split('.')[-1]
        if ext in self.img_extensions:
            return self.genImCaption(f, beam_size, as_string)
        elif ext in self.vid_extensions:
            return self.genVidCaption(f, beam_size, as_string)
        else:
            return "ERROR: Invalid file type: " + ext

    def genVidCaption(self, f, beam_size=5, as_string=False):
        if not os.path.exists(f):
            return "ERROR: File does not exist!"

        cap = cv2.VideoCapture(f)

        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        num_frames = self.num_frames
        if total_frames < self.num_frames:
            num_frames = total_frames

        vid_array = torch.zeros(num_frames, 3, 224, 224)

        if torch.cuda.is_available():
            vid_array = vid_array.cuda()

        frame_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret or frame_idx == num_frames:
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
                print("Could not process frame in " + f)

        cap.release()

        vid_array = self.encoder(vid_array)
        caption = self.video_captioner.predict(vid_array.unsqueeze(0), beam_size=beam_size)[0].cpu().numpy().astype(int)
        caption = self.msrvtt_vocab.decode(caption, clean=True, join=as_string)
        return caption

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
    











# DIR_NAME = os.path.dirname(os.path.realpath(__file__))

# sys.path.append(DIR_NAME + '/features/')
# from create_transformer import create_transformer
# from Vocabulary import Vocabulary

# sys.path.append(DIR_NAME + '/data/')
# from VideoDataloader import get_video_dataloader, VideoDataset

# sys.path.append(DIR_NAME + '/models/')
# from VideoCaptioner import VideoCaptioner

# lr = 0.0001
# initial_checkpoint_file = None # 'image_caption-model1-460-0.2898-7.1711.pkl'
# val_interval = 15
# save_int = 100
# num_epochs = 10000
# version = 10
# beam_size = 5

# # 8: Pre-training
# # 10: VGG16

# videos_path = os.environ['HOME'] + '/Database/MSR-VTT/train-video/'
# vocab_path  = 'data/processed/msrvtt_vocab.pkl'
# captions_path = 'data/processed/msrvtt_captions.csv'
# models_path = 'models/'
# base_model = 'vgg16' # 'resnet152'
# batch_size = 32
# embedding_size = 25088 # 2048
# embed_size = 256  
# hidden_size = 512
# load_features = True
# load_captions = True
# preload = False

# print ("Loading training data...\r", end="")
# train_loader = get_video_dataloader('train',videos_path, 
#                                   vocab_path, captions_path, 
#                                   batch_size,
#                                   load_features=load_features,
#                                   load_captions=load_captions,
#                                   preload=preload,
#                                   model=base_model,
#                                   embedding_size=embedding_size,
#                                   num_workers=0)
# train_loader.dataset.mode = 'train'
# print ("Loading training data...Done")
# print ("Loading validation data...\r", end="")
# val_loader = get_video_dataloader('dev',videos_path, 
#                                   vocab_path, captions_path, 
#                                   batch_size, 
#                                   load_features=load_features,
#                                   load_captions=load_captions,
#                                   preload=preload,
#                                   model=base_model,
#                                   embedding_size=embedding_size,
#                                   num_workers=0)
# val_loader.dataset.mode = 'dev'
# print ("Loading validation data...Done", end="")

# vocab_size = train_loader.dataset.get_vocab_size()
# start_id = train_loader.dataset.get_idx()[train_loader.dataset.vocab.start_word]
# end_id = train_loader.dataset.get_idx()[train_loader.dataset.vocab.end_word]
# max_caption_length = train_loader.dataset.max_len

# captioner = VideoCaptioner(embedding_size, embed_size, 
#                            hidden_size, vocab_size, 
#                            max_caption_length, 
#                            start_id, end_id)

# if torch.cuda.is_available():
#     captioner.cuda()
#     criterion = nn.CrossEntropyLoss().cuda()
# else:
#     criterion = nn.CrossEntropyLoss()

#     # Define the optimizer
# optimizer = torch.optim.Adam(params=captioner.parameters(), lr=lr)

# train_losses = []
# val_losses = []
# val_bleus = []

# best_val_bleu = -1000.0
# start_time = time.time()

# if initial_checkpoint_file:
#     checkpoint = torch.load(os.path.join(models_path,initial_checkpoint_file))
#     captioner.load_state_dict(checkpoint['params'])
#     optimizer.load_state_dict(checkpoint['optim_params'])
#     train_losses = checkpoint['train_losses']
#     val_losses = checkpoint['val_losses']
#     val_bleus = checkpoint['val_bleus']
#     best_val_bleu = np.array(val_bleus).max()
#     starting_epoch = checkpoint['epoch']
# else:
#     starting_epoch = 0

# for epoch in range(starting_epoch,num_epochs):
#     print ('Epoch: [{}/{}]'.format(epoch,num_epochs))
#     captioner.train()
#     epoch_start_time = time.time()
#     train_loss = 0.0

#     # Loop through batches
#     for train_id, batch in enumerate(train_loader):
#         batch_start_time = time.time()
#         _, vid_embeddings, caption_embeddings = batch
                              
#         if torch.cuda.is_available():
#           vid_embeddings = vid_embeddings.cuda()
#           caption_embeddings = caption_embeddings.cuda()
        
#         # Forward propagate
#         probs = captioner(vid_embeddings,caption_embeddings)
        
#         # Calculate loss, and backpropagate
#         loss = criterion(probs.view(-1, vocab_size), caption_embeddings.view(-1))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # Compute loss
#         train_loss += loss.item()
        
#         # Get training statistics
#         stats = "Epoch %d, Train step [%d/%d], %ds, Loss: %.4f" \
#                 % (epoch, train_id, train_loader.dataset.get_seq_len(), time.time() - batch_start_time, loss.item())
#         print("\r" + stats, end="")
#         sys.stdout.flush()
        
#         if train_id % 100 == 0:
#             print ("\r" + stats)
    
#     sys.stdout.flush()
#     print ('\n')
#     train_losses.append(train_loss/train_loader.dataset.get_seq_len())

#     if epoch > 0 and epoch % val_interval == 0:
#         val_loss = 0.0
#         val_bleu = 0.0
#         captioner.eval()

#         for val_id, val_batch in enumerate(val_loader):
#             batch_start_time = time.time()
#             idxs, vid_embeddings, caption_embeddings = val_batch

#             if torch.cuda.is_available():
#                 vid_embeddings = vid_embeddings.cuda()
#                 caption_embeddings = caption_embeddings.cuda()

#             # Get ground truth captions
#             refs = val_loader.dataset.get_references(idxs)
            
#             if not beam_size:
#                 preds, probs = captioner.predict(vid_embeddings, True, beam_size=beam_size)

#                 # Get loss and update val loss
#                 losses = torch.ones(val_loader.dataset.num_captions)
#                 for i in range(val_loader.dataset.num_captions):
#                     losses[i] = criterion(probs.view(-1, vocab_size), caption_embeddings[:,i].contiguous().view(-1))
#                 #loss = losses.min()
#                 #loss = criterion(probs.view(-1, vocab_size), caption_embeddings.view(-1))
#                 val_loss += losses.min().item()
#             else:
#                 preds = captioner.predict(vid_embeddings, beam_size=beam_size)
#                 val_loss += 5

#             # Calculate bleu loss per sample in batch
#             # Sum and add length normalized sum to val_loss
#             batch_bleu = 0.0
#             for pred_id in range(len(preds)):
#                 pred = preds[pred_id].cpu().numpy().astype(int)
#                 pred_embed = val_loader.dataset.vocab.decode(pred, clean=True)
#                 batch_bleu += val_loader.dataset.vocab.evaluate(refs[pred_id], pred_embed)
#             val_bleu += (batch_bleu/len(preds))

#             # Get training statistics
#             stats = "Epoch %d, Validation step [%d/%d], %ds, Loss: %.4f, Bleu: %.4f" \
#                     % (epoch, val_id, val_loader.dataset.get_seq_len(), 
#                         time.time() - batch_start_time, loss.item(), batch_bleu/len(preds))

#             print("\r" + stats, end="")
#             sys.stdout.flush()

#             if val_id % 100 == 0:
#                 print('\r' + stats)

#         val_losses.append(val_loss/val_loader.dataset.get_seq_len())
#         val_bleus.append(val_bleu/val_loader.dataset.get_seq_len())

#         if val_bleus[-1] > best_val_bleu:
#             best_val_bleu = val_bleus[-1]
#             print ("\nBest model found -- bleu: %.4f, val_loss: %.4f, train_loss: %.4f" % (val_bleus[-1], val_losses[-1], train_losses[-1]))
#             filename = os.path.join(models_path, "video_caption-model{}-{}-{}-{}.pkl".format(version, epoch, round(val_bleus[-1],4), round(val_losses[-1],4)))
#             torch.save({'params':captioner.state_dict(),
#                     'optim_params':optimizer.state_dict(),
#                     'train_losses':train_losses,
#                     'val_losses':val_losses,
#                     'val_bleus':val_bleus,
#                     'epoch': epoch},filename)
#         else:
#             print ("\nValidation -- bleu: %.4f, val_loss: %.4f, train_loss: %.4f" % (val_bleus[-1], val_losses[-1], train_losses[-1]))
        
#     if epoch > 0 and epoch % save_int == 0:
#         filename = os.path.join(models_path, "video_caption-ckpt-model{}-{}-{}-{}.pkl".format(version, epoch, round(val_bleus[-1],4), round(val_losses[-1],4)))
#         torch.save({'params':captioner.state_dict(),
#                     'optim_params':optimizer.state_dict(),
#                     'train_losses':train_losses,
#                     'val_losses':val_losses,
#                     'val_bleus':val_bleus,
#                     'epoch': epoch},filename)