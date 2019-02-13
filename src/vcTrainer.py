from __future__ import print_function
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable

from utils.create_transformer import create_transformer
from utils.Vocabulary import Vocabulary
from utils.VideoDataloader import get_video_dataloader, VideoDataset

from models.VideoCaptioner import VideoCaptioner

lr = 0.0001
val_interval = 15
save_int = 100
num_epochs = 10000
beam_size = 5

initial_checkpoint_file = None  # 'video_caption-model10-45-0.3175-5.0.pkl'
version = 12
# 8: Pre-training w/ COCO
# 10: VGG16 w/ LSTM and beam search
# 11: Resnet152 w/ GRU w/ beam search
# 12: Resnet152 w/ LSTM (embed_size: 512) w/ beam search

videos_path = os.environ['HOME'] + '/Database/MSR-VTT/train-video/'
vocab_path = 'data/processed/msrvtt_vocab.pkl'
captions_path = 'data/processed/msrvtt_captions.csv'
models_path = 'models/'
base_model = 'vgg16'  # 'resnet152'
batch_size = 32
embedding_size = 25088  # 2048
embed_size = 256
hidden_size = 512
load_features = True
load_captions = True
preload = False

print("Loading training data...\r", end="")
train_loader = get_video_dataloader('train', videos_path,
                                    vocab_path, captions_path,
                                    batch_size,
                                    load_features=load_features,
                                    load_captions=load_captions,
                                    preload=preload,
                                    model=base_model,
                                    embedding_size=embedding_size,
                                    num_workers=0)
train_loader.dataset.mode = 'train'
print("Loading training data...Done")
print("Loading validation data...\r", end="")
val_loader = get_video_dataloader('dev', videos_path,
                                  vocab_path, captions_path,
                                  batch_size,
                                  load_features=load_features,
                                  load_captions=load_captions,
                                  preload=preload,
                                  model=base_model,
                                  embedding_size=embedding_size,
                                  num_workers=0)
val_loader.dataset.mode = 'dev'
print("Loading validation data...Done")

vocab_size = train_loader.dataset.get_vocab_size()
start_id = train_loader.dataset.get_idx(
)[train_loader.dataset.vocab.start_word]
end_id = train_loader.dataset.get_idx()[train_loader.dataset.vocab.end_word]
max_caption_length = train_loader.dataset.max_len

captioner = VideoCaptioner(embedding_size, embed_size,
                           hidden_size, vocab_size,
                           max_caption_length,
                           start_id, end_id)

if torch.cuda.is_available():
    captioner.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
optimizer = torch.optim.Adam(params=captioner.parameters(), lr=lr)

train_losses = []
val_losses = []
val_bleus = []

best_val_bleu = -1000.0
start_time = time.time()

if initial_checkpoint_file:
    checkpoint = torch.load(os.path.join(models_path, initial_checkpoint_file))
    captioner.load_state_dict(checkpoint['params'])
    optimizer.load_state_dict(checkpoint['optim_params'])
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    val_bleus = checkpoint['val_bleus']
    best_val_bleu = np.array(val_bleus).max()
    starting_epoch = checkpoint['epoch']
else:
    starting_epoch = 0

for epoch in range(starting_epoch, num_epochs):
    print('Epoch: [{}/{}]'.format(epoch, num_epochs))
    captioner.train()
    epoch_start_time = time.time()
    train_loss = 0.0

    # Loop through batches
    for train_id, batch in enumerate(train_loader):
        batch_start_time = time.time()
        _, vid_embeddings, caption_embeddings = batch

        if torch.cuda.is_available():
            vid_embeddings = vid_embeddings.cuda()
            caption_embeddings = caption_embeddings.cuda()

        # Forward propagate
        probs = captioner(vid_embeddings, caption_embeddings)

        # Calculate loss, and backpropagate
        loss = criterion(probs.view(-1, vocab_size),
                         caption_embeddings.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute loss
        train_loss += loss.item()

        # Get training statistics
        stats = "Epoch %d, Train step [%d/%d], %ds, Loss: %.4f" \
                % (epoch, train_id, train_loader.dataset.get_seq_len(), time.time() - batch_start_time, loss.item())
        print("\r" + stats, end="")
        sys.stdout.flush()

        if train_id % 100 == 0:
            print("\r" + stats)

    sys.stdout.flush()
    print('\n')
    train_losses.append(train_loss / train_loader.dataset.get_seq_len())

    if epoch > 0 and epoch % val_interval == 0:
        val_loss = 0.0
        val_bleu = 0.0
        captioner.eval()

        for val_id, val_batch in enumerate(val_loader):
            batch_start_time = time.time()
            idxs, vid_embeddings, caption_embeddings = val_batch

            if torch.cuda.is_available():
                vid_embeddings = vid_embeddings.cuda()
                caption_embeddings = caption_embeddings.cuda()

            # Get ground truth captions
            refs = val_loader.dataset.get_references(idxs)

            if not beam_size:
                preds, probs = captioner.predict(
                    vid_embeddings, True, beam_size=beam_size)

                # Get loss and update val loss
                losses = torch.ones(val_loader.dataset.num_captions)
                for i in range(val_loader.dataset.num_captions):
                    losses[i] = criterion(
                        probs.view(-1, vocab_size), caption_embeddings[:, i].contiguous().view(-1))
                #loss = losses.min()
                #loss = criterion(probs.view(-1, vocab_size), caption_embeddings.view(-1))
                val_loss += losses.min().item()
            else:
                preds = captioner.predict(vid_embeddings, beam_size=beam_size)
                val_loss += 5

            # Calculate bleu loss per sample in batch
            # Sum and add length normalized sum to val_loss
            batch_bleu = 0.0
            for pred_id in range(len(preds)):
                pred = preds[pred_id].cpu().numpy().astype(int)
                pred_embed = val_loader.dataset.vocab.decode(pred, clean=True)
                batch_bleu += val_loader.dataset.vocab.evaluate(
                    refs[pred_id], pred_embed)
            val_bleu += (batch_bleu / len(preds))

            # Get training statistics
            stats = "Epoch %d, Validation step [%d/%d], %ds, Loss: %.4f, Bleu: %.4f" \
                    % (epoch, val_id, val_loader.dataset.get_seq_len(),
                        time.time() - batch_start_time, loss.item(), batch_bleu / len(preds))

            print("\r" + stats, end="")
            sys.stdout.flush()

            if val_id % 100 == 0:
                print('\r' + stats)

        val_losses.append(val_loss / val_loader.dataset.get_seq_len())
        val_bleus.append(val_bleu / val_loader.dataset.get_seq_len())

        if val_bleus[-1] > best_val_bleu:
            best_val_bleu = val_bleus[-1]
            print("\nBest model found -- bleu: %.4f, val_loss: %.4f, train_loss: %.4f" %
                  (val_bleus[-1], val_losses[-1], train_losses[-1]))
            filename = os.path.join(models_path, "video_caption-model{}-{}-{}-{}.pkl".format(
                version, epoch, round(val_bleus[-1], 4), round(val_losses[-1], 4)))
            torch.save({'params': captioner.state_dict(),
                        'optim_params': optimizer.state_dict(),
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'val_bleus': val_bleus,
                        'epoch': epoch}, filename)
        else:
            print("\nValidation -- bleu: %.4f, val_loss: %.4f, train_loss: %.4f" %
                  (val_bleus[-1], val_losses[-1], train_losses[-1]))

    if epoch > 0 and epoch % save_int == 0:
        filename = os.path.join(models_path, "video_caption-ckpt-model{}-{}-{}-{}.pkl".format(
            version, epoch, round(val_bleus[-1], 4), round(val_losses[-1], 4)))
        torch.save({'params': captioner.state_dict(),
                    'optim_params': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_bleus': val_bleus,
                    'epoch': epoch}, filename)
