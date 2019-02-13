# -*- coding: utf-8 -*-
"""
Module for training the Image Captioner class.

Model params are set at top of file.
"""
from __future__ import print_function
import sys
import time
import os
import numpy as np
import torch
import torch.nn as nn
import argparse

from utils.Vocabulary import Vocabulary
from utils.ImageDataloader import get_image_dataloader, ImageDataset
from models.ImageCaptioner import ImageCaptioner

lr = 0.001
val_interval = 10
save_int = 10
num_epochs = 1000
beam_size = 3

initial_checkpoint_file = None  # 'image_caption-model2-10-0.1863-4.3578.pkl'
version = 11
# version < 2: validation loss is invalid
# version 3: lowest validiation loss
# version 4: beam_size 3
# version 10: w/ VGG16
# 11: Resnet152 w/ GRU w/o beam search

images_path = os.environ['HOME'] + '/Database/coco/images/'
vocab_path = 'data/processed/coco_vocab.pkl'
captions_path = os.environ['HOME'] + \
    '/programs/cocoapi/annotations/coco_captions.csv'
models_path = 'models/'
batch_size = 64
coco_set = 2014
load_features = True
load_captions = True
preload = True
base_model = 'resnet152'
embedding_size = 2048
embed_size = 256
hidden_size = 512
rnn_type = 'lstm'

print("Loading training data...\r", end="")
train_loader = get_image_dataloader('train', coco_set,
                                    images_path,
                                    vocab_path, captions_path,
                                    batch_size,
                                    embedding_size=embedding_size,
                                    load_features=load_features,
                                    load_captions=load_captions,
                                    model=base_model,
                                    preload=preload)
train_loader.dataset.mode = 'train'
print("Loading training data...Done")
print("Loading validation data...\r", end="")
val_loader = get_image_dataloader('val', coco_set,
                                  images_path,
                                  vocab_path, captions_path,
                                  batch_size,
                                  embedding_size=embedding_size,
                                  load_features=load_features,
                                  load_captions=load_captions,
                                  model=base_model,
                                  preload=preload)
val_loader.dataset.mode = 'val'
print("Loading validation data...Done")

vocab_size = train_loader.dataset.get_vocab_size()
start_id = train_loader.dataset.get_idx(
)[train_loader.dataset.vocab.start_word]
end_id = train_loader.dataset.get_idx()[train_loader.dataset.vocab.end_word]
max_caption_length = train_loader.dataset.max_len

captioner = ImageCaptioner(embedding_size, embed_size,
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
        _, im_embeddings, caption_embeddings = batch

        if torch.cuda.is_available():
            im_embeddings = im_embeddings.cuda()
            caption_embeddings = caption_embeddings.cuda()

        # Forward propagate
        probs = captioner(im_embeddings, caption_embeddings)

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
                % (epoch, train_id, train_loader.dataset.get_seq_len(),
                   time.time() - batch_start_time, loss.item())
        print("\r" + stats, end="")
        sys.stdout.flush()

        if train_id % 250 == 0:
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
            idxs, im_embeddings, caption_embeddings = val_batch

            if torch.cuda.is_available():
                im_embeddings = im_embeddings.cuda()
                caption_embeddings = caption_embeddings.cuda()

            # Get ground truth captions
            refs = val_loader.dataset.get_references(idxs.numpy())

            if not beam_size:
                preds, probs = captioner.predict(im_embeddings, True)

                # Get loss and update val loss
                losses = torch.ones(val_loader.dataset.num_captions)
                for i in range(val_loader.dataset.num_captions):
                    losses[i] = criterion(
                        probs.view(-1, vocab_size),
                        caption_embeddings[:, i].contiguous().view(-1))

                # loss = criterion(probs.view(-1, vocab_size),
                #                  caption_embeddings.view(-1))
                val_loss += losses.min().item()
            else:
                preds = captioner.predict(im_embeddings, beam_size=beam_size)
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
            stats = "Epoch %d, Validation step [%d/%d], \
                     %ds, Loss: %.4f, Bleu: %.4f" \
                    % (epoch, val_id, val_loader.dataset.get_seq_len(),
                       time.time() - batch_start_time, loss.item(),
                       batch_bleu / len(preds))

            print("\r" + stats, end="")
            sys.stdout.flush()

            if val_id % 250 == 0:
                print('\r' + stats)

        val_losses.append(val_loss / val_loader.dataset.get_seq_len())
        val_bleus.append(val_bleu / val_loader.dataset.get_seq_len())

        if val_bleus[-1] > best_val_bleu:
            best_val_bleu = val_bleus[-1]
            print("\nBest model found -- bleu: %.4f, \
                   val_loss: %.4f, train_loss: %.4f" %
                  (val_bleus[-1], val_losses[-1], train_losses[-1]))

            filename = os.path.join(
                models_path, "image_caption-model{}-{}-{}-{}.pkl".\
                format(version, epoch, round(val_bleus[-1], 4),
                       round(val_losses[-1], 4)))

            torch.save({'params': captioner.state_dict(),
                        'optim_params': optimizer.state_dict(),
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'val_bleus': val_bleus,
                        'epoch': epoch}, filename)
        else:
            print("\nValidation -- bleu: %.4f, \
                  val_loss: %.4f, train_loss: %.4f" %
                  (val_bleus[-1], val_losses[-1], train_losses[-1]))

    if epoch > 0 and epoch % save_int == 0:
        filename = os.path.join(models_path,
                                "image_caption-model{}-{}-{}-{}.pkl".format(
                                    version, epoch, round(val_bleus[-1], 4),
                                    round(val_losses[-1], 4)))

        torch.save({'params': captioner.state_dict(),
                    'optim_params': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_bleus': val_bleus,
                    'epoch': epoch}, filename)
