from __future__ import print_function
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pickle
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable

from utils.Vocabulary import Vocabulary
from utils.ImageDataloader import get_image_dataloader, ImageDataset
from models.ImageCaptioner import ImageCaptioner

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='model_path', required=True)
parser.add_argument('--beam', type=int, help='Beam size', required=True)
parser.add_argument('--vocab_path', type=str, help='vocab_path', required=True)
args = parser.parse_args()

images_path = os.environ['HOME'] + '/Database/coco/images/'
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
rnn_type = 'gru'

print("Loading validation data...\r", end="")
val_loader = get_image_dataloader('val', coco_set,
                                  images_path,
                                  args.vocab_path, captions_path,
                                  batch_size,
                                  embedding_size=embedding_size,
                                  load_features=load_features,
                                  load_captions=load_captions,
                                  model=base_model,
                                  preload=preload)
val_loader.dataset.mode = 'val'
print("Loading validation data...Done")

vocab_size = val_loader.dataset.get_vocab_size()
start_id = val_loader.dataset.get_idx()[val_loader.dataset.vocab.start_word]
end_id = val_loader.dataset.get_idx()[val_loader.dataset.vocab.end_word]
max_caption_length = val_loader.dataset.max_len

captioner = ImageCaptioner(embedding_size, embed_size,
                           hidden_size, vocab_size,
                           max_caption_length,
                           start_id, end_id)

if torch.cuda.is_available():
    captioner.cuda()

checkpoint = torch.load(args.model_path)

captioner.load_state_dict(checkpoint['params'])
captioner.eval()

val_bleu = 0.0

for val_id, val_batch in enumerate(val_loader):
    idxs, im_embeddings, caption_embeddings = val_batch

    if torch.cuda.is_available():
        im_embeddings = im_embeddings.cuda()
        caption_embeddings = caption_embeddings.cuda()

    # Get ground truth captions
    refs = val_loader.dataset.get_references(idxs.numpy())

    preds = captioner.predict(im_embeddings, beam_size=args.beam)

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
    stats = "Validation step [%d/%d], Bleu: %.4f" \
        % (val_id, val_loader.dataset.get_seq_len(),
           batch_bleu / len(preds))

    print("\r" + stats, end="")
    sys.stdout.flush()

    if val_id % 250 == 0:
        print('\r' + stats)

val_bleu /= val_loader.dataset.get_seq_len()
print("\nValidation -- bleu: %.4f" % (val_bleu))
