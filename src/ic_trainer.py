# -*- coding: utf-8 -*-
"""Module for training the Image Captioner class."""
from __future__ import print_function
import sys
import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from utils.Vocabulary import Vocabulary
from utils.ImageDataloader import get_image_dataloader, ImageDataset
from models.ImageCaptioner import ImageCaptioner

# version < 2: validation loss is invalid
# version 3: lowest validiation loss
# version 4: beam_size 3
# version 10: w/ VGG16
# 11: Resnet152 w/ GRU w/o beam search


def main(args):
    """
    Train image captioner with given parameters.

    Args:
        args: commandline parameters

    """
    # Get training and validation COCO dataloaders
    print("Loading training data...\r", end="")
    train_loader = get_image_dataloader(
        'train', args.coco_set,
        args.images_path,
        args.vocab_path,
        args.captions_path,
        args.batch_size,
        embedding_size=args.embedding_size,
        load_features=args.load_features,
        load_captions=args.load_captions,
        model=args.base_model,
        preload=args.preload)
    train_loader.dataset.mode = 'train'
    print("Loading training data...Done")

    print("Loading validation data...\r", end="")
    val_loader = get_image_dataloader(
        'val', args.coco_set,
        args.images_path,
        args.vocab_path, args.captions_path,
        args.batch_size,
        embedding_size=args.embedding_size,
        load_features=args.load_features,
        load_captions=args.load_captions,
        model=args.base_model,
        preload=args.preload)
    val_loader.dataset.mode = 'val'
    print("Loading validation data...Done")

    # Extract information from the training set
    vocab_size = train_loader.dataset.get_vocab_size()
    start_id = train_loader.dataset.get_idx(
        )[train_loader.dataset.vocab.start_word]
    end_id = train_loader.dataset.get_idx()[
        train_loader.dataset.vocab.end_word]
    max_caption_length = train_loader.dataset.max_len

    # Build Image Captioner model with the given parameters
    captioner = ImageCaptioner(args.embedding_size, args.embed_size,
                               args.hidden_size, vocab_size,
                               max_caption_length,
                               start_id, end_id)

    # Define loss function
    if torch.cuda.is_available():
        captioner.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(params=captioner.parameters(),
                                 lr=args.lr)

    # Initialize storage vectors and scores
    train_losses = []
    val_losses = []
    val_bleus = []

    best_val_bleu = -1000.0

    # Load checkpoint if requested
    if args.initial_checkpoint_file:
        checkpoint = torch.load(
            os.path.join(args.models_path,
                         args.initial_checkpoint_file))
        captioner.load_state_dict(checkpoint['params'])
        optimizer.load_state_dict(checkpoint['optim_params'])
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        val_bleus = checkpoint['val_bleus']
        best_val_bleu = np.array(val_bleus).max()
        starting_epoch = checkpoint['epoch']
    else:
        starting_epoch = 0

    # Loop through requested epochs
    for epoch in range(starting_epoch, args.num_epochs):
        print('Epoch: [{}/{}]'.format(epoch, args.num_epochs))
        captioner.train()
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

        # Validate at given interval
        if epoch > 0 and epoch % args.val_interval == 0:
            val_loss = 0.0
            val_bleu = 0.0
            captioner.eval()

            # Loop through val batches
            for val_id, val_batch in enumerate(val_loader):
                batch_start_time = time.time()
                idxs, im_embeddings, caption_embeddings = val_batch

                if torch.cuda.is_available():
                    im_embeddings = im_embeddings.cuda()
                    caption_embeddings = caption_embeddings.cuda()

                # Get ground truth captions
                refs = val_loader.dataset.get_references(idxs.numpy())

                # Use greedy search or beam search
                if not args.beam_size:
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
                    preds = captioner.predict(
                        im_embeddings, beam_size=args.beam_size)
                    val_loss += 5

                # Calculate bleu loss per sample in batch
                # Sum and add length normalized sum to val_loss
                batch_bleu = 0.0
                for pred_id in range(len(preds)):
                    pred = preds[pred_id].cpu().numpy().astype(int)
                    pred_embed = val_loader.dataset.vocab.decode(
                        pred, clean=True)
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

            # Save model if better one is found
            if val_bleus[-1] > best_val_bleu:
                best_val_bleu = val_bleus[-1]
                print("\nBest model found -- bleu: %.4f, \
                       val_loss: %.4f, train_loss: %.4f" %
                      (val_bleus[-1], val_losses[-1], train_losses[-1]))

                filename = os.path.join(
                    args.models_path, "image_caption-model{}-{}-{}-{}.pkl".
                    format(args.version, epoch, round(val_bleus[-1], 4),
                           round(val_losses[-1], 4)))

                torch.save({'params': captioner.state_dict(),
                            'optim_params': optimizer.state_dict(),
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'val_bleus': val_bleus,
                            'hidden_size': args.hidden_size,
                            'embed_size': args.embed_size,
                            'embedding_size': args.embedding_size,
                            'rnn_type': args.rnn_type,
                            'epoch': epoch}, filename)
            else:
                print("\nValidation -- bleu: %.4f, \
                      val_loss: %.4f, train_loss: %.4f" %
                      (val_bleus[-1], val_losses[-1], train_losses[-1]))

        # Save checkpoint
        if epoch > 0 and epoch % args.save_int == 0:
            filename = os.path.join(
                args.models_path, "image_caption-model{}-{}-{}-{}.pkl".
                format(args.version, epoch, round(val_bleus[-1], 4),
                       round(val_losses[-1], 4)))

            torch.save({'params': captioner.state_dict(),
                        'optim_params': optimizer.state_dict(),
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'val_bleus': val_bleus,
                        'hidden_size': args.hidden_size,
                        'embed_size': args.embed_size,
                        'embedding_size': args.embedding_size,
                        'rnn_type': args.rnn_type,
                        'epoch': epoch}, filename)


if __name__ == '__main__':
    # Training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path', type=str, required=False,
                        help='Path to store models',
                        default='models/')
    parser.add_argument('--beam_size', type=str, required=False,
                        help='Beam size to use during validation',
                        default=3)
    parser.add_argument('--vocab_path', type=str, required=False,
                        help='Path to vocab file',
                        default='data/processed/coco_vocab.pkl')
    parser.add_argument('--captions_path', type=str, required=False,
                        help='Path to COCO captions csv',
                        default=os.environ['HOME'] +
                        '/programs/cocoapi/annotations/' +
                        'coco_captions.csv')
    parser.add_argument('--images_path', type=str, required=False,
                        help='Path to COCO images',
                        default=os.environ['HOME'] +
                        '/Database/coco/images/')
    parser.add_argument('--lr', type=float, required=False,
                        help='Learning rate',
                        default=0.001)
    parser.add_argument('--val_interval', type=int, required=False,
                        help='Frequency of epochs to validate',
                        default=10)
    parser.add_argument('--save_int', type=int, required=False,
                        help='Frequency of epochs to save checkpoint',
                        default=10)
    parser.add_argument('--num_epochs', type=int, required=False,
                        help='Number of epochs',
                        default=1000)
    parser.add_argument('--initial_checkpoint_file', type=str,
                        required=False, help='starting checkpoint file',
                        default=None)
    parser.add_argument('--version', type=int, required=False,
                        help='Tag for current model',
                        default=11)
    parser.add_argument('--batch_size', type=int, required=False,
                        help='Batch size',
                        default=64)
    parser.add_argument('--coco_set', type=int, required=False,
                        help='coco set year to use (2014/2017)',
                        default=2014)
    parser.add_argument('--load_features', type=bool, required=False,
                        help='Load image features or generate',
                        default=True)
    parser.add_argument('--load_captions', type=bool, required=False,
                        help='Load captions from file or generate',
                        default=True)
    parser.add_argument('--preload', type=bool, required=False,
                        help='Load all captions/images at start',
                        default=True)
    parser.add_argument('--base_model', type=str, required=False,
                        help='Base model for CNN',
                        default='resnet152')
    parser.add_argument('--embedding_size', type=int, required=False,
                        help='Image embedding size',
                        default=2048)
    parser.add_argument('--embed_size', type=int, required=False,
                        help='Embedding size for RNN input',
                        default=256)
    parser.add_argument('--hidden_size', type=int, required=False,
                        help='Hidden size for RNN',
                        default=512)
    parser.add_argument('--rnn_type', type=int, required=False,
                        help='Type of RNN unit',
                        default='lstm')
    args = parser.parse_args()
    main(args)
