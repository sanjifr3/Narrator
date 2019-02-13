# -*- coding: utf-8 -*-
"""Module for validating the Video Captioner models."""
from __future__ import print_function
import sys
import os
import argparse
import torch

from utils.Vocabulary import Vocabulary
from utils.VideoDataloader import get_video_dataloader, VideoDataset
from models.VideoCaptioner import VideoCaptioner


def main(args):
    """
    Vaidate video captioner with given parameters.

    Args:
        args: commandline parameters

    """
    # Get validation MSR-VTT dataloader
    print('Loading validation data...\r', end='')
    val_loader = get_video_dataloader(
        'dev',
        args.videos_path,
        args.vocab_path,
        args.captions_path,
        args.batch_size,
        embedding_size=args.embedding_size,
        load_features=args.load_features,
        load_captions=args.load_captions,
        model=args.base_model,
        preload=args.preload)
    val_loader.dataset.mode = 'dev'
    print('Loading validation data...Done')

    # Extract information from dataset
    vocab_size = val_loader.dataset.get_vocab_size()
    start_id = val_loader.dataset.get_idx(
        )[val_loader.dataset.vocab.start_word]
    end_id = val_loader.dataset.get_idx(
        )[val_loader.dataset.vocab.end_word]
    max_caption_length = val_loader.dataset.max_len

    # Build video captioner with given parameters
    captioner = VideoCaptioner(args.embedding_size, args.embed_size,
                               args.hidden_size, vocab_size,
                               max_caption_length,
                               start_id, end_id)

    if torch.cuda.is_available():
        captioner.cuda()

    # Load weights
    checkpoint = torch.load(args.model_path)
    captioner.load_state_dict(checkpoint['params'])
    captioner.eval()

    val_bleu = 0.0

    # Loop through val batches
    for val_id, val_batch in enumerate(val_loader):
        idxs, vid_embeddings, caption_embeddings = val_batch

        if torch.cuda.is_available():
            vid_embeddings = vid_embeddings.cuda()
            caption_embeddings = caption_embeddings.cuda()

        # Get ground truth captions
        refs = val_loader.dataset.get_references(idxs)

        preds = captioner.predict(vid_embeddings, beam_size=args.beam_size)

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

        # Get validation statistics
        stats = 'Validation step [%d/%d], Bleu: %.4f' \
            % (val_id, val_loader.dataset.get_seq_len(),
               batch_bleu / len(preds))

        print('\r' + stats, end='')
        sys.stdout.flush()

        if val_id % 100 == 0:
            print('\r' + stats)

    val_bleu /= val_loader.dataset.get_seq_len()
    print('\nValidation -- bleu: %.4f' % (val_bleu))


if __name__ == '__main__':
    # Validation parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--beam_size', type=str, required=False,
                        help='Beam size to use during validation',
                        default=None)
    parser.add_argument('--vocab_path', type=str, required=False,
                        help='Path to vocab file',
                        default='data/processed/coco_vocab.pkl')
    parser.add_argument('--captions_path', type=str, required=False,
                        help='Path to COCO captions csv',
                        default=os.environ['HOME'] +
                        '/programs/cocoapi/annotations/' +
                        'coco_captions.csv')
    parser.add_argument('--videos_path', type=str, required=False,
                        help='Path to MSR-VTT videos',
                        default=os.environ['HOME'] +
                        '/Database/MSR-VTT/train-video/')
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
