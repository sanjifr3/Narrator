# -*- coding: utf-8 -*-
"""Builds a Vocabulary with the COCO trainings set."""
from __future__ import print_function, unicode_literals, division
import re
import unicodedata
import os
import sys
import pickle
import argparse
from collections import Counter
import pandas as pd
import nltk

# Get file directory name
DIR_NAME = os.path.dirname(os.path.realpath(__file__))

sys.path.append(DIR_NAME + '/../')
from utils.Vocabulary import Vocabulary


def unicode_to_ascii(s):
    """Convert unicide string to plain ASCII."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    """Lowercase, trim, and remove non-letter characters."""
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def build_vocabulary(captions_path, threshold):
    """
    Generate a vocabulary from the MSR-VTT dataset.

    Args:
        captions_path: Path to MSR-VTT captions
        threshold: Word counter threshold for adding to vocabulary
    Return:
        A Vocabulary built on the MSR-VTT training set.
    """
    counter = Counter()

    # Read in captions
    df = pd.read_csv(captions_path)

    # Tokenize and update token counter
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print('Tokenizing progress: {}%\r'.format(
                round(i / float(len(df)) * 100.0), 2), end='')

        caption = normalize_string(row['caption'])
        tokens = nltk.tokenize.word_tokenize(caption)
        counter.update(tokens)

    # Keep wordsthat have more occurances than threshold
    words = sorted([word for word, cnt in counter.items() if cnt >= threshold])

    # Create vocabulary
    vocab = Vocabulary()

    # Add words to vocabulary
    for i, word in enumerate(words):
        vocab.addWord(word)

    return vocab


def main(args):
    """
    Build MSR-VTT vocabulary from MSR-VTT training dataset.

    Args:
        args: commandline arguments
    """
    # Build vocabulary
    vocab = build_vocabulary(args.captions_path, args.threshold)

    # Save vocabulary
    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print('\n')
    print('Vocabulary size: {}'.format(len(vocab)))
    print('Vocabulary saved to {}'.format(args.vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--captions_path', type=str,
                        help='caption file path',
                        default='data/processed/msrvtt_captions.csv')
    parser.add_argument('--vocab_path', type=str,
                        help='desired vocab file path',
                        default='data/processed/msrvtt_vocab.pkl')
    parser.add_argument('--threshold', type=int,
                        help='minimum words threshold',
                        default=5)
    args = parser.parse_args()
    main(args)
