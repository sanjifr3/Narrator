# -*- coding: utf-8 -*-
"""
Builds a Vocabulary with the COCO trainings eet
"""
from __future__ import print_function, unicode_literals, division
import re
import unicodedata
import sys
import nltk
import pickle
import argparse
from collections import Counter
import os

DIR_NAME = os.path.dirname(os.path.realpath(__file__))

sys.path.append(DIR_NAME + '/../utils/')
from Vocabulary import Vocabulary


def unicodeToAscii(s):
    '''Convert unicide string to plain ASCII'''

    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    '''Lowercase, trim, and remove non-letter characters'''

    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def updateCounter(counter, captions):
    '''Update word counters'''

    # Tokenize and update token counter
    for i, caption in enumerate(captions):
        if i % 1000:
            print(
                'Tokenizing process: {}%\r'.format(
                    round(
                        i /
                        float(
                            len(captions)) *
                        100.0),
                    2),
                end='')

        caption = normalizeString(caption)
        tokens = nltk.tokenize.word_tokenize(caption)
        counter.update(tokens)


def generateVocabulary(counter, threshold):
    '''Generates vocabulary'''

    vocab = Vocabulary()

    # Keep words that have more occurances thatn threshold
    words = sorted([word for word, cnt in counter.items() if cnt >= threshold])

    # Add words to dictionary
    for i, word in enumerate(words):
        vocab.addWord(word)

    return vocab


def main(args):
    """
    Main function for building coco vocabulary

    Args:
        args: commandline arguments
    """

    # Load coco library
    sys.path.append(args.coco_path + '/PythonAPI')
    from pycocotools.coco import COCO

    # Create token counter
    counter = Counter()

    # Sets to include in vocabulary
    sets = args.sets.split(',')

    for st in sets:
        print('\nProcessing {}'.format(st))

        # initialize coco dataset classes
        coco = COCO(
            args.coco_path +
            'annotations/instances_{}.json'.format(st))
        coco_anns = COCO(
            args.coco_path +
            'annotations/captions_{}.json'.format(st))

        # get all categories
        cats = coco.loadCats(coco.getCatIds())

        # Get all unique image Ids
        imgIds = []
        for cat in cats:
            imgId = coco.getImgIds(catIds=cat['id'])
            imgIds += imgId
        imgIds = list(set(imgIds))

        # Extract captions from annotations
        annIds = coco_anns.getAnnIds(imgIds=imgIds)
        anns = coco_anns.loadAnns(annIds)
        captions = [ann['caption'] for ann in anns]

        # Update vocabulary with new captions
        updateCounter(counter, captions)

    # Generate vocabulary
    vocab = generateVocabulary(counter, args.threshold)

    # Save files
    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print('\n')
    print('Vocabulary size: {}'.format(len(vocab)))
    print('Vocabulary saved to {}'.format(args.vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str,
                        help='coco root path',
                        default=os.environ['HOME'] + '/programs/cocoapi/')
    parser.add_argument('--vocab_path', type=str,
                        help='desired vocab file path',
                        default='data/processed/coco_vocab.pkl')
    parser.add_argument('--threshold', type=int,
                        help='minimum words threshold',
                        default=5)
    parser.add_argument('--sets', type=str,
                        help='sets to include in vocabulary',
                        default='train2014,train2017')
    args = parser.parse_args()
    main(args)
