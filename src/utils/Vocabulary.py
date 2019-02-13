# -*- coding: utf-8 -*-
"""
Vocabulary class.
"""
from __future__ import print_function
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class Vocabulary(object):
    """An object for containing a set of words and their mappings."""
    start_word = '<SOS>'
    end_word = '<EOS>'
    unk_word = '<UNK>'
    pad_word = '<PAD>'

    def __init__(self):
        """Construct the vocab class"""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.smoothing = SmoothingFunction()

        # Add tags to dictionary
        for word in [self.pad_word, self.start_word,
                     self.end_word, self.unk_word]:
            self.add_word(word)

    def add_word(self, word):
        """Add given word to vocabulary."""
        if word not in self.word2idx.keys():
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def clean(self, caption):
        """Strip tags from given list of words in a caption."""
        caption = [word for word in caption if word not in [
            self.start_word, self.end_word, self.pad_word]]

        return caption

    def encode(self, caption, length=None):
        """
        Encode a string numerically, optionally to a fixed length.

        Args:
            caption: string to encode
            length: Length to encode string to

        Return:
            a list of word id tokens
        """
        # Tokenize, add start word tag, mapping, and end word tag
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption_ids = []
        caption_ids.append(self(self.start_word))
        caption_ids.extend([self(token) for token in tokens])
        caption_ids.append(self(self.end_word))

        # Pad to fixed length if requested
        if length:
            caption_ids.extend([self(self.pad_word)
                                for i in range(len(caption_ids), length)])
            caption_ids = caption_ids[:length]

        return caption_ids

    def decode(self, caption_ids, join=False, clean=False):
        """
        Decode list of word ids to text.

        Args:
            caption_ids: list of word ids
            join: Option to join list of tokens
            clean: Option to strip tags

        Returns:
            word tags
        """
        caption = [self.idx2word[idx] for idx in caption_ids]
        if clean:
            caption = self.clean(caption)
        if join:
            return ' '.join(caption)

        return caption

    def evaluate(self, references, hypothesis):
        """Compute BLEU-4 score for given refs and hypothesis."""
        if not isinstance(hypothesis, list):
            hypothesis = self.decode(self.encode(hypothesis))

        return sentence_bleu(references, hypothesis,
                             smoothing_function=self.smoothing.method1)

    def __call__(self, word):
        """Return idx of given word."""
        if word not in self.word2idx.keys():
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        """Return length of vocabulary."""
        return len(self.word2idx)
