# -*- coding: utf-8 -*-
from __future__ import print_function
import nltk
import pickle
import argparse
import pandas as pd
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

class Vocabulary(object):
  start_word = '<SOS>'
  end_word = '<EOS>'
  unk_word = '<UNK>'
  pad_word = '<PAD>'

  def __init__(self):
    self.word2idx = {}
    self.idx2word = {}
    self.idx = 0
    
    self.smoothing = SmoothingFunction()
    
    for word in [self.pad_word, self.start_word, self.end_word, self.unk_word]:
      self.addWord(word)

  def addWord(self, word):
    if not word in self.word2idx.keys():
      self.word2idx[word] = self.idx
      self.idx2word[self.idx] = word
      self.idx += 1
  
  def clean(self, caption):
    caption = [word for word in caption if word not in [self.start_word, self.end_word, self.pad_word]]
    
    return caption
  
  def encode(self, caption, length=None):
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    caption_ids = []
    caption_ids.append(self(self.start_word))
    caption_ids.extend([self(token) for token in tokens])
    caption_ids.append(self(self.end_word))
    if length:
      caption_ids.extend([self(self.pad_word) for i in range(len(caption_ids),length)])
      caption_ids = caption_ids[:length]

    return caption_ids
  
  def decode(self, caption_ids, join=False, clean=False):
    caption = [self.idx2word[idx] for idx in caption_ids]
    if clean:
      caption = self.clean(caption)
    if join:
      return ' '.join(caption)
    else:
      return caption

  def evaluate(self, references, hypothesis):
    if not isinstance(hypothesis, list):
      hypothesis = self.decode(self.encode(hypothesis))
    
    return sentence_bleu(references, hypothesis, smoothing_function=self.smoothing.method1)
      
  def __call__(self, word):
    if not word in self.word2idx.keys():
      return self.word2idx[self.unk_word]
    return self.word2idx[word]

  def __len__(self):
    return len(self.word2idx)