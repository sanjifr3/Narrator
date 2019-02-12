# -*- coding: utf-8 -*-
"""Extract a MSR-VTT captions dataframe from the annotation files."""
from __future__ import print_function, unicode_literals, division
import os
import json
import argparse
import unicodedata
from random import shuffle
import pandas as pd


def unicode_to_ascii(s):
    """Convert unicode to ASCII."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def main(args):
    """Extract a MSR-VTT captions dataframe from the annotation files."""
    with open(args.raw_data_path) as data_file:
        data = json.load(data_file)

    df = pd.DataFrame(columns=['vid_id', 'sen_id', 'caption'])
    df_idx = 0
    if args.continue_converting:
        if os.path.is_file(args.interim_data_path + 'all_captions.csv'):
            df = pd.read_csv(args.interim_data_path + 'all_captions.csv')
            df_idx = len(df)

    if df_idx == 0:
        print('Number of captions: {}'.format(len(data['sentences'])))
    else:
        print('Number of captions remaining: {}'.format(
            len(data['sentences']) - len(df)))

    for i in range(len(data['sentences'])):
        if i % 1000 == 0:
            print('Converting json to csv: {}%\r'.format(
                round(i / float(len(data['sentences'])) * 100, 2)), end='')
            df.to_csv(args.interim_data_path + 'all_captions.csv', index=False)
        df.loc[df_idx, 'vid_id'] = data['sentences'][i]['video_id']
        df.loc[df_idx, 'sen_id'] = data['sentences'][i]['sen_id']
        df.loc[df_idx, 'caption'] = unicode_to_ascii(
            data['sentences'][i]['caption'])
        df_idx += 1

    df.to_csv(args.interim_data_path + 'all_captions.csv', index=False)

    print('\nDone Converting')
    print('Number of videos: {}'.format(df['vid_id'].nunique()))

    # Get and shuffle video names
    vid_names = df['vid_id'].unique()
    shuffle(vid_names)

    # Determine number of videos in training and development set
    num_train_vids = int(len(vid_names) * args.train_pct)
    num_dev_vids = int(len(vid_names) * args.dev_pct)

    # Parition videos into respective sets
    train_vids = vid_names[:num_train_vids]
    dev_vids = vid_names[num_train_vids:num_train_vids + num_dev_vids]
    test_vids = vid_names[num_train_vids + num_dev_vids:]

    print('Number of training videos: {}'.format(len(train_vids)))
    print('Number of development videos: {}'.format(len(dev_vids)))
    print('Number of testing videos: {}'.format(len(test_vids)))

    for i, row in df.iterrows():
        if i % 1000 == 0:
            print('Assigning to sets: {}%\r'.format(
                round(i / float(len(df)) * 100, 2)), end='')
            df.to_csv(args.interim_data_path +
                      'partially_distributed_captions.csv', index=False)
        if row['vid_id'] in train_vids:
            df.loc[i, 'set'] = 'train'
        elif row['vid_id'] in dev_vids:
            df.loc[i, 'set'] = 'dev'
        elif row['vid_id'] in test_vids:
            df.loc[i, 'set'] = 'test'

    df.to_csv(args.final_data_path + 'msrvtt_captions.csv', index=False)
    print('Saved final distributed set to ' +
          args.final_data_path + 'msrvtt_captions.csv')
    print('\nDone')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pct', type=float,
                        help='Percentage of dataset to use for training',
                        default=0.8)
    parser.add_argument('--dev_pct', type=float,
                        help='Percentage of dataset to use for development',
                        default=0.15)
    parser.add_argument('--raw_data_path', type=str,
                        help='Path to raw datafile',
                        default='data/raw/videodatainfo_2017_ustc.json')
    parser.add_argument('--interim_data_path', type=str,
                        help='Path to interim datafile',
                        default='data/interim/')
    parser.add_argument('--final_data_path', type=str,
                        help='Path to final datafile',
                        default='data/processed/')
    parser.add_argument('--continue_converting', type=bool,
                        help='Continue converting json to csv',
                        default=True)
    args = parser.parse_args()
    main(args)
