# -*- coding: utf-8 -*-
"""Extract a COCO captions dataframe from the annotation files."""
from __future__ import print_function
import os
import sys
import argparse
import pandas as pd


def main(args):
    """Extract a COCO captions dataframe from the annotation files."""
    # Load coco library
    sys.path.append(args.coco_path + '/PythonAPI')
    from pycocotools.coco import COCO

    set_2014 = ['val2014', 'train2014']
    set_2017 = ['val2017', 'train2017']

    # Make dataframe to store captions in
    cocoDF = pd.DataFrame(columns=['id', 'set', 'filename', 'caption'])

    for st in set_2014 + set_2017:
        print('\nProcessing {}'.format(st))
        # Instantiate coco classes
        coco = COCO(args.coco_path +
                    'annotations/instances_{}.json'.format(st))
        coco_anns = COCO(args.coco_path +
                         'annotations/captions_{}.json'.format(st))

        # Get Categories
        cats = coco.loadCats(coco.getCatIds())

        # Get unique image ids
        imgIds = []
        for cat in cats:
            imgId = coco.getImgIds(catIds=cat['id'])
            imgIds += imgId
        imgIds = list(set(imgIds))

        # Get annotations
        annIds = coco_anns.getAnnIds(imgIds=imgIds)
        anns = coco_anns.loadAnns(annIds)

        # Extract ids and captions as tuples
        captions = [(int(ann['image_id']), ann['caption']) for ann in anns]
        print(len(captions))

        # Extract filenames as tuples
        img_ids = list(set([ann['image_id'] for ann in anns]))
        imgs = coco.loadImgs(img_ids)
        filenames = [(int(img['id']), st + '/' + img['file_name'])
                     for img in imgs]

        # Make dataframe of captions and filenames
        captionDF = pd.DataFrame(captions, columns=['id', 'caption'])
        filesDF = pd.DataFrame(filenames, columns=['id', 'filename'])

        # Merge dataframes on image id
        df = captionDF.merge(filesDF, how='outer', on='id')

        # Assign to set
        df['set'] = st

        # Concatenate to resultsDF
        cocoDF = pd.concat([cocoDF, df], axis=0)

        # Temporarily store intermediate data
        df.to_csv(args.interim_result_path + 'coco_' +
                  st + '_captions.csv', index=False)

    print('\nDone Converting')
    print('Number of images: {}'.format(cocoDF['id'].nunique()))

    cocoDF.to_csv(args.coco_path +
                  'annotations/coco_captions.csv', index=False)

    print('Saved merged set to ' + args.coco_path +
          'annotations/coco_captions.csv')

    # Make 2014 and 2017 dataframes
    val2014DF = pd.read_csv(args.interim_result_path +
                            'coco_val2014_captions.csv')
    val2017DF = pd.read_csv(args.interim_result_path +
                            'coco_val2017_captions.csv')
    train2014DF = pd.read_csv(
        args.interim_result_path + 'coco_train2014_captions.csv')
    train2017DF = pd.read_csv(
        args.interim_result_path + 'coco_train2017_captions.csv')

    # Concate by year
    df2014 = pd.concat([val2014DF, train2014DF], axis=0)
    df2017 = pd.concat([val2017DF, train2017DF], axis=0)

    # Save
    df2014.to_csv(args.results_path + 'coco_2014_captions.csv', index=False)
    df2017.to_csv(args.results_path + 'coco_2017_captions.csv', index=False)

    print('Saved 2014 set to ' + args.results_path + 'coco_2014_captions.csv')
    print('Saved 2017 set to ' + args.results_path + 'coco_2017_captions.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str, help='coco root path',
                        default=os.environ['HOME'] + '/programs/cocoapi/')
    parser.add_argument('--interim_result_path', type=str,
                        help='Path to interim datafile',
                        default='data/interim/')
    parser.add_argument('--results_path', type=str,
                        help='Path to year datafile',
                        default='data/processed/')
    args = parser.parse_args()
    main(args)
