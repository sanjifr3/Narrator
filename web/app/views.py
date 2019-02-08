from flask import render_template, request, redirect, url_for, send_from_directory
from app import app
import sys
import os
import numpy as np
import pandas as pd
import pickle
import shutil
import torch
import skimage.io as io
import PIL
import os.path as osp
import cv2

sys.path.append(app.config['COCOAPI_PATH'] + 'PythonAPI')
from pycocotools.coco import COCO

sys.path.append('../src/')
from Narrator import Narrator

# Construct classes
narrator = Narrator(
    root_path = app.config['ROOT_PATH'],
    coco_vocab_path = app.config['COCO_VOCAB_PATH'],
    msrvtt_vocab_path = app.config['MSRVTT_VOCAB_PATH'],
    base_model = app.config['ENCODER_MODEL'],
    ic_model_path = app.config['IC_MODEL_PATH'],
    vc_model_path = app.config['VC_MODEL_PATH'])

try:
  samplesDF = pd.read_csv(app.config['SAMPLES_DIR'] + 'sample_captions.csv', index_col=0)
except:
  samplesDF = pd.DataFrame(columns=['id','caption','gt'],index=['name']).head()

# Update any existing samples
if len(app.config['SAMPLES_TO_UPDATE']) > 0:
    coco = COCO(app.config['COCOAPI_PATH'] + 'annotations/instances_{}.json'.format(app.config['COCO_SET']))
    cocoCaptionDF = pd.read_csv(app.config['COCOAPI_PATH'] + 'annotations/coco_captions.csv')
    msrvttCaptionDF = pd.read_csv(app.config['MSRVTT_CAPTION_PATH'])

    im_names = [x for x in app.config['SAMPLES_TO_UPDATE'] if 'image' in x]
    vid_names = [x for x in app.config['SAMPLES_TO_UPDATE'] if 'video' in x]

    rand_im_ids = cocoCaptionDF[cocoCaptionDF['set'] == app.config['COCO_SET']].sample(n=32)['id'].values.tolist()
    rand_im_ids = [x for x in rand_im_ids if x not in samplesDF['id'].values.tolist()][:len(im_names)]
    
    rand_vid_ids = msrvttCaptionDF[msrvttCaptionDF['set'] == 'test'].sample(n=32)['vid_id'].values.tolist()
    rand_vid_ids = [x for x in rand_vid_ids if x not in samplesDF['id'].values.tolist()][:len(vid_names)]

    for i, (name, im_id) in enumerate(zip(im_names,rand_im_ids)):
        url = coco.loadImgs(im_id)[0]['coco_url']
        caption = narrator.genCaption(url, beam_size=8)
    
        gts = cocoCaptionDF[cocoCaptionDF['id'] == im_id]['caption']
        gts = gts.apply(lambda x: narrator.coco_vocab.encode(x, app.config['MAX_LEN']+1))
        gts = gts.apply(lambda x: narrator.coco_vocab.decode(x, clean=True))
        
        nearest_gt = ''
        best_score = 0.0
        for gt in gts:
            bleu = narrator.coco_vocab.evaluate([gt], caption)
            if bleu > best_score:
                best_score = bleu
                nearest_gt = gt
        gt = ' '.join(nearest_gt).capitalize()
        caption = ' '.join(caption).capitalize()
    
        im = PIL.Image.fromarray(io.imread(url)).convert('RGB') 
        im.save(app.config['SAMPLES_DIR'] + name + '.jpg')
        narrator.genAudioFile(gt, app.config['SAMPLES_DIR'] + name + '_gt.ogg')
        narrator.genAudioFile(caption, app.config['SAMPLES_DIR'] + name + '.ogg')

        samplesDF.loc[name,'id'] = im_id
        samplesDF.loc[name,'caption'] = caption
        samplesDF.loc[name,'gt'] = gt

    print ('Images updated!')

    for i, (name, vid_id) in enumerate(zip(vid_names,rand_vid_ids)):
        url = app.config['MSRVTT_DATA_PATH'] + vid_id + '.mp4'
        caption = narrator.genCaption(url, beam_size=8)
    
        gts = msrvttCaptionDF[msrvttCaptionDF['vid_id'] == vid_id]['caption']
        gts = gts.apply(lambda x: narrator.msrvtt_vocab.encode(x, app.config['MAX_LEN']+1))
        gts = gts.apply(lambda x: narrator.msrvtt_vocab.decode(x, clean=True))
    
        nearest_gt = ''
        best_score = 0.0
        for gt in gts:
            bleu = narrator.msrvtt_vocab.evaluate([gt], caption)
            if bleu > best_score:
                best_score = bleu
                nearest_gt = gt
        gt = ' '.join(nearest_gt).capitalize()
        caption = ' '.join(caption).capitalize()

        shutil.copy(url, app.config['SAMPLES_DIR'] + name + '.mp4')
        narrator.genAudioFile(gt, app.config['SAMPLES_DIR'] + name + '_gt.ogg')
        narrator.genAudioFile(caption, app.config['SAMPLES_DIR'] + name + '.ogg')

        samplesDF.loc[name,'id'] = vid_id
        samplesDF.loc[name,'caption'] = caption
        samplesDF.loc[name,'gt'] = gt

    print ('Videos updated!')

    samplesDF.to_csv(app.config['SAMPLES_DIR'] + 'sample_captions.csv')

# Array to store samples
imSamplesDict = [[],[],[],[]]
vidSamplesDict = [[],[],[],[]]
for i, ix in enumerate(range(16)):
    im_sample = samplesDF.loc['image' + str(ix)]
    vid_sample = samplesDF.loc['video' + str(ix)]
    imSamplesDict[int(i/4)].append({
      'id': im_sample['id'],
      'url': 'image' + str(ix) + '.jpg',
      'gt': im_sample['gt'],
      'gt_audio': 'image' + str(ix) + '_gt.ogg',
      'caption': im_sample['caption'],
      'cap_audio': 'image' + str(ix) + '.ogg'
    }) 

    vidSamplesDict[int(i/4)].append({
      'id': vid_sample['id'],
      'url': 'video' + str(ix) + '.mp4',
      'gt': vid_sample['gt'],
      'gt_audio': 'video' + str(ix) + '_gt.ogg',
      'caption': vid_sample['caption'],
      'cap_audio': 'video' + str(ix) + '.ogg'
    })
    
print ("Samples loaded")

##############################################################################
##################################### APP ####################################
##############################################################################

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
    
@app.route('/')
@app.route('/index')
def index():
  return render_template('main.html', page='main', title=app.config['TITLE'])

@app.route('/about')
def about():
  return render_template('about.html', page='about', title=app.config['TITLE'])

@app.route('/images')
def images():
  return render_template('images.html', im_dict=imSamplesDict, page='images', title=app.config['TITLE'])

@app.route('/videos')
def videos():
  return render_template('videos.html', vid_dict=vidSamplesDict, page='videos', title=app.config['TITLE'])

@app.route('/demo', methods=['GET','POST'])
def demo():
  if request.method == 'POST':
    try:
      file = request.files['file']
      if file and allowed_file(file.filename):
        file.filename = file.filename.replace(' ','_')
        file_path = os.path.join(app.config['UPLOAD_DIR'], file.filename) 
        file.save(os.path.join(app.config['UPLOAD_DIR'], file.filename))
        caption = narrator.genCaption(file_path, beam_size=app.config['BEAM_SIZE'], as_string=True).capitalize()
        *filename, ext = file.filename.split('.')
        if isinstance(filename, list):
            filename = '_'.join(filename) # Reaplce existing . with _
        cap_audio = filename + '.ogg'
        narrator.genAudioFile(caption, app.config['UPLOAD_DIR'] + cap_audio)
        typ = 'image'
        if ext in app.config['VID_EXTENSIONS']:
          typ = 'video'
        
        return redirect(url_for('uploaded_file', filename=file.filename, cap_audio=cap_audio, caption=caption, typ=typ))
    except KeyError as e:
      print (e)
  return render_template('demo.html', page='demo', title=app.config['TITLE'])

@app.route('/demo/<filename>&<cap_audio>&<typ>&<caption>', methods=['GET','POST'])
def uploaded_file(filename, cap_audio=None, caption="Test", typ='image'):
  if request.method == 'POST':
    try:
      file = request.files['file']
      if file and allowed_file(file.filename):
        file.filename = file.filename.replace(' ','_')
        file_path = os.path.join(app.config['UPLOAD_DIR'], file.filename) 
        file.save(os.path.join(app.config['UPLOAD_DIR'], file.filename))
        caption = narrator.genCaption(file_path, beam_size=app.config['BEAM_SIZE'], as_string=True).capitalize()
        *filename, ext = file.filename.split('.')
        if isinstance(filename, list):
            filename = '_'.join(filename) # Reaplce existing . with _
        cap_audio = filename + '.ogg'
        narrator.genAudioFile(caption, app.config['UPLOAD_DIR'] + cap_audio)
        typ = 'image'
        if ext in app.config['VID_EXTENSIONS']:
          typ = 'video'
        return redirect(url_for('uploaded_file', filename=file.filename, cap_audio=cap_audio, caption=caption, typ=typ))
    except KeyError as e:
      print (e)
  return render_template('demo_output.html', typ=typ, caption=caption, filename=filename, cap_audio=cap_audio, page='demo', title=app.config['TITLE'])

@app.route('/uploads/<filename>')
def send_file(filename):
  print (filename)
  return send_from_directory(app.config['UPLOAD_DIR'], filename)

@app.route('/samples/<filename>')
def get_sample(filename):
  print(filename)
  return send_from_directory(app.config['SAMPLES_DIR'], filename)
