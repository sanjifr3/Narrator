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

sys.path.append('../src/features/')
from create_transformer import createTransformer
from Vocabulary import Vocabulary

sys.path.append('../src/models')
from ImageCaptioner import ImageCaptioner
from VideoCaptioner import VideoCaptioner 
from EncoderCNN import EncoderCNN

sys.path.append('../src/tts')
from TTS import TTS

def generateVideoCaption(f, beam_size):
  if not osp.exists(f):
    return "ERROR: File doesn't exist"
  
  cap = cv2.VideoCapture(f)

  num_frames = app.config['NUM_FRAMES']
  total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
  if total_frames < num_frames:
    num_frames = total_frames
  
  vid_array = torch.zeros(num_frames,3, app.config['IM_RES'], app.config['IM_RES'])
  
  if torch.cuda.is_available():
    vid_array = vid_array.cuda()
  
  frame_idx = 0
  while True:
    ret, frame = cap.read()
    
    if not ret or frame_idx == num_frames:
      break
    
    try:
      frame = PIL.Image.fromarray(frame).convert('RGB')
        
      if torch.cuda.is_available():
        frame = transformer(frame).cuda().unsqueeze(0)
      else:
        frame = transformer(frame).unsqueeze(0)

      vid_array[frame_idx] = frame
      frame_idx += 1
    except OSError as e:
      print("Could not process frame in " + video_path)
  
  cap.release()
  
  vid_array = encoder(vid_array)
 
  # Generate caption ids
  caption = video_captioner.predict(vid_array.unsqueeze(0), beam_size=beam_size)[0].cpu().numpy().astype(int)
  caption = msrvtt_vocab.decode(caption, clean=True)

  return caption
def generateImageCaption(f, beam_size, ix):
  if osp.exists(f): # Local image
    im = PIL.Image.open(f).convert('RGB')
  else: # Web image
    try:
      im = PIL.Image.fromarray(io.imread(f)).convert('RGB')

      # Save image if from web
      im.save(app.config['SAMPLES_DIR'] + "image" + str(ix) + '.jpg')
    except:
      return "ERROR: File doesn't exist!"
  
  # Preprocess image
  im = transformer(im).cuda().unsqueeze(0)

  # Encode image
  im = encoder(im)

  # Generate caption ids
  caption = image_captioner.predict(im, beam_size=beam_size)[0].cpu().numpy().astype(int)
  caption = coco_vocab.decode(caption, clean=True)

  return caption

def generateCaption(f, beam_size=app.config['DEFAULT_BEAM_SIZE'], ix=0):
  if f.split('.')[-1] in app.config['IMG_EXTENSIONS']:
    return generateImageCaption(f, beam_size, ix)
  elif f.split('.')[-1] in app.config['VID_EXTENSIONS']:
    return generateVideoCaption(f, beam_size)
  return "ERROR: Invalid file type: " + f.split('.')[-1]

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# Load COCO stuff
coco = COCO(app.config['COCOAPI_PATH'] + 'annotations/instances_{}.json'.format(app.config['COCO_SET']))
cocoCaptionDF = pd.read_csv(app.config['COCOAPI_PATH'] + 'annotations/coco_captions.csv')

msrvttCaptionDF = pd.read_csv(app.config['MSRVTT_CAPTION_PATH'])

# Load Vocabulary
with open(app.config['COCO_VOCAB_PATH'],'rb') as f:
  coco_vocab = pickle.load(f)

coco_vocab_size = len(coco_vocab)
coco_start_id = coco_vocab.word2idx[coco_vocab.start_word]
coco_end_id = coco_vocab.word2idx[coco_vocab.end_word]

with open(app.config['MSRVTT_VOCAB_PATH'],'rb') as f:
  msrvtt_vocab = pickle.load(f)

msrvtt_vocab_size = len(msrvtt_vocab)
msrvtt_start_id = msrvtt_vocab.word2idx[msrvtt_vocab.start_word]
msrvtt_end_id = msrvtt_vocab.word2idx[msrvtt_vocab.end_word]

# Load Image Transformer
transformer = createTransformer()

# Load Encoder
encoder = EncoderCNN(app.config['ENCODER_MODEL'])

# Create image captioner model
image_captioner = ImageCaptioner(app.config['IM_EMBEDDING_SIZE'],
                                 app.config['EMBED_SIZE'],
                                 app.config['HIDDEN_SIZE'],
                                 coco_vocab_size, start_id=coco_start_id, 
                                 end_id=coco_end_id)

# Create video captioner model
video_captioner = VideoCaptioner(app.config['VID_EMBEDDING_SIZE'],
                               app.config['EMBED_SIZE'],
                               app.config['HIDDEN_SIZE'],
                               msrvtt_vocab_size, start_id=msrvtt_start_id,
                               end_id=msrvtt_end_id)

# Load TTS library - default store audio in UPLOAD_DIR
tts = TTS(app.config['UPLOAD_DIR'])

# Load weights
ic_checkpoint = torch.load(app.config['IC_MODEL_PATH'])
image_captioner.load_state_dict(ic_checkpoint['params'])
vc_checkpoint = torch.load(app.config['VC_MODEL_PATH'])
video_captioner.load_state_dict(vc_checkpoint['params'])

# Move models to GPU and set to inference mode
encoder.cuda()
image_captioner.cuda()
video_captioner.cuda()

# Move 
encoder.eval()
image_captioner.eval()
video_captioner.eval()

# Randomly sample image ids
rand_ids = cocoCaptionDF[cocoCaptionDF['set'] == app.config['COCO_SET']].sample(n=16)['id'].values.tolist()
# rand_ids[:] = [:]

# Extract image urls
imSamplesDict = [[],[],[],[]]
for i, im_id in enumerate(rand_ids):
  url = coco.loadImgs(im_id)[0]['coco_url']
  caption = generateCaption(url, beam_size=app.config['INITIAL_BEAM_SIZE'], ix=i)

  gts = cocoCaptionDF[cocoCaptionDF['id'] == rand_ids[i]]['caption']
  gts = gts.apply(lambda x: coco_vocab.encode(x, app.config['MAX_LEN']+1))
  gts = gts.apply(lambda x: coco_vocab.decode(x, clean=True))

  nearest_gt = ''
  best_score = 0.0
  for gt in gts:
      bleu = msrvtt_vocab.evaluate([gt], caption)
      if bleu > best_score:
          best_score = bleu
          nearest_gt = gt

  gt = ' '.join(nearest_gt).capitalize()
  caption = ' '.join(caption).capitalize()

  gt_audio = 'image' + str(i) + '_gt.ogg'
  cap_audio = 'image' + str(i) + '.ogg' 

  tts.generateAudio(gt, app.config['SAMPLES_DIR'] + gt_audio)
  tts.generateAudio(caption, app.config['SAMPLES_DIR'] + cap_audio)

  imSamplesDict[int(i/4)].append({
    'id': im_id,
    'url': url,
    'gt': gt,
    'gt_audio': gt_audio,
    'caption': caption,
    'cap_audio':cap_audio
  })

print ("Images loaded!")

rand_ids = msrvttCaptionDF[msrvttCaptionDF['set'] == 'test'].sample(n=16)['vid_id'].values.tolist()
# rand_ids[:] = [:]

vidSamplesDict = [[],[],[],[]]
for i, vid_id in enumerate(rand_ids):
  url = app.config['MSRVTT_DATA_PATH'] + vid_id + '.mp4'

  caption = generateCaption(url)

  gts = msrvttCaptionDF[msrvttCaptionDF['vid_id'] == vid_id]['caption']
  gts = gts.apply(lambda x: msrvtt_vocab.encode(x, app.config['MAX_LEN']+1))
  gts = gts.apply(lambda x: msrvtt_vocab.decode(x, clean=True))

  nearest_gt = ''
  best_score = 0.0
  for gt in gts:
    bleu = msrvtt_vocab.evaluate([gt], caption)
    if bleu > best_score:
      best_score = bleu
      nearest_gt = gt

  shutil.copy(url, app.config['SAMPLES_DIR'] + 'video' + str(i) + '.mp4')

  # Merge list of words into a string
  gt = ' '.join(nearest_gt).capitalize()
  caption = ' '.join(caption).capitalize()

  gt_audio = 'video' + str(i) + '_gt.ogg'
  cap_audio = 'video' + str(i) + '.ogg' 

  # Create speech files
  tts.generateAudio(gt, app.config['SAMPLES_DIR'] + gt_audio)
  tts.generateAudio(caption, app.config['SAMPLES_DIR'] + cap_audio)

  vidSamplesDict[int(i/4)].append({
    'id': i,
    'url': 'video' + str(i) + '.mp4',
    'gt': gt,
    'gt_audio': gt_audio,
    'caption': caption,
    'cap_audio':cap_audio
  })
print ("Videos loaded!")

##############################################################################
##################################### APP ####################################
##############################################################################
    
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
        file_path = os.path.join(app.config['UPLOAD_DIR'], file.filename) 
        file.save(file_path)
        caption = ' '.join(generateCaption(file_path)).capitalize()
        *filename, ext = file.filename.split('.')
        if isinstance(filename, list):
            filename = '_'.join(filename) # Reaplce existing . with _
        cap_audio = filename + '.ogg'
        tts.generateAudio(caption, app.config['UPLOAD_DIR'] + cap_audio)
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
        file_path = os.path.join(app.config['UPLOAD_DIR'], file.filename) 
        file.save(os.path.join(app.config['UPLOAD_DIR'], file.filename))
        caption = ' '.join(generateCaption(file_path)).capitalize()
        *filename, ext = file.filename.split('.')
        if isinstance(filename, list):
            filename = '_'.join(filename) # Reaplce existing . with _
        cap_audio = filename + '.ogg'
        tts.generateAudio(caption,app.config['UPLOAD_DIR'] + cap_audio)
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
