import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
  """
  Default parameters for app
  """
  SAMPLES_DIR = os.path.join(basedir + '/app/static/samples/')
  UPLOAD_DIR = os.path.join(basedir + '/app/static/uploads/')
  TITLE = 'Narrator'
  IMG_EXTENSIONS = set(['png','jpg','jpeg'])
  VID_EXTENSIONS = set(['mp4'])
  ALLOWED_EXTENSIONS = IMG_EXTENSIONS | VID_EXTENSIONS
  DEMO_DIR = os.path.join(basedir + '/app/static/demo/')
  COCOAPI_PATH = os.environ['HOME'] + '/programs/cocoapi/'
  COCO_SET = 'val2017'
  COCO_VOCAB_PATH = os.path.join(basedir + '/../data/processed/coco_vocab.pkl')
  MSRVTT_VOCAB_PATH = os.path.join(basedir + '/../data/processed/msrvtt_vocab.pkl')
  MSRVTT_CAPTION_PATH = os.path.join(basedir + '/../data/processed/msrvtt_captions.csv')
  MSRVTT_DATA_PATH = os.environ['HOME'] + '/Database/MSR-VTT/train-video/'
  IC_MODEL_PATH = os.path.join(basedir + '/../models/image_caption-model3-25-0.1895-4.7424.pkl')
  VC_MODEL_PATH = os.path.join(basedir + '/../models/video_caption-model4-480-0.3936-5.0.pkl')
  IM_EMBEDDING_SIZE = 2048
  VID_EMBEDDING_SIZE = 2048
  EMBED_SIZE = 256
  HIDDEN_SIZE = 512
  NUM_FRAMES = 40
  IM_RES = 224
  INITIAL_BEAM_SIZE = 8
  DEFAULT_BEAM_SIZE = 5
  MAX_LEN = 35
  ENCODER_MODEL = 'resnet152'
