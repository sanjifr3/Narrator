"""
Constants for web app.
"""
import os
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    """
    Default parameters for app.
    """
    TITLE = 'Narrator'
    SAMPLES_DIR = os.path.join(BASE_DIR + '/app/static/samples/')
    UPLOAD_DIR = os.path.join(BASE_DIR + '/app/static/uploads/')
    IMG_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
    VID_EXTENSIONS = set(['mp4'])
    ALLOWED_EXTENSIONS = IMG_EXTENSIONS | VID_EXTENSIONS
    COCO_SET = 'val2017'
    COCOAPI_PATH = os.environ['HOME'] + '/programs/cocoapi/'
    ROOT_PATH = '../'
    COCO_VOCAB_PATH = 'data/processed/coco_vocab.pkl'
    MSRVTT_VOCAB_PATH = 'data/processed/msrvtt_vocab.pkl'
    ENCODER_MODEL = 'resnet152'
    IC_MODEL_PATH = 'models/image_caption-model3-25-0.1895-4.7424.pkl'
    VC_MODEL_PATH = 'models/video_caption-model11-160-0.3501-5.0.pkl'
    VC_RNN_TYPE = 'gru'
    IC_RNN_TYPE = 'lstm'
    MSRVTT_CAPTION_PATH = os.path.join(
        BASE_DIR + '/../data/processed/msrvtt_captions.csv')
    MSRVTT_DATA_PATH = os.environ['HOME'] + '/Database/MSR-VTT/train-video/'
    BEAM_SIZE = 5
    MAX_LEN = 35
    SAMPLES_TO_UPDATE = []  # Specify any images to update for webapp - eg ['image4']
    SCENE_EXAMPLE_FILE = 'scenes'
    SECRET_KEY = b'_5#y2L"F4Q8z\n\xec]/'
