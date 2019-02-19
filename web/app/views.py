"""
Backend of Narrator web app.
"""
import os
import sys
import shutil
import pandas as pd
import skimage.io as io
import PIL
from flask import render_template, request, redirect, url_for, send_from_directory, session
from app import app

sys.path.append(app.config['COCOAPI_PATH'] + 'PythonAPI')
from pycocotools.coco import COCO

sys.path.append('../src/')
from Narrator import Narrator

# Construct classes
narrator = Narrator(
    root_path=app.config['ROOT_PATH'],
    coco_vocab_path=app.config['COCO_VOCAB_PATH'],
    msrvtt_vocab_path=app.config['MSRVTT_VOCAB_PATH'],
    base_model=app.config['ENCODER_MODEL'],
    ic_model_path=app.config['IC_MODEL_PATH'],
    vc_model_path=app.config['VC_MODEL_PATH'],
    ic_rnn_type=app.config['IC_RNN_TYPE'],
    vc_rnn_type=app.config['VC_RNN_TYPE']
)

# Load samples from file
try:
    samplesDF = pd.read_csv(
        app.config['SAMPLES_DIR'] + 'sample_captions.csv', index_col=0)
except:
    samplesDF = pd.DataFrame(
        columns=['id', 'caption', 'gt'], index=['name']).head()

# Update any existing samples
if len(app.config['SAMPLES_TO_UPDATE']) > 0:
    # Load image and video datasets
    coco = COCO(app.config[
                'COCOAPI_PATH'] + 'annotations/instances_{}.json'.format(app.config['COCO_SET']))
    cocoCaptionDF = pd.read_csv(
        app.config['COCOAPI_PATH'] + 'annotations/coco_captions.csv')
    msrvttCaptionDF = pd.read_csv(app.config['MSRVTT_CAPTION_PATH'])

    # Determine images and videos to update
    im_names = [x for x in app.config['SAMPLES_TO_UPDATE'] if 'image' in x]
    vid_names = [x for x in app.config['SAMPLES_TO_UPDATE'] if 'video' in x]

    # Randomly select ids from their respective datasets and reject any that already have been
    # chosen
    rand_im_ids = cocoCaptionDF[cocoCaptionDF['set'] == app.config[
        'COCO_SET']].sample(n=32)['id'].values.tolist()
    rand_im_ids = [x for x in rand_im_ids if x not in samplesDF['id'].values.tolist()][
        :len(im_names)]

    rand_vid_ids = msrvttCaptionDF[msrvttCaptionDF['set'] == 'test'].sample(n=32)[
        'vid_id'].values.tolist()
    rand_vid_ids = [x for x in rand_vid_ids if x not in samplesDF['id'].values.tolist()][
        :len(vid_names)]

    # Generate sample information and store to file
    for i, (name, im_id) in enumerate(zip(im_names, rand_im_ids)):
        # Get image and generated caption
        url = coco.loadImgs(im_id)[0]['coco_url']
        caption = narrator.gen_caption(url, beam_size=8)

        # Get all gt captions and encode/decode using vocabulary
        gts = cocoCaptionDF[cocoCaptionDF['id'] == im_id]['caption']
        gts = gts.apply(lambda x: narrator.coco_vocab.encode(
            x, app.config['MAX_LEN'] + 1))
        gts = gts.apply(lambda x: narrator.coco_vocab.decode(x, clean=True))

        # Find nearest gt
        nearest_gt = ''
        best_score = 0.0
        for gt in gts:
            bleu = narrator.coco_vocab.evaluate([gt], caption)
            if bleu > best_score:
                best_score = bleu
                nearest_gt = gt
        gt = ' '.join(nearest_gt).capitalize()
        caption = ' '.join(caption).capitalize()

        # Load and save imge
        im = PIL.Image.fromarray(io.imread(url)).convert('RGB')
        im.save(app.config['SAMPLES_DIR'] + name + '.jpg')

        # Generate audio files
        narrator.gen_audio_file(
            gt, app.config['SAMPLES_DIR'] + name + '_gt.ogg')
        narrator.gen_audio_file(
            caption, app.config['SAMPLES_DIR'] + name + '.ogg')

        # Update samples dataframe
        samplesDF.loc[name, 'id'] = im_id
        samplesDF.loc[name, 'caption'] = caption
        samplesDF.loc[name, 'gt'] = gt

    print('Images updated!')

    for i, (name, vid_id) in enumerate(zip(vid_names, rand_vid_ids)):
        # Get video and generated caption
        url = app.config['MSRVTT_DATA_PATH'] + vid_id + '.mp4'
        caption = narrator.gen_caption(url, beam_size=8)

        # Get all gt captions and encode/decode using vocabulary
        gts = msrvttCaptionDF[msrvttCaptionDF['vid_id'] == vid_id]['caption']
        gts = gts.apply(lambda x: narrator.msrvtt_vocab.encode(
            x, app.config['MAX_LEN'] + 1))
        gts = gts.apply(lambda x: narrator.msrvtt_vocab.decode(x, clean=True))

        # Find the nearest gt
        nearest_gt = ''
        best_score = 0.0
        for gt in gts:
            bleu = narrator.msrvtt_vocab.evaluate([gt], caption)
            if bleu > best_score:
                best_score = bleu
                nearest_gt = gt
        gt = ' '.join(nearest_gt).capitalize()
        caption = ' '.join(caption).capitalize()

        # Copy image to samples directory
        shutil.copy(url, app.config['SAMPLES_DIR'] + name + '.mp4')

        # Generate audio files
        narrator.gen_audio_file(
            gt, app.config['SAMPLES_DIR'] + name + '_gt.ogg')
        narrator.gen_audio_file(
            caption, app.config['SAMPLES_DIR'] + name + '.ogg')

        # update samples dataframe
        samplesDF.loc[name, 'id'] = vid_id
        samplesDF.loc[name, 'caption'] = caption
        samplesDF.loc[name, 'gt'] = gt

    print('Videos updated!')

    # Save samples dataframe
    samplesDF.to_csv(app.config['SAMPLES_DIR'] + 'sample_captions.csv')

# Load samples
IM_SAMPLES_DICT = [[], [], [], []]
VID_SAMPLES_DICT = [[], [], [], []]
for i, ix in enumerate(range(16)):
    im_sample = samplesDF.loc['image' + str(ix)]
    vid_sample = samplesDF.loc['video' + str(ix)]
    IM_SAMPLES_DICT[int(i / 4)].append({
        'id': im_sample['id'],
        'url': 'image' + str(ix) + '.jpg',
        'gt': im_sample['gt'],
        'gt_audio': 'image' + str(ix) + '_gt.ogg',
        'caption': im_sample['caption'],
        'cap_audio': 'image' + str(ix) + '.ogg'
    })

    VID_SAMPLES_DICT[int(i / 4)].append({
        'id': vid_sample['id'],
        'url': 'video' + str(ix) + '.mp4',
        'gt': vid_sample['gt'],
        'gt_audio': 'video' + str(ix) + '_gt.ogg',
        'caption': vid_sample['caption'],
        'cap_audio': 'video' + str(ix) + '.ogg'
    })

print("Samples loaded")

# Get filepath for scene example
scene_example_file = app.config[
    'SAMPLES_DIR'] + app.config['SCENE_EXAMPLE_FILE']

# Create scene example if it doesn't already exist
if not os.path.exists(scene_example_file + '.csv'):
    # Generate captions by scene
    captions, scene_change_timecodes = narrator.gen_caption(
        scene_example_file + '.mp4', by_scene=True, as_string=True)

    # Create dataframe
    sceneSamplesDF = pd.DataFrame({
        'time': scene_change_timecodes,
        'caption': captions
    })

    # Capitalize
    sceneSamplesDF['caption'] = sceneSamplesDF[
        'caption'].apply(lambda x: x.capitalize())

    # Generate audio files for each caption
    for i, caption in enumerate(captions):
        narrator.gen_audio_file(
            caption, scene_example_file + '.' + str(i) + '.ogg')

    # Save samples dataframe
    sceneSamplesDF.to_csv(scene_example_file + '.csv', index=False)

# Load samples dataframe
else:
    sceneSamplesDF = pd.read_csv(scene_example_file + '.csv')

# Load scene example
SCENE_SAMPLES_DICT = []
for i, row in sceneSamplesDF.iterrows():
    SCENE_SAMPLES_DICT.append({
        'time': row['time'],
        'cap_audio': app.config['SCENE_EXAMPLE_FILE'] + '.' + str(i) + '.ogg',
        'caption': row['caption'].capitalize()
    })

##############################################################################
##################################### APP ####################################
##############################################################################


def allowed_file(filename):
    """Determine if a file has an allowed extension."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def save_file(file):
    """Save given file and return path."""
    file_path = os.path.join(app.config['UPLOAD_DIR'], file.filename)
    file.save(os.path.join(app.config['UPLOAD_DIR'], file.filename))
    return file_path


def split_filename(file):
    """Split filename into name and ext."""
    *filename, ext = file.filename.split('.')
    if isinstance(filename, list):
        filename = '_'.join(filename)  # Replace existing . with _
    return filename, ext


def determine_type(ext, by_scene):
    """Determine if image or video."""
    if ext in app.config['VID_EXTENSIONS']:
        if by_scene:
            return 'scene'
        return 'video'
    return 'image'


def generate_caption(file, by_scene):
    """Generate caption for given file"""
    file.filename = file.filename.replace(' ', '_')


@app.route('/')
@app.route('/index')
def index():
    """Render homepage."""
    return render_template('main.html', page='main', title=app.config['TITLE'])


@app.route('/images')
def images():
    """Render image examples page."""
    return render_template('images.html', im_dict=IM_SAMPLES_DICT, page='images',
                           title=app.config['TITLE'])


@app.route('/videos')
def videos():
    """Render video examples page."""
    return render_template('videos.html', vid_dict=VID_SAMPLES_DICT, page='videos',
                           title=app.config['TITLE'])


@app.route('/scenes')
def scenes():
    """Render scene examples page."""
    return render_template('scenes.html', page='scenes', scenes_dict=SCENE_SAMPLES_DICT,
                           title=app.config['TITLE'])


@app.route('/demo', methods=['GET', 'POST'])
def demo():
    """Render demo page."""
    # Check if file is uploaded
    if request.method == 'POST':
        try:
            # Grab file, and if by_scene is requested from website
            file = request.files['file']
            by_scene = 'by_scene' in request.form

            # Check if filetype is allowed
            if file and allowed_file(file.filename):

                # Fix filename, save to file, get ext and determine type
                file.filename = file.filename.replace(' ', '_')
                file_path = save_file(file)
                filename, ext = split_filename(file)
                typ = determine_type(ext, by_scene)

                if typ == 'image':
                    by_scene = False

                # Generate caption/audio and redirect to demo_output page
                if not by_scene:
                    caption = narrator.gen_caption(file_path,
                                                   beam_size=app.config['BEAM_SIZE'],
                                                   as_string=True,
                                                   by_scene=by_scene).capitalize()

                    cap_audio = filename + '.ogg'
                    narrator.gen_audio_file(
                        caption, app.config['UPLOAD_DIR'] + cap_audio)

                    return redirect(url_for('uploaded_file',
                                            filename=file.filename,
                                            cap_audio=cap_audio,
                                            caption=caption,
                                            typ=typ))

                # Generate caption/audio by scene and redirect to demo_output
                # page
                captions, time_codes = narrator.gen_caption(file_path,
                                                            beam_size=app.config[
                                                                'BEAM_SIZE'],
                                                            as_string=True,
                                                            by_scene=by_scene)

                scenes_dict = []
                for i, caption in enumerate(captions):
                    narrator.gen_audio_file(caption,
                                            app.config['UPLOAD_DIR'] +
                                            filename + '.' + str(i) + '.ogg')
                    scenes_dict.append({
                        'time': time_codes[i],
                        'cap_audio': filename + '.' + str(i) + '.ogg',
                        'caption': caption.capitalize()
                    })
                session['scenes_dict'] = scenes_dict
                return redirect(url_for('uploaded_file',
                                        filename=file.filename,
                                        typ='scene',
                                        caption='scene',
                                        cap_audio='scene'))
        except KeyError as e:
            print(e)
    return render_template('demo.html', page='demo', title=app.config['TITLE'])


@app.route('/demo/<filename>&<cap_audio>&<typ>&<caption>', methods=['GET', 'POST'])
def uploaded_file(filename, typ='image', caption="", cap_audio=None):
    """Render demo output page."""

    # Duplicate of above -- allows
    if request.method == 'POST':
        try:
            # Grab file, and if by_scene is requested from website
            file = request.files['file']
            by_scene = 'by_scene' in request.form

            # Check if filetype is allowed
            if file and allowed_file(file.filename):

                # Fix filename, save to file, get ext and determine type
                file.filename = file.filename.replace(' ', '_')
                file_path = save_file(file)
                filename, ext = split_filename(file)
                typ = determine_type(ext, by_scene)

                if typ == 'image':
                    by_scene = False

                # Generate caption/audio and redirect to demo_output page
                if not by_scene:
                    caption = narrator.gen_caption(file_path,
                                                   beam_size=app.config[
                                                       'BEAM_SIZE'],
                                                   as_string=True,
                                                   by_scene=by_scene).capitalize()

                    cap_audio = filename + '.ogg'
                    narrator.gen_audio_file(
                        caption, app.config['UPLOAD_DIR'] + cap_audio)

                    return redirect(url_for('uploaded_file',
                                            filename=file.filename,
                                            cap_audio=cap_audio,
                                            caption=caption,
                                            typ=typ))

                # Generate caption/audio by scene and redirect to demo_output
                # page
                captions, time_codes = narrator.gen_caption(file_path,
                                                            beam_size=app.config[
                                                                'BEAM_SIZE'],
                                                            as_string=True,
                                                            by_scene=by_scene)

                scenes_dict = []
                for i, caption in enumerate(captions):
                    narrator.gen_audio_file(caption,
                                            app.config['UPLOAD_DIR'] + filename +
                                            '.' + str(i) + '.ogg')
                    scenes_dict.append({
                        'time': time_codes[i],
                        'cap_audio': filename + '.' + str(i) + '.ogg',
                        'caption': caption.capitalize()
                    })
                session['scenes_dict'] = scenes_dict
                return redirect(url_for('uploaded_file',
                                        filename=file.filename,
                                        typ='scene',
                                        caption='scene',
                                        cap_audio='scene'))
        except KeyError as e:
            print(e)
    return render_template('demo_output.html',
                           filename=filename,
                           typ=typ,
                           caption=caption,
                           cap_audio=cap_audio,
                           page='demo',
                           title=app.config['TITLE'])


@app.route('/uploads/<filename>')
def get_upload(filename):
    """Get path to file in upload directory."""
    return send_from_directory(app.config['UPLOAD_DIR'], filename)


@app.route('/samples/<filename>')
def get_sample(filename):
    """Get path to file in samples directory."""
    return send_from_directory(app.config['SAMPLES_DIR'], filename)
