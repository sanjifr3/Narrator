# Narrator
### A scene description generator.
====================================================

The Narrator library generates audio descriptions for provided images and videos using two CNN-RNN neural networks developed in PyTorch: 1) an image to text description network based on the show-and-tell network, and 2) an extension of this network into video to text description. The video description network can additionally be used to generate descriptions per scene in a video. 

The Narrator is currently served in two ways: 1) a Flask web app currently being hosted on AWS and served via a [website](http://sraj.ca), and 2) a standalone library: Narrator.py. Examples of usage of the website can be seen on the website, and examples of using the library can be seen in notebooks/Narrator Usage Examples.ipynb.

The Narrator library uses Amazon Polly to generate audio descriptions from text, and PySceneDetect for detecting scene changes within a video.

The image description network is trained using the COCO 2014 dataset.
The video description network is trained using the MSR-VTT dataset.

1. [Project organization](#project-organization)
2. [Requirements](#requirements)
3. [Performance](#performance)
<!-- 4. [How to train](#how-to-train)
5. [How to validate](#how-to-validate) -->

## Project organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Location to store trained models to be used by Narrator
    │
    ├── notebooks          <- Example Jupyter notebook + notebooks for validation
    │   ├── Narrator-Usage-Examples.ipynb        <- Examples of how to use Narrator independently of the web interface
    │   ├── Image-Captioner-Validation.ipynb     <- Notebook for validating image captioner model
    │   ├── Video-Captioner-Validation.ipynb     <- Notebook for validating video captioner model
    │   └── Data-Analysis.ipynb                  <- Notebook for analyzing image/data analysis
    │
    ├── results            <- Directory to store results
    │
    ├── samples            <- Image and video samples to use with Narrator.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── ic_trainer.py  <- Module for training image description model
    │   ├── ic_validate.py <- Module for evaluating image description model
    │   ├── Narrator.py    <- Python class for deploying trained image/video description models
    │   ├── vc_trainer.py  <- Module for training video description model
    │   ├── vc_validate.py <- Module for evaluating video description model
    │   │
    │   ├── models        <- Scripts to prepare COCO/MSR-VTT datasets for training
    │   │   ├── EncoderCNN.py <-- Pre-trained CNN PyTorch model
    │   │   ├── ImageCaptioner.py <- Image Captioner Show-and-tell PyTorch Model
    │   │   ├── VideoCaptioner.py <- Video Captioner PyTorch Model
    │   │   ├── S2VTCaptioner.py <- S2VT PyTorch PyTorch Model
    │   │   ├── YoloObjectDetector.py <- Wrapper for yolo submodule
    │   │   └── pytorch-yolo-v3 <- git submodule: https://github.com/ayooshkathuria/pytorch-yolo-v3
    │   │
    │   ├── prepare        <- Scripts to prepare COCO/MSR-VTT datasets for training
    │   │   ├── build_coco_vocabulary.py
    │   │   ├── build_msrvtt_vocabulary.py
    │   │   ├── make_coco_dataset.py
    │   │   ├── make_msrvtt_dataset.py
    │   │   ├── preprocess_coco_images.py
    │   │   ├── preprocess_coco_objects.py
    │   │   ├── preprocess_msrvtt_objects.py
    │   │   └── preprocess_msrvtt_videos.py
    │   │
    │   └── utils          <- Assistance classes and modules
    │       ├── create_transformer.py  <- Create image transformer with torchvision
    │       ├── TTS.py                 <- Amazon Polly wrapper
    │       ├── ImageDataloader.py     <- Torch dataloader for the COCO dataset
    │       ├── VideoDataloader.py     <- Torch dataloader for the MSR-VTT dataset
    │       └── Vocabulary.py          <- Vocabulary class
    │   
    │
    └── web                <- Scripts/templates related to deploying web app with Flask

## Requirements

The dependencies can be downloaded using:

```
pip install -r requirements.txt
```

## Performance

### Image description model

| **Architecture** | **CNN** | **Initialization** | **Greedy** | **Beam = 3** |
| --- | --- | --- | --- | --- |
| LSTM (embed: 256) | Resnet152 | Random | 0.123 | 0.132 |
| GRU (embed:256)   | Resnet152 | Random | 0.122 | 0.131 |
| LSTM (embed: 256) | VGG16 | Random | 0.108 | 0.117 |

### Video description model

| **Architecture** | **CNN** | **Initialization** | **Greedy** | **Beam = 3** |
| --- | --- | --- | --- | --- |
| GRU (embed:256) | Resnet152 |Random | 0.317  | 0.351 |
| LSTM (embed: 256) | Resnet152 |Random | 0.305 | 0.320 |
| LSTM (embed: 256) | VGG16 |Random | 0.283 | 0.318 |
| LSTM (embed: 512) | Resnet152 |Random | 0.270 | 0.317 |
| LSTM (embed: 256) | Resnet152 |Pre-trained COCO | 0.278 |0.310 |

<!-- ## How to train

### 1. Build vocabulary files for COCO and MSRVTT dataset

Build coco vocabulary:
```
python3 src/prepare/build_coco_vocabulary \
    --coco_path <path_to_cocoapi>\
    --vocab_path <desired_path_of_vocab>\
    --threshold <min_word_threshold>\
    --sets <coco_sets_to_include>
```

Build MSR-VTT vocabulary:
```
python3 src/prepare/build_coco_vocabulary \
    --coco_path <path_to_cocoapi>\
    --vocab_path <desired_path_of_vocab>\
    --threshold <min_word_threshold>\
    --sets <coco_sets_to_include>
``` -->

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

