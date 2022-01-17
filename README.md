Overview

  - FasterRCNN from scratch in PyTorch and TF2/Keras, mention Python 3.8/3.9 (dataclasses)
  - Learning exercise, reproducing paper from scratch
  - Go over the basic usage as well as a discussion of problems encountered

  - Example results

Environment Set Up

  - Create a PyTorch venv
  - Install dependencies, maybe resorting to web site
  - Create a TF2 venv
  - Install dependencies

  - Obtain dataset
  - Obtain pre-trained models if desired

Run

  - PyTorch training and inference
  - Keras training and inference
  - Brief discussion of Keras options

Problems encountered and solutions

  - Use all RPN predictions regardless of objectness score being positive or not
  - Sampling edge proposals
  - Anchor quality
  - State saving PyTorch
  - Instability in Keras due to gradient propagation

Suggestions for improvement

  - Batch size > 1
  - COCO or other datasets
  - Replacement of RPN anchor map with a simple list of anchors like most implementations

# FasterRCNN in PyTorch and TensorFlow 2 w/ Keras
*Copyright 2021-2022 Bart Trzynadlowski*

## Overview

This is a fresh implementation of the FasterRCNN object detection model in both PyTorch and TensorFlow 2 with Keras, using Python 3.7 or higher. Although several years old now, FasterRCNN remains a foundational work in the field and still influences modern object detectors. 

I set out to replicate [the original paper](docs/publications/faster_rcnn.pdf) from scratch using Keras but quickly ran into difficulties. I had a model that was learning *something*, just not nearly as well as the published results. After relenting and peaking at existing implementations, I realized I had misinterpreted some crucial details that were not clearly articulated in the paper itself. Frustrated with Keras, I reimplemented the project in PyTorch and only then back-ported it to Keras, encountering numerous struggles along the way. For the benefit of those undertaking a similar self-learning exercise, whether involving this or other machine learning models, my struggles and learnings are documented here.

My final results using the VOC2007 dataset's 5011 `trainval` images match the paper's. Convergence is achieved in 14 epochs (10 epochs at a learning rate of 0.001 and 4 more at 0.0001), consistent with the learning schedule the paper used. My implementation includes only a VGG-16 backbone as the feature extractor.

| Class | Average Precision |
|-------|-------------------|
| car        | 85.6% |
| horse      | 84.4% |
| cat        | 83.7% |
| bicycle    | 83.5% |
| dog        | 82.0% |
| person     | 81.1% |
| bus        | 78.8% |
| train      | 78.5% |
| cow        | 78.0% |
| motorbike  | 77.8% |
| tvmonitor  | 74.3% |
| sheep      | 68.5% |
| aeroplane  | 68.4% |
| diningtable| 67.5% |
| bird       | 66.9% |
| sofa       | 60.6% |
| boat       | 55.2% |
| bottle     | 52.7% |
| chair      | 50.5% |
| pottedplant| 41.1% |
|**Mean**   | **71.0%** |

**TODO: EXAMPLE IMAGES HERE **

## Background Material

**TODO: write me. Include links to publications and other informative web sites.**

## Environment Setup

Python 3.7 (for `dataclass` support) or higher is required and I personally use 3.8.5. Dependencies for the PyTorch and TensorFlow versions of the model are located in `pytorch/requirements.txt` and `tf2/requirements.txt`, respectively. Separate virtual environments for both are required.

Instructions here are given for Linux systems.

### PyTorch Setup with CUDA

As far as I know, it is not possible to obtain CUDA-enabled PyTorch packages using pip. Therefore, the required packages are commented out in `pytorch/requirements.txt` and must be obtained manually using a command found on the PyTorch web site. Begin by executing the following commands in the base FasterRCNN source directory:

```
python -m venv pytorch_venv
source pytorch_venv/bin/activate
pip install -r pytorch/requirements.txt
```

Next, go to the [PyTorch web site](https://pytorch.org/) and use their installation picker to select a pip package compatible with your version of CUDA. In my case, CUDA 11.3, as shown.


![PyTorch package configuration](docs/images/PyTorch_Configuration.png)

Run the command shown, e.g.:

```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

If all goes well, this should supplant the CPU-only version of PyTorch that was pulled in from `requirements.txt`.

### TensorFlow 2 Setup

TensorFlow environment set up *without* CUDA is very straightforward. The included `tf2/requirements.txt` file should suffice.

```
python -m venv tf2_venv
source tf2_venv/bin/activate
pip install -r tf2/requirements.txt
```

Getting CUDA working is more involved and beyond the scope of this document. I use an NVIDIA docker container and `tf-nightly-gpu` packages.


## Dataset

**TODO: write me**

## Pre-Trained Models and Initial Weights

To train the model, initial weights for the shared VGG-16 layers are required. Keras provides these but PyTorch does not. Instead, the PyTorch model supports initialization from one of two sources:

1. Pre-trained VGG-16 Caffe weights that can be found online as `vgg16_caffe.pth` (SHA1: `e6527a06abfac585939b8d50f235569a33190570`). 
2. Pre-trained VGG-16 weights obtained using [my own Keras model](https://github.com/trzy/VGG16).

Fortunately, `vgg16_caffe.pth` and pre-trained FasterRCNN weights for both the PyTorch and TensorFlow versions can be obtained using `download_models.sh`. My web host is not particularly reliable so if the site is down, try again later or contact me. The models were trained using the scripts included in this repository (`train_pytorch.sh` and `train_tf2.sh`).

When training the TensorFlow version of the model from scratch and no initial weights are loaded explicitly, the Keras pre-trained VGG-16 weights will automatically be used. When training the PyTorch version, remember to load initial VGG-16 weights explicitly, e.g.:

```
python -m pytorch.FasterRCNN --train --epochs=10 --learning-rate=1e-3 --load-from=vgg16_caffe.pth
```

## Running the Model

From the base directory and assuming the proper environment is configured, the PyTorch model is run like this:

```
python -m pytorch.FasterRCNN
```

And the TensorFlow model like this:
```
python -m tf2.FasterRCNN
```

Use `--help` for a summary of options or poke around the included scripts as well as `pytorch/FasterRCNN/__main__.py` and `tf2/FasterRCNN/__main__.py`.

### Training the Model
**TODO: write me**

### Running Predictions
**TODO: write me**

## Development Learnings
**TODO: write me**

## Suggestions for Future Improvement
**TODO: write me**



