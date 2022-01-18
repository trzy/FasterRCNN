#!/bin/sh
python -m pytorch.FasterRCNN --train --dataset-dir=VOCdevkit/VOC2007 --epochs=10 --learning-rate=1e-3 --load-from=vgg16_caffe.pth --save-best-to=pytorch_model_tmp.pth
python -m pytorch.FasterRCNN --train --dataset-dir=VOCdevkit/VOC2007 --epochs=4 --learning-rate=1e-4 --load-from=pytorch_model_tmp.pth --save-best-to=pytorch_model.pth
rm pytorch_model_tmp.pth

