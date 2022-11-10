#!/bin/sh
python -m pytorch.FasterRCNN --train --dataset-dir=VOCdevkit/VOC2007 --epochs=10 --learning-rate=1e-3 --load-from=vgg16_caffe.pth --save-best-to=fasterrcnn_pytorch_tmp.pth
python -m pytorch.FasterRCNN --train --dataset-dir=VOCdevkit/VOC2007 --epochs=4 --learning-rate=1e-4 --load-from=fasterrcnn_pytorch_tmp.pth --save-best-to=fasterrcnn_pytorch_vgg16.pth
rm fasterrcnn_pytorch_tmp.pth

