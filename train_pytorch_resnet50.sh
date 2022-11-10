#!/bin/sh
python -m pytorch.FasterRCNN --train --dataset-dir=VOCdevkit/VOC2007 --backbone=resnet50 --epochs=10 --learning-rate=1e-3 --save-best-to=fasterrcnn_pytorch_tmp.pth
python -m pytorch.FasterRCNN --train --dataset-dir=VOCdevkit/VOC2007 --backbone=resnet50 --epochs=4 --learning-rate=1e-4 --load-from=fasterrcnn_pytorch_tmp.pth --save-best-to=fasterrcnn_pytorch_resnet50.pth
rm fasterrcnn_pytorch_tmp.pth

