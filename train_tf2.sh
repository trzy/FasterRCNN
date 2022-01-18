#!/bin/sh
python -m tf2.FasterRCNN --train --dataset-dir=VOCdevkit/VOC2007 --epochs=10 --learning-rate=1e-3 --save-best-to=tf2_model_tmp.pth
python -m tf2.FasterRCNN --train --dataset-dir=VOCdevkit/VOC2007 --epochs=4 --learning-rate=1e-4 --load-from=tf2_model_tmp.pth --save-best-to=tf2_model.pth
rm tf2_model_tmp.pth
