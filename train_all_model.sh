#!/bin/bash

python3 run_train_segmentation.py --dataset Figaro1k --model-type mobile_unet --augmentation
python3 run_train_segmentation.py --dataset Figaro1k --model-type mobile_unet --no-augmentation
python3 run_train_segmentation.py --dataset Lfw --model-type mobile_unet --augmentation
python3 run_train_segmentation.py --dataset Lfw --model-type mobile_unet --no-augmentation
python3 run_train_segmentation.py --dataset Lfw+Figaro1k --model-type mobile_unet --augmentation
python3 run_train_segmentation.py --dataset Lfw+Figaro1k --model-type mobile_unet --no-augmentation