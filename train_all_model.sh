#!/bin/bash

if [ "$#" -ne 1 ]; then
    image_size=128
else
    image_size=$1
fi

python3 run_train_segmentation.py --dataset Figaro1k --model-type unet --augmentation --size $image_size
python3 run_train_segmentation.py --dataset Figaro1k --model-type unet --no-augmentation --size $image_size
python3 run_train_segmentation.py --dataset Lfw --model-type unet --augmentation --size $image_size
python3 run_train_segmentation.py --dataset Lfw --model-type unet --no-augmentation --size $image_size
python3 run_train_segmentation.py --dataset Lfw+Figaro1k --model-type unet --augmentation --size $image_size
python3 run_train_segmentation.py --dataset Lfw+Figaro1k --model-type unet --no-augmentation --size $image_size

python3 run_train_segmentation.py --dataset Figaro1k --model-type mobile_unet --augmentation --size $image_size
python3 run_train_segmentation.py --dataset Figaro1k --model-type mobile_unet --no-augmentation --size $image_size
python3 run_train_segmentation.py --dataset Lfw --model-type mobile_unet --augmentation --size $image_size
python3 run_train_segmentation.py --dataset Lfw --model-type mobile_unet --no-augmentation --size $image_size
python3 run_train_segmentation.py --dataset Lfw+Figaro1k --model-type mobile_unet --augmentation --size $image_size
python3 run_train_segmentation.py --dataset Lfw+Figaro1k --model-type mobile_unet --no-augmentation --size $image_size

python3 run_train_segmentation.py --dataset Figaro1k --model-type mobile_unet --augmentation --pretrained --size $image_size
python3 run_train_segmentation.py --dataset Figaro1k --model-type mobile_unet --no-augmentation --pretrained --size $image_size
python3 run_train_segmentation.py --dataset Lfw --model-type mobile_unet --augmentation --pretrained --size $image_size
python3 run_train_segmentation.py --dataset Lfw --model-type mobile_unet --no-augmentation --pretrained --size $image_size
python3 run_train_segmentation.py --dataset Lfw+Figaro1k --model-type mobile_unet --augmentation --pretrained --size $image_size
python3 run_train_segmentation.py --dataset Lfw+Figaro1k --model-type mobile_unet --no-augmentation --pretrained --size $image_size

python3 baseline.py --size $image_size