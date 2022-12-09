#!/bin/bash

mkdir Lfw
cd Lfw

echo "Now downloading lfw-funneled.zip ..."

wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz

echo "Unzip lfw-funneled.tgz ..."

tar -xvf lfw-funneled.tgz

echo "Now downloading GT Images ... "
wget wget http://vis-www.cs.umass.edu/lfw/part_labels/parts_lfw_funneled_gt_images.tgz

echo "Unzip parts_lfw_funneled_gt_images.tgz ..."
tar -xvf parts_lfw_funneled_gt_images.tgz

# rename parts_lfw_funneled_gt_images to masks
mv parts_lfw_funneled_gt_images masks

# remove the zip files
rm lfw-funneled.tgz
rm parts_lfw_funneled_gt_images.tgz

echo "Now downloading txt files"
wget http://vis-www.cs.umass.edu/lfw/part_labels/parts_train.txt
wget http://vis-www.cs.umass.edu/lfw/part_labels/parts_validation.txt
wget http://vis-www.cs.umass.edu/lfw/part_labels/parts_test.txt

echo "Making parts_train_val.txt ..."
cat parts_train.txt parts_validation.txt > parts_train_val.txt

echo "dataset downloaded!"

echo "start preprocessing "
# remove the folders that contain the images and put the images in the Lfw/lfw_funneled folder

mkdir images

mv lfw_funneled/* images/
rm -r lfw_funneled

#run lfw_preprocessing.py
python3 lfw_preprocessing.py

mkdir Lfw/Training/imgs/Original
mkdir Lfw/Training/masks/GT   
mkdir Lfw/Testing/imgs/Original
mkdir Lfw/Testing/masks/GT

# move the images and masks to the corresponding folders
mv images/* Lfw/Training/imgs/Original/
mv masks/* Lfw/Training/masks/GT/








