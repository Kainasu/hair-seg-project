import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


#convert all ppm files to png files
for file in os.listdir('Lfw/masks'):
    if file.endswith('.ppm'):
        hair = Image.open('Lfw/masks/' + file)
        hair = hair.convert("RGB")
        hair.save('Lfw/masks_png/' + file.split('.')[0] + '.png')

# delete folder lfw/masks and rename folder lfw/masks_png to lfw/masks
os.rmdir('Lfw/masks')
os.rename('Lfw/masks_png', 'Lfw/masks')

#remove the files that start with '.'
for file in os.listdir('Lfw/masks'):
    if file.startswith('.'):
        os.remove('Lfw/masks/' + file)

# convert masks to hair masks
for file in os.listdir('Lfw/masks'):
    mask = Image.open('Lfw/masks/' + file)
    mask_arr = np.array(mask)
    mask_map = mask_arr == np.array([255, 0, 0])
    mask_map = np.all(mask_map, axis=2).astype(np.uint8)
    mask_map[mask_map == 1] = 255
    mask_map = Image.fromarray(mask_map)
    mask_map.convert('RGB')
    mask_map.save('Lfw/masks_binary/' + file)

#delete folder lfw/masks and rename folder lfw/masks_binary to lfw/masks
os.rmdir('Lfw/masks')
os.rename('Lfw/masks_binary', 'Lfw/masks')
    
# split the dataset into train and testing directories

def parse_name_list(fp):
        with open(fp, 'r') as fin:
            lines = fin.readlines()
        parsed = list()
        for line in lines:
            name, num = line.strip().split(' ')
            num = format(num, '0>4')
            filename = '{}_{}'.format(name, num)
            parsed.append((name, filename))
        return parsed


# split LFW into training folder and testing folder 
def split_lfw():
    train=True
    txt_file = 'parts_train_val.txt' if train else 'parts_test.txt'
    txt_dir = os.path.join('Lfw/', txt_file)
    name_list = parse_name_list(txt_dir)
    img_dir = os.path.join('Lfw/', 'imgs')
    mask_dir = os.path.join('Lfw/', 'masks')
    train_dir = os.path.join('Lfw/', 'Training')
    test_dir = os.path.join('Lfw/', 'Testing')
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    for elem in name_list:
        img = Image.open(os.path.join(img_dir, elem[1]+'.jpg'))
        mask = Image.open(os.path.join(mask_dir, elem[1]+'.png'))
        img.save(os.path.join(train_dir, 'imgs', elem[1]+'.jpg'))
        mask.save(os.path.join(train_dir, 'masks', elem[1]+'.png'))
    train=False
    for elem in name_list:    
        img = Image.open(os.path.join(img_dir, elem[1]+'.jpg'))
        mask = Image.open(os.path.join(mask_dir, elem[1]+'.png'))
        img.save(os.path.join(test_dir, 'imgs', elem[1]+'.jpg'))
        mask.save(os.path.join(test_dir, 'masks', elem[1]+'.png'))

split_lfw()

