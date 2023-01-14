import argparse
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import load_model
import time
from coloration import change_color
import cv2

def predict(image, model, image_size=(128,128,3)):
    height, width, _ = image_size
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = load_model(model)
    """Preprocess the input image before prediction"""
    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)
    pred = model.predict(im)   
    mask = pred.reshape((height, width))
    return mask


def predict_and_plot(img_path, model, color, mask_path=None, image_size=(128,128,3)):
    ncols = 2
    dim = (image_size[0], image_size[1])
    img = cv2.imread(img_path)
    img = cv2.resize(img, dim)

    if color is not None:
        ncols += 1
        col_color = 3        

    if mask_path is not None:

        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, dim)
        ncols += 1
        col_mask = 3 if ncols < 4 else 4

    pred = predict(img, model, image_size=image_size)
    
    treshold = 0.7
    pred_mask = ((pred > treshold) * 255.)[..., np.newaxis].repeat(3, axis=2)
    
    fig, axes = plt.subplots(nrows=1, ncols=ncols)    
    plt.subplot(1,ncols,1)    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.subplot(1,ncols,2)
    plt.imshow(pred_mask)
    plt.title("pred")

    if color is not None:                
        plt.subplot(1,ncols,col_color)
        colored_img = change_color(img, pred, color) 
        plt.imshow(cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB))
        plt.title("Colored photo")

    if mask_path is not None:
        plt.subplot(1,ncols, col_mask)
        plt.imshow(mask)
        plt.title("GT")
        # score = model.evaluate(img, mask[np.newaxis,:,:])
        # print("accuacy : ", score[1])
    
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', dest='img_path', action='store', required=True)
    parser.add_argument('--mask', dest='mask_path', action='store')
    parser.add_argument('--model', dest='model', action='store', required=True)
    parser.add_argument('--color', dest='color', action='store', required=False)
    parser.add_argument('--size', dest='size', type=int, default=128)
    args = parser.parse_args()
    image_size = (args.size, args.size, 3)
    
    predict_and_plot(args.img_path, args.model, args.color, mask_path=args.mask_path, image_size=image_size)