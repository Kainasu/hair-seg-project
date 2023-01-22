import argparse
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import load_model
import time
from coloration import change_color
import cv2

def predict(image, model, height=128, width=128):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = load_model(model)
    """Preprocess the input image before prediction"""
    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)
    pred = model.predict(im)   
    mask = pred.reshape((height, width))
    return mask


def predict_and_plot(img_path, model, color, mask_path=None):
    ncols = 2

    #Load model and get inputs dims    
    model = load_model(model)
    _, height, width, _ = model.layers[0].input_shape[0]    

    #Load image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (height, width))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if color is not None:
        ncols += 1
        col_color = 3        

    if mask_path is not None:

        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (height, width))
        ncols += 1
        col_mask = 3 if ncols < 4 else 4

    """Preprocess the input image before prediction"""
    img = img / 255
    img = img.reshape((1,) + img.shape)
    pred = model.predict(img)       
    pred = np.squeeze(mask)
    
    treshold = 0.7
    pred_mask = ((pred > treshold) * 255.)[..., np.newaxis].repeat(3, axis=2)    
    
    fig, axes = plt.subplots(nrows=1, ncols=ncols)    
    plt.subplot(1,ncols,1)    
    plt.imshow(img)
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
    args = parser.parse_args()

    predict_and_plot(args.img_path, args.model, args.color, mask_path=args.mask_path)