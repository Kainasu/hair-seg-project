import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
import time
from coloration import change_color

def predict_and_plot(img_path, model, color, mask_path=None):
    ncols = 2
    
    model = load_model(model)
    img = np.asarray(Image.open(img_path).resize((128,128)).convert('RGB'))/255.

    if color is not None:
        ncols += 1
        col_color = 3        

    if mask_path is not None:
        mask = np.asarray(Image.open(mask_path).resize((128,128)).convert('L'))/255.
        ncols += 1
        col_mask = 3 if ncols < 4 else 4

    pred = model.predict(img[np.newaxis,:,:,:])
    
    treshold = 0.7
    pred_mask = ((pred > treshold) * 255.)
    
    fig, axes = plt.subplots(nrows=1, ncols=ncols)
    plt.subplot(1,ncols,1)
    plt.imshow(img)
    plt.title("Original")
    plt.subplot(1,ncols,2)
    plt.imshow(pred_mask[0])
    plt.title("pred")

    if color is not None:                
        plt.subplot(1,ncols,col_color)
        plt.imshow(change_color(img, pred[0], color))
        plt.title("Colored photo")

    if mask_path is not None:
        plt.subplot(1,ncols, col_mask)
        plt.imshow(mask)
        plt.title("GT")
        score = model.evaluate(img[np.newaxis,:,:,:], mask[np.newaxis,:,:])
        print("accuacy : ", score[1])
    
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', dest='img_path', action='store', required=True)
    parser.add_argument('--mask', dest='mask_path', action='store')
    parser.add_argument('--model', dest='model', action='store', required=True)
    parser.add_argument('--color', dest='color', action='store', required=False)
    args = parser.parse_args()
    
    predict_and_plot(args.img_path, args.model, args.color, mask_path=args.mask_path)