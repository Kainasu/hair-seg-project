import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
import time
import cv2

def predict_and_plot(img_path, model, color, mask_path=None):
    ncols = 3
    
    model = load_model(model)
    img = np.asarray(Image.open(img_path).resize((128,128)).convert('RGB'))/255.

    if mask_path is not None:
        mask = np.asarray(Image.open(mask_path).resize((128,128)).convert('L'))/255.
        ncols += 1

    start = time.time()
    pred = model.predict(img[np.newaxis,:,:,:])
    end = time.time()
    print(end-start)

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
        plt.subplot(1,ncols,3)
        plt.imshow(change_color(img, pred[0], color))
        plt.title("Colored photo")

    if mask_path is not None:
        plt.subplot(1,ncols,4)
        plt.imshow(mask)
        plt.title("GT")
        
        start = time.time()
        score = model.evaluate(img[np.newaxis,:,:,:], mask[np.newaxis,:,:])
        end = time.time()
        print(end-start)
        print("accuacy : ", score[1])
    
    plt.show()

def change_color(image, mask, color):
    color = color.lstrip('#')
    color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    color = np.array(color)
    print(color)
    thresh = 0.7
    """Create 3 copies of the mask, one for each color channel"""
    blue_mask = mask.copy()
    blue_mask[mask > thresh] = color[0]
    blue_mask[mask <= thresh] = 0

    green_mask = mask.copy()
    green_mask[mask > thresh] = color[1]
    green_mask[mask <= thresh] = 0

    red_mask = mask.copy()
    red_mask[mask > thresh] = color[2]
    red_mask[mask <= thresh] = 0

    blue_mask = cv2.resize(blue_mask, (image.shape[1], image.shape[0]))
    green_mask = cv2.resize(green_mask, (image.shape[1], image.shape[0]))
    red_mask = cv2.resize(red_mask, (image.shape[1], image.shape[0]))

    """Create an rgb mask to superimpose on the image"""
    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = blue_mask
    mask_n[:, :, 1] = green_mask
    mask_n[:, :, 2] = red_mask

    alpha = 0.85
    beta = (1.0 - alpha)
    out = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

    return (out*255).astype(np.uint8)


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', dest='img_path', action='store', required=True)
    parser.add_argument('--mask', dest='mask_path', action='store')
    parser.add_argument('--model', dest='model', action='store', required=True)
    parser.add_argument('--color', dest='color', action='store', required=False)
    args = parser.parse_args()

    predict_and_plot(args.img_path, args.model, args.color, mask_path=args.mask_path)