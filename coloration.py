import numpy as np
import cv2
from keras.models import load_model

model = 'mobile_unet.h5'
height = 128
width = 128

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


def change_color(image, color, mask=None, alpha=0.85):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if mask is None:
        mask = predict(image, model, height, width)
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
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out

