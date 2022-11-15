import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model

def predict_and_plot(img_path, model, mask_path=None):
    ncols = 3
    
    model = load_model(model)
    img = np.asarray(Image.open(img_path).resize((128,128)).convert('RGB'))/255.

    if mask_path is not None:
        mask = np.asarray(Image.open(mask_path).resize((128,128)).convert('L'))/255.
        ncols += 1

    pred = model.predict(img[np.newaxis,:,:,:])
    treshold = 0.7
    pred_mask = ((pred > treshold) * 255.)
    
    fig, axes = plt.subplots(nrows=1, ncols=ncols)
    plt.subplot(1,ncols,1)
    plt.imshow(img)
    plt.title("Original")
    plt.subplot(1,ncols,2)
    plt.imshow(pred[0])
    plt.title("pred")
    plt.subplot(1,ncols,3)
    plt.imshow(pred_mask[0])
    plt.title("tresholded pred")

    if mask_path is not None:
        plt.subplot(1,ncols,4)
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
    args = parser.parse_args()
    predict_and_plot(args.img_path, args.model, mask_path=args.mask_path)