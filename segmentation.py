import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
import time
from coloration import change_color

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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', dest='img_path', action='store', required=True)
    parser.add_argument('--mask', dest='mask_path', action='store')
    parser.add_argument('--model', dest='model', action='store', required=True)
    parser.add_argument('--color', dest='color', action='store', required=False)
    args = parser.parse_args()

    predict_and_plot(args.img_path, args.model, args.color, mask_path=args.mask_path)