from generator import create_testing_generator
import os
from keras.models import load_model
import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Add argument for dataset
    parser.add_argument('--test-dataset', dest='test_dataset', type=str, choices=['Lfw', 'Figaro1k', 'Lfw+Figaro1k'],
    help='dataset used (Lfw or Figaro1k', default='Lfw+Figaro1k')

    # Add argument for model
    parser.add_argument('--model', dest='model', type=str, action='store',
    help='model used', required=True)
    
    args = parser.parse_args()

    # Get the values of the arguments
    model_file = args.model
    model_dir = model_file[:-8]               
    test_dataset = args.test_dataset
    test_dataset_path = os.path.join('data', test_dataset)

    test_generator, test_steps = create_testing_generator(dataset=test_dataset_path, shuffle=True)
    model = load_model(model_file)

    # Generate mask from testing set
    cols = ['Original', 'GT', 'pred', 'tresholded pred']
    max_png = 5
    fig, axes = plt.subplots(15, 4, figsize=(10, 20))
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    i = 0
    nb_png = 1
    for generator in test_generator:    
        #Original
        axes[i, 0].imshow(generator[0][0])
        axes[i, 0].axis('off')
        #GT
        axes[i, 1].imshow(generator[1][0])
        axes[i, 1].axis('off')
        #Pred
        pred = model.predict(generator[0])
        axes[i, 2].imshow(pred[0])
        axes[i, 2].axis('off')
        #Tresholded pred
        treshold = 0.7
        pred_mask = ((pred > treshold) * 255.)
        axes[i, 3].imshow(pred_mask[0])
        axes[i, 3].axis('off')
        i += 1
        if i % 15 == 0:        
            plt.savefig(os.path.join(model_dir, f'test_image_{test_dataset}_{nb_png}.png'))
            nb_png += 1
            i = 0
        if nb_png > max_png:
            break        