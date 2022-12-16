from generator import create_training_generators, create_testing_generator
import os
import numpy as np
from datetime import datetime
import argparse
from model import create_unet
from mobile_model import create_mobile_unet
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Add argument for dataset
    parser.add_argument('--dataset', type=str, choices=['Lfw', 'Figaro1k', 'Lfw+Figaro1k'],
    help='dataset used for training', default='Figaro1k')

    parser.add_argument('--test-dataset', dest='test_dataset', type=str, choices=['Lfw', 'Figaro1k', 'Lfw+Figaro1k', None],
    help='dataset used for testing', default=None)

    # Add argument for augmentation
    parser.add_argument('--no-augmentation', dest='augmentation', action='store_false')
    parser.add_argument('--augmentation', dest='augmentation', action='store_true')
    parser.set_defaults(augmentation=True)

    # Add argument for the number of epochs
    parser.add_argument('--epochs', type=int,
                    help='Number of epochs to train the model for', default=50)

    # Add argument for model type
    parser.add_argument('--model-type', type=str, dest='model_type', choices=['unet', 'mobile_unet'],
    help='model used (unet or mobile_unet', default='unet')

    args = parser.parse_args()


    # Get the values of the arguments
    dataset = args.dataset
    dataset_path =os.path.join('data', dataset)
    augmentation = args.augmentation
    aug = 'aug' if augmentation else 'no-aug'
    epochs = args.epochs
    test_dataset = dataset if args.test_dataset is None else args.test_dataset
    test_dataset_path = os.path.join('data', test_dataset)
    model_type = args.model_type
    
    # Create model and generators
    if model_type == 'unet':
        model = create_unet()
    else : 
        model = create_mobile_unet()
    train_generator, val_generator, train_steps, val_steps = create_training_generators(dataset=dataset_path, augmentation=augmentation)
    test_generator, test_steps = create_testing_generator(dataset=test_dataset_path)

    #Create directory to save model and history    
    dirname = f'models/{model_type}/{dataset}-{aug}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    save_dir = os.path.join(dirname, f'model-{now}')
    os.makedirs(save_dir)

    # Train model
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-7)
    history = model.fit(train_generator,validation_data=val_generator,epochs=epochs,batch_size=32, steps_per_epoch=train_steps, validation_steps=val_steps, callbacks=[reduce_lr])

    # Save model
    model.save(os.path.join(save_dir, 'model.h5'))
    # Save history and curves
    np.save(os.path.join(save_dir, 'history.npy') ,history)
    h = history
    plt.plot(h.history['acc'], label='acc')
    plt.plot(h.history['val_acc'], label='val_acc')
    plt.title('Model accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'model_accuracy.png'))
    plt.close()
    
    plt.plot(h.history['loss'], label='loss')
    plt.plot(h.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'model_loss.png'))
    plt.close()
    
    plt.plot(h.history['binary_io_u'], label='iou')
    plt.plot(h.history['val_binary_io_u'], label='val_iou')
    plt.title('Model IoU')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'model_iou.png'))
    plt.close()

    #TODO Generate mask for all testing test (with parameter test-dataset = `all`)
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
            plt.savefig(os.path.join(save_dir, f'test_image_{test_dataset}_{nb_png}.png'))
            nb_png += 1
            i = 0
        if nb_png > max_png:
            break        

    