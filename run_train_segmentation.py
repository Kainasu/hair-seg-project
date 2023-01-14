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

    parser.add_argument('--test-dataset', dest='test_dataset', type=str, choices=['all', 'Lfw', 'Figaro1k', 'Lfw+Figaro1k'],
    help='dataset used for testing', default='all')

    # Add argument for augmentation
    parser.add_argument('--no-augmentation', dest='augmentation', action='store_false')
    parser.add_argument('--augmentation', dest='augmentation', action='store_true')
    parser.set_defaults(augmentation=True)

    # Add argument for use of pretrained encoder
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.set_defaults(pretrained=False)

    # Add argument for the number of epochs
    parser.add_argument('--epochs', type=int,
                    help='Number of epochs to train the model for', default=50)

    # Add argument for input size
    parser.add_argument('--size', type=int,
                    help='Input size', default=128)

    # Add argument for model type
    parser.add_argument('--model-type', type=str, dest='model_type', choices=['unet', 'mobile_unet'],
    help='model used (unet or mobile_unet)', default='unet')

    args = parser.parse_args()


    # Get the values of the arguments
    dataset = args.dataset
    dataset_path =os.path.join('data', dataset)
    augmentation = args.augmentation
    aug = 'aug' if augmentation else 'no-aug'
    pretrained = args.pretrained
    pretrain = 'pretrained' if pretrained else 'no-pretrained'
    epochs = args.epochs
    test_dataset = ['Lfw', 'Figaro1k', 'Lfw+Figaro1k'] if args.test_dataset == 'all' else [args.test_dataset]    
    model_type = args.model_type
    image_size = (args.size, args.size, 3)
    
    # Create model and generators for training
    if model_type == 'unet':
        model = create_unet(image_size=image_size)
    else : 
        model = create_mobile_unet(pretrained=pretrained, image_size=image_size)
        model_type = f'{model_type}-{pretrain}'
    train_generator, val_generator, train_steps, val_steps = create_training_generators(dataset=dataset_path, augmentation=augmentation, image_size=image_size)

    #Create directory to save model and history    
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    dirname = f'models/{model_type}/{dataset}-{aug}/{image_size[0]}x{image_size[1]}'
    save_dir = os.path.join(dirname, f'model-{now}')
    latest_dir = os.path.join(dirname, 'latest')
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Train model
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-7)
    history = model.fit(train_generator,validation_data=val_generator,epochs=epochs,batch_size=32, steps_per_epoch=train_steps, validation_steps=val_steps, callbacks=[reduce_lr])

    # Save model
    model.save(os.path.join(save_dir, 'model.h5'))
    model.save(os.path.join(latest_dir, 'model.h5'))
    # Save history and curves
    np.save(os.path.join(save_dir, 'history.npy') ,history)
    np.save(os.path.join(latest_dir, 'history.npy') ,history)
    h = history
    graph_save_dir = os.path.join(save_dir, 'graphs')
    graph_latest_dir = os.path.join(latest_dir, 'graphs')
    os.makedirs(graph_save_dir, exist_ok=True)
    os.makedirs(graph_latest_dir, exist_ok=True)
    
    plt.plot(h.history['acc'], label='acc')
    plt.plot(h.history['val_acc'], label='val_acc')
    plt.title('Model accuracy')
    plt.legend()
    plt.savefig(os.path.join(graph_save_dir, 'model_accuracy.png'))
    plt.savefig(os.path.join(graph_latest_dir, 'model_accuracy.png'))
    plt.close()
    
    plt.plot(h.history['loss'], label='loss')
    plt.plot(h.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.legend()
    plt.savefig(os.path.join(graph_save_dir, 'model_loss.png'))
    plt.savefig(os.path.join(graph_latest_dir, 'model_loss.png'))
    plt.close()
    
    plt.plot(h.history['binary_io_u'], label='iou')
    plt.plot(h.history['val_binary_io_u'], label='val_iou')
    plt.title('Model IoU')
    plt.legend()
    plt.savefig(os.path.join(graph_save_dir, 'model_iou.png'))
    plt.savefig(os.path.join(graph_latest_dir, 'model_iou.png'))
    plt.close()
    
    # Generate mask from testing set
    for test_set in test_dataset:
        test_set_path = os.path.join('data', test_set)
        test_generator, test_steps = create_testing_generator(dataset=test_set_path, image_size=image_size)
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
                img_save_dir = os.path.join(save_dir, 'segmentation_img')
                img_latest_dir = os.path.join(latest_dir, 'segmentation_img')
                os.makedirs(img_save_dir, exist_ok=True)
                os.makedirs(img_latest_dir, exist_ok=True)
                plt.savefig(os.path.join(img_save_dir, f'test_image_{test_set}_{nb_png}.png'))
                plt.savefig(os.path.join(img_latest_dir, f'test_image_{test_set}_{nb_png}.png'))
                nb_png += 1
                i = 0
            if nb_png > max_png:
                break        

    