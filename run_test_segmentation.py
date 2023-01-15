from generator import create_testing_generator, create_training_generators
import os
from keras.models import load_model
import argparse
import glob
import pandas
from keras import backend as K
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryIoU

def get_model_dir(model_type):
    """Returns path(s) to model_type"""
    if model_type == 'all':
        model_type_dir = sorted(glob.glob(os.path.join('models', '*')))
    elif model_type == 'unet':
        model_type_dir = sorted(glob.glob(os.path.join('models', 'unet')))
    elif model_type == 'mobile_unet-pretrained':
        model_type_dir = sorted(glob.glob(os.path.join('models', 'mobile_unet-pretrained')))
    elif model_type == 'mobile_unet-no-pretrained':
        model_type_dir = sorted(glob.glob(os.path.join('models', 'mobile_unet-no-pretrained')))
    elif model_type == 'baseline':
        model_type_dir = sorted(glob.glob(os.path.join('models', 'baseline')))
    return model_type_dir

def get_trained_dataset_dir(model_type, train_dataset):
    """Returns path to model type trained with train_dataset"""
    if train_dataset == 'all':
        dataset_dir = sorted(glob.glob(os.path.join(model_type, '*')))
    elif train_dataset == 'Figaro1k-aug':
        dataset_dir = sorted(glob.glob(os.path.join(model_type, 'Figaro1k-aug')))
    elif train_dataset == 'Figaro1k-no-aug':
        dataset_dir = sorted(glob.glob(os.path.join(model_type, 'Figaro1k-no-aug')))
    elif train_dataset == 'Lfw+Figaro1k-aug':
        dataset_dir = sorted(glob.glob(os.path.join(model_type, 'Lfw+Figaro1k-no-aug')))
    elif train_dataset == 'Lfw+Figaro1k-aug':
        dataset_dir = sorted(glob.glob(os.path.join(model_type, 'Lfw+Figaro1k-no-aug')))
    elif train_dataset == 'Lfw-aug':
        dataset_dir = sorted(glob.glob(os.path.join(model_type, 'Lfw-aug')))
    elif train_dataset == 'Lfw-no-aug':
        dataset_dir = sorted(glob.glob(os.path.join(model_type, 'Lfw-no-aug')))
    elif train_dataset == 'baseline':
        dataset_dir = sorted(glob.glob(os.path.join(model_type, 'only*')))
    return dataset_dir

def get_input_size_dir(dataset_dir, size):
    if size == 'all':
        input_size_dir = sorted(glob.glob(os.path.join(dataset_dir, '*')))
    elif size == '128':
        input_size_dir = sorted(glob.glob(os.path.join(dataset_dir, '128x128')))
    elif size == '256':
        input_size_dir = sorted(glob.glob(os.path.join(dataset_dir, '256x256')))
    return input_size_dir


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Add argument for dataset used for training
    parser.add_argument('--dataset', type=str, choices=['Lfw', 'Figaro1k', 'Lfw+Figaro1k', 'all'],
    help='dataset used for training', default='all')

    parser.add_argument('--test-dataset', dest='test_dataset', type=str, choices=['Lfw', 'Figaro1k', 'Lfw+Figaro1k', None],
    help='dataset used for testing', default=None)

    # Add argument for augmentation
    parser.add_argument('--no-augmentation', dest='augmentation', action='store_false')
    parser.add_argument('--augmentation', dest='augmentation', action='store_true')
    parser.set_defaults(augmentation=True)

    # Add argument for pretrained mobile unet or not
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.set_defaults(pretrained=True)

    # Add argument for model type
    parser.add_argument('--model-type', type=str, dest='model_type', choices=['unet', 'mobile_unet', 'baseline' 'all'],
    help='model used', default='all')

    # Add argument for input size
    parser.add_argument('--size', type=str, choices=['128', '256', 'all'],
    help='size of inputs', default='all')

    args = parser.parse_args()

    # Get the values of the arguments
    dataset = args.dataset    
    augmentation = args.augmentation
    aug = 'aug' if augmentation else 'no-aug'
    if dataset != 'all':
        dataset = f'{dataset}-{aug}'

    test_dataset = dataset if args.test_dataset is None else args.test_dataset    

    model_type = args.model_type
    pretrained = args.pretrained
    pretrain = 'pretrained' if pretrained else 'no-pretrained'    
    if model_type == 'mobile_unet':
        model_type = f'{model_type}-{pretrain}'
    size = args.size

    #Empty list to store results
    results = []

    model_type_dir = get_model_dir(model_type)
    for model_type in model_type_dir:
        if model_type.__contains__('baseline'): 
            trained_dataset_dir = get_trained_dataset_dir(model_type, 'baseline')
        else: 
            trained_dataset_dir = get_trained_dataset_dir(model_type, dataset)
        for dataset_dir in trained_dataset_dir:
            input_size_dir = get_input_size_dir(dataset_dir, size)
            for input_size in input_size_dir:
                print(input_size)
                model_dir = os.path.join(input_size, 'latest')
                model_filename  = os.path.join(model_dir, 'model.h5')
                model = load_model(model_filename)
                _, height, width, depth = model.layers[0].input_shape[0]
                image_size = (height, width, depth)
                model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['acc', BinaryIoU(target_class_ids=[1])])
                if dataset_dir.endswith('no-aug'):
                    dataset_name = dataset_dir[len(model_type)+1:-7]
                    dataset_path = os.path.join('data', dataset_name)
                    test_dataset = dataset_name if args.test_dataset is None else args.test_dataset
                    test_dataset_path = os.path.join('data', test_dataset)
                    augmentation = False
                    train_generator, val_generator, train_steps, val_steps = create_training_generators(dataset_path, augmentation, image_size=image_size)
                    test_generator, test_steps = create_testing_generator(test_dataset_path, image_size=image_size)
                    score_train = model.evaluate(train_generator, steps=train_steps)                
                    score_val = model.evaluate(val_generator, steps= val_steps)
                    score_test = model.evaluate(test_generator, steps=test_steps)
                elif  dataset_dir.endswith('aug'): 
                    dataset_name = dataset_dir[len(model_type)+1:-4]
                    dataset_path = os.path.join('data', dataset_name)
                    test_dataset = dataset_name if args.test_dataset is None else args.test_dataset
                    test_dataset_path = os.path.join('data', test_dataset)
                    augmentation = True
                    train_generator, val_generator, train_steps, val_steps = create_training_generators(dataset_path, augmentation, image_size=image_size)
                    test_generator, test_steps = create_testing_generator(test_dataset_path, image_size=image_size)
                    score_train = model.evaluate(train_generator, steps=train_steps)                
                    score_val = model.evaluate(val_generator, steps= val_steps)
                    score_test = model.evaluate(test_generator, steps=test_steps) 
                else:
                    test_dataset = 'Lfw+Figaro1k' if args.test_dataset is None else args.test_dataset
                    test_dataset_path = os.path.join('data', test_dataset)
                    test_generator, test_steps = create_testing_generator(test_dataset_path, image_size=image_size)
                    score_train = [-1] * 3
                    score_val = [-1] * 3
                    score_test = model.evaluate(test_generator, steps=test_steps)
    
                results.append({'model':model_type[7:], 'dataset': dataset_name, 'augmentation': augmentation, 'input_size': image_size,
                        'loss_train' : f'{score_train[0]:.5f}', 'acc_train' : f'{score_train[1]:.5f}', 'iou_train' : f'{score_train[2]:.5f}',
                        'loss_val' : f'{score_val[0]:.5f}', 'acc_val' : f'{score_val[1]:.5f}', 'iou_val' : f'{score_val[2]:.5f}',
                        'test_dataset' : test_dataset,
                        'loss_test' : f'{score_test[0]:.5f}', 'acc_test' : f'{score_test[1]:.5f}', 'iou_test' : f'{score_test[2]:.5f}',
                    })

    #Print and save results
    results_dataframe = pandas.DataFrame(results)
    print(results_dataframe)

    dirname = 'results'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    filename = os.path.join(dirname, 'results.csv')
    results_dataframe.to_csv(filename, header=True, index=False, sep='\t', mode='a')