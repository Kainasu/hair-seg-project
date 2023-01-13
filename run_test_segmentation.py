from generator import create_testing_generator, create_training_generators
import os
from keras.models import load_model
import argparse
import glob
import pandas
from keras import backend as K
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryIoU

def latest_model_dir(directory = 'models/Figaro1k-aug'):
    return sorted(glob.glob(os.path.join(directory, 'model*')))[-1]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Add argument for dataset
    parser.add_argument('--dataset', type=str, choices=['Lfw', 'Figaro1k', 'Lfw+Figaro1k', 'all'],
    help='dataset used in training', default='all')

    parser.add_argument('--test-dataset', dest='test_dataset', type=str, choices=['Lfw', 'Figaro1k', 'Lfw+Figaro1k', None],
    help='dataset used for testing', default=None)

    # Add argument for augmentation
    parser.add_argument('--no-augmentation', dest='augmentation', action='store_false')
    parser.add_argument('--augmentation', dest='augmentation', action='store_true')
    parser.set_defaults(augmentation=True)

    # Add argument for model type
    parser.add_argument('--model-type', type=str, dest='model_type', choices=['unet', 'mobile_unet'],
    help='model used (unet or mobile_unet)', default='unet')

    args = parser.parse_args()

    # Get the values of the arguments
    dataset = args.dataset
    dataset_path = os.path.join('data', dataset)
    augmentation = args.augmentation
    aug = 'aug' if augmentation else 'no-aug'
    test_dataset = dataset if args.test_dataset is None else args.test_dataset
    test_dataset_path = os.path.join('data', test_dataset)
    model_type = args.model_type
        
    #Empty list to store results
    results = []

    if dataset == 'all':
        model_type_dir = sorted(glob.glob(os.path.join('models', '*')))        
        for model_type in model_type_dir:
            if model_type.__contains__('baseline'):                
                baseline_dir = sorted(glob.glob(os.path.join(model_type, '*')))        
                for baseline in baseline_dir:
                    model_filename  = os.path.join(baseline, 'model.h5')
                    model = load_model(model_filename)
                    test_dataset = 'Lfw+Figaro1k' if args.test_dataset is None else args.test_dataset
                    test_dataset_path = os.path.join('data', test_dataset)
                    test_generator, test_steps = create_testing_generator(test_dataset_path)
                    score_test = model.evaluate(test_generator, steps=test_steps)
                    results.append({'model':baseline[7:], 'dataset': None, 'augmentation': None,
                        'loss_train' : None, 'acc_train' : None, 'iou_train' : None,
                        'loss_val' : None, 'acc_val' : None, 'iou_val' : None,
                        'test_dataset' : test_dataset,
                        'loss_test' : f'{score_test[0]:.5f}', 'acc_test' : f'{score_test[1]:.5f}', 'iou_test' : f'{score_test[2]:.5f}',
                    })     
            else :
                dataset_dir = sorted(glob.glob(os.path.join(model_type, '*')))        
                for dir in dataset_dir:
                    model_dir = os.path.join(dir, 'latest')
                    model_filename  = os.path.join(model_dir, 'model.h5')
                    model = load_model(model_filename)
                    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['acc', BinaryIoU(target_class_ids=[1])])
                    print('==================', dir)
                    if dir.endswith('no-aug'):                
                        dataset = dir[len(model_type)+1:-7]
                        dataset_path = os.path.join('data', dataset)
                        test_dataset = dataset if args.test_dataset is None else args.test_dataset
                        test_dataset_path = os.path.join('data', test_dataset)
                        augmentation = False
                        train_generator, val_generator, train_steps, val_steps = create_training_generators(dataset_path, augmentation)
                        test_generator, test_steps = create_testing_generator(test_dataset_path)
                        score_train = model.evaluate(train_generator, steps=train_steps)                
                        score_val = model.evaluate(val_generator, steps= val_steps)
                        score_test = model.evaluate(test_generator, steps=test_steps)                
                    else : 
                        dataset = dir[len(model_type)+1:-4]
                        dataset_path = os.path.join('data', dataset)
                        test_dataset = dataset if args.test_dataset is None else args.test_dataset
                        test_dataset_path = os.path.join('data', test_dataset)
                        augmentation = True
                        train_generator, val_generator, train_steps, val_steps = create_training_generators(dataset_path, augmentation)
                        test_generator, test_steps = create_testing_generator(test_dataset_path)
                        score_train = model.evaluate(train_generator, steps=train_steps)                
                        score_val = model.evaluate(val_generator, steps= val_steps)
                        score_test = model.evaluate(test_generator, steps=test_steps)                
                    
                    results.append({'model':model_type[7:], 'dataset': dataset, 'augmentation': augmentation,
                        'loss_train' : f'{score_train[0]:.5f}', 'acc_train' : f'{score_train[1]:.5f}', 'iou_train' : f'{score_train[2]:.5f}',
                        'loss_val' : f'{score_val[0]:.5f}', 'acc_val' : f'{score_val[1]:.5f}', 'iou_val' : f'{score_val[2]:.5f}',
                        'test_dataset' : test_dataset,
                        'loss_test' : f'{score_test[0]:.5f}', 'acc_test' : f'{score_test[1]:.5f}', 'iou_test' : f'{score_test[2]:.5f}',
                    })

    else : 
        model_type_dir = os.path.join('models', model_type)
        dataset_dir = os.path.join(model_type_dir, dataset + '-' + aug )
        print('==================', dataset_dir)
        model_dir = os.path.join(dataset_dir, 'latest')
        model_filename  = os.path.join(model_dir, 'model.h5')
        model = load_model(model_filename)
        train_generator, val_generator, train_steps, val_steps = create_training_generators(dataset_path, augmentation)
        test_generator, test_steps = create_testing_generator(test_dataset_path)
        score_train = model.evaluate(train_generator, steps=train_steps)                
        score_val = model.evaluate(val_generator, steps= val_steps)
        score_test = model.evaluate(test_generator, steps=test_steps)

        results.append({'model':model_type, 'dataset': dataset, 'augmentation': augmentation,
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
    