import pandas as pd 
import argparse
from pathlib import Path


if __name__ == '__main__':

    columns = ['model', 'input_size', 'dataset', 'augmentation', 'test_dataset']

    parser = argparse.ArgumentParser()

    #Add accuracy columns
    parser.add_argument('--acc', action='store_true')
    parser.add_argument('--iou', action='store_true')
    parser.add_argument('--both', action='store_true')

    #Sort by
    parser.add_argument('--sort', type=str, dest='sort', choices=['model', 'dataset', 'input_size'],
    help='sort by')
        
    args = parser.parse_args()

    sort = [] if args.sort is None else [args.sort]
    both = args.both 
    acc = True if both else args.acc
    iou = True if both or acc is False else args.iou
    
    if iou :
        columns.append('iou_test')
        columns.append('iou_val')
        columns.append('iou_train')
        sort.append('iou_test')

    if acc :
        columns.append('acc_test')
        columns.append('acc_val')
        columns.append('acc_train')
        sort.append('acc_test')


        
    data = pd.read_csv(Path(__file__).parent / 'results.csv', sep='\t')

    

    print(data[columns].sort_values(by=sort, ascending=False))
