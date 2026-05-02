#!/usr/bin/env python3

from glob import glob
from os import environ
from os.path import join
from sys import argv, exit
from common import load_test_data
import numpy as np
from pytest import main

def grade(data_path):

    from solution import Trainer
    from datasets import load_dataset

    
    trainset = load_dataset("phystech/logreg_train", token=None, split='train')
    X_train, y_train = np.stack([x['data'] for x in trainset]), np.stack([x['label'] for x in trainset])

    if environ.get('CHECKER'):
        token = environ.get('DATASET_TOKEN')
        testset = load_dataset("phystech/logreg_test", token=token, split='test')
        X_test, y_test = np.stack([x['data'] for x in testset]), np.stack([x['label'] for x in testset])
    else:
        X_test, y_test = X_train, y_train

    trainer = Trainer()
    trainer.train_model(X_train, y_train)
    y_pred =  trainer.predict_model(X_test)
    accuracy = np.mean(y_pred==y_test)

    return float(accuracy)


if __name__ == '__main__':
    if len(argv) != 3:
        print(f'Usage: {argv[0]} test/unittest test_name')
        exit(0)

    mode = argv[1]
    test_name = argv[2]
    test_dir = glob(f'tests/[0-9][0-9]_{mode}_{test_name}_input')
    if not test_dir:
        print('Test not found')
        exit(0)

    if environ.get('CHECKER'):
        if mode == 'unittest':
            code = main(['-vvs', '-p', 'no:cacheprovider', join(test_dir[0], 'test.py')])
            res = 'ok' if str(code) == 'ExitCode.OK' else 'failed'
        if mode == 'grade':
            res = grade(test_dir[0])
            res = f'{res:.3f}'
        with open(environ['GITHUB_STEP_SUMMARY'], 'a') as file:
            file.write(f'`{test_name:<15}  {res:>6}`\n')
            print(f'<<{test_name}={res}>>')
        exit(0)
           
    else:
        if mode == 'unittest':
            exit(main(['-vvxs', join(test_dir[0], 'test.py')]))
        if mode == 'grade':
            res = grade(test_dir[0])
            print(f'accuracy: {res:.3f}')
            exit(0)
        
