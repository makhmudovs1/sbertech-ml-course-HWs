#!/usr/bin/env python3

from glob import glob
from os import environ
from os.path import join
from sys import argv, exit
from common import load_test_data
import numpy as np
from pytest import main



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
        code = main(['-vvs', '-p', 'no:cacheprovider', join(test_dir[0], 'test.py')])
        res = 'ok' if str(code) == 'ExitCode.OK' else 'failed'
        with open(environ['GITHUB_STEP_SUMMARY'], 'a') as file:
            file.write(f'`{test_name:<15}  {res:>6}`\n\n')
            print(f'<<{test_name}={res}>>')
        exit(0)
           
    else:
        exit(main(['-vvxs', join(test_dir[0], 'test.py')]))
        exit(0)
        
