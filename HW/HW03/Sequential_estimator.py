from Random_data_generator import uni_gaussian_data_generator
import numpy as np
import argparse
import math

# Sequential estimator
def sequential_estimator(m, s):
    print(f'Data point source function: N({m}, {s})\n')
    
    N = 0
    cur_sum = 0
    cur_sqr_sum = 0
    prev_mean = 0
    prev_var = 0
    epsilon = 1e-7
    
    while True:
        x = uni_gaussian_data_generator(m, s)
        N += 1
        cur_sum += x
        cur_sqr_sum += x ** 2
        
        mean = cur_sum / N
        var = (cur_sqr_sum - (cur_sum ** 2) / N) / (N - 1) if N > 1 else 0
        
        print(f'Add data point: {x}')
        if var == 0:
            print(f'Mean = {mean:.16f}\tVariance = 0.0')
        else:
            print(f'Mean = {mean:.16f}\tVariance = {var:.16f}')

    
        # default L2 norm
        if np.linalg.norm(prev_mean - mean) <= epsilon:
            break
        
        prev_mean = mean
        prev_var = var


def parse_arguements():
    parser = argparse.ArgumentParser(description='HW3 sequential estimator')
    parser.add_argument('--mean', default=0, type=float)
    parser.add_argument('--variance', default=1.0, type=float)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguements()
    mean = args.mean
    variance = args.variance
    
    sequential_estimator(mean, variance)