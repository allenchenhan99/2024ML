from Random_data_generator import poly_linear_data_generator
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt
from copy import deepcopy


# Designed matrix
def designed_matrix(x, n):
    phi = np.vander([x], N=n, increasing=True)
    return phi
    
# b: initial prior
# n: bases
# a: variance
# w: vector
def baysian_linear_regression(b, n, a ,w):
    # initial values
    prior_mean = np.zeros(n)
    precision = b
    inv_var = 1.0 / a
    epsilon = 0.0001
    count = 1
    
    # plot info
    points = []
    mean_10 = 0
    cov_10 = 0
    mean_50 = 0
    var_50 = 0
    
    while True:
        x, y = poly_linear_data_generator(n, a, w)
        points.append([x, y])
        
        phi = designed_matrix(x, n)
        if count == 1:
            posterior_cov = np.linalg.inv(precision * np.identity(n) +  inv_var * np.matmul(phi.T, phi))
            posterior_mean = inv_var * np.matmul(posterior_cov, phi.T) * y
            count += 1
        else:
            posterior_cov = np.linalg.inv(inv_var * np.matmul(phi.T, phi) + np.linalg.inv(prior_cov))
            posterior_mean = np.matmul(posterior_cov, np.matmul(np.linalg.inv(prior_cov), prior_mean) + inv_var * phi.T * y)
            count += 1
            
        predictive_mean = np.matmul(phi, posterior_mean)
        predictive_var = a + np.matmul(np.matmul(phi, posterior_cov), phi.T)
        
        # print result
        print(f'Add data point ({x:.5f}, {y:.5f}):\n')
        print(f'Posterior mean:')
        for i in range(n):
            print(f'{posterior_mean[i][0]:.10f}')
            
        print(f'\nPosterior variance:')
        for i in range(n):
            for j in range(n):
                if j != n - 1:
                    print(f'{posterior_cov[i, j]:.10f},', end='\t')
                else:
                    print(f'{posterior_cov[i, j]:.10f}')
        
        print(f'\nPredictive distribution ~ N({predictive_mean[0][0]:.5f}, {predictive_var[0][0]:.5f})\n')
        
        # plot info
        if count == 10:
            mean_10 = deepcopy(posterior_mean)
            cov_10 = deepcopy(posterior_cov)
        if count == 50:
            mean_50 = deepcopy(posterior_mean)
            cov_50 = deepcopy(posterior_cov)
        
        # check convergence
        if np.linalg.norm(prior_mean - posterior_mean) <= epsilon and count >= 50:
            break
        
        # update
        prior_mean = posterior_mean
        prior_cov = posterior_cov

    return points, posterior_mean, posterior_cov, mean_10, cov_10, mean_50, cov_50

def parse_arguements():
    parser = argparse.ArgumentParser(description='HW3 Bayesian Linear Regrssion')
    parser.add_argument('--b', default=0, type=int)
    parser.add_argument('--n', default=1.0, type=int)
    parser.add_argument('--a', default=0, type=float)
    parser.add_argument('--w', nargs='+', help='e.g. w 1 2 3 means w = [1, 2, 3]', default=[1,2,3,4], type=float)
    
    # --w 1.0,2.5,-3.2
    # parser.add_argument('--w', type=lambda s:[float(item) for item in s.split(',')], default=[])
    
    return parser.parse_args()


# Inputs are 'b', 'n', 'a', '[w]', respectively
if __name__ == '__main__':
    args = parse_arguements()
    b = args.b
    n = args.n
    a = args.a
    w = args.w
    
    # python Baysian_linear_regression.py --b 1 --n 4 --a 1 --w 1 2 3 4
    # python Baysian_linear_regression.py --b 100 --n 4 --a 1 --w 1 2 3 4
    points, posterior_mean, posterior_cov, mean_10, cov_10, mean_50, cov_50 = baysian_linear_regression(b, n, a, w)
    
    # show the result
    x = np.linspace(-2.0, 2.0, 100)
    points = np.transpose(points)
    
    # Ground Truth
    plt.subplot(221)
    plt.title('Ground Truth')
    f = np.poly1d(np.flip(w))
    y = f(x)
    plt.plot(x, y, color='k')
    plt.plot(x, y + a, color='r')
    plt.plot(x, y - a, color='r')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15.0, 25.0)
    
    # Predict Result
    plt.subplot(222)
    plt.title('Predict Result')
    f = np.poly1d(np.flip(np.reshape(posterior_mean, n)))
    y = f(x)
    var = np.zeros(100)
    for i in range(100):
        phi = designed_matrix(x[i], n)
        var[i] = a + np.matmul(phi, np.matmul(posterior_cov, phi.T))[0][0]

    plt.scatter(points[0], points[1], s=10)
    plt.plot(x, y, color='k')
    plt.plot(x, y + var, color='r')
    plt.plot(x, y - var, color='r')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15.0, 25.0)
    
    # After 10 times
    plt.subplot(223)
    plt.title('After 10 times')
    f = np.poly1d(np.flip(np.reshape(mean_10, n)))
    y = f(x)
    var = np.zeros(100)
    for i in range(100):
        phi = designed_matrix(x[i], n)
        var[i] = a + np.matmul(phi, np.matmul(cov_10, phi.T))[0][0]
    plt.scatter(points[0][:10], points[1][:10], s=10)
    plt.plot(x, y, color='k')
    plt.plot(x, y + var, color='r')
    plt.plot(x, y - var, color='r')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15.0, 25.0)
    
    # After 50 times
    plt.subplot(224)
    plt.title('After 50 times')
    f = np.poly1d(np.flip(np.reshape(mean_50, n)))
    y = f(x)
    var = np.zeros(100)
    for i in range(100):
        phi = designed_matrix(x[i], n)
        var[i] = a + np.matmul(phi, np.matmul(cov_50, phi.T))[0][0]

    plt.scatter(points[0][:50], points[1][:50], s=10)
    plt.plot(x, y, color='k')
    plt.plot(x, y + var, color='r')
    plt.plot(x, y - var, color='r')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15.0, 25.0)
    
    plt.subplots_adjust(left=0.125, 
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.2, 
                        hspace=0.35)
    plt.show()