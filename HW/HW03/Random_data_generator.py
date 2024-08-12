import numpy as np
# import argparse

def uni_gaussian_data_generator(m, s):
    # m : mean
    # s : variance (這邊應該是stadard deviation嗎?)
    
    # Box-Muller Algo
    # 1. Generate U1~uniform(0,1) U2~uniform(0,1) U1⊥U2
    # 2. R, theta
    # 3. X = Rcos(theta), Y = Rsin(theta)

    U1, U2 = np.random.uniform(0, 1, 2)    
    # U1 = np.random.uniform(0, 1)
    # U2 = np.random.uniform(0, 1)
    
    R = np.sqrt(-2 * np.log(U1))
    theta = 2 * np.pi * U2
    
    # X = m + np.sqrt(s) * R * np.cos(theta)
    # Y = m + np.sqrt(s) * R * np.sin(theta)
    
    # return X, Y
    
    Z = R * np.cos(theta)
    return m + np.sqrt(s) * Z
    
    # # Central Limit Theorem
    # return m + s * (sum(np.random.uniform(0, 1, 12)) - 6)
    
def poly_linear_data_generator(n, a, w):
    # n : basis number
    # a : ~ N(0, a)
    # w : [n, 1] vector

    x = np.random.uniform(-1, 1)
    
    # bias
    e = np.random.normal(0, a)
    
    y = np.dot(np.vander([x], n, increasing=True), w)
    # y = 0
    # for i in range(n):
    #     y += np.power(x, i) * w[i]
    
    return x, (y + e)[0]