import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# closed-form lse
    # LU decompos (ATA + lambda I)
# steepest descent method
    # lse and l1 norm
    # learning rate as samll as possible
# Newton's method


# (1) data to matrix 'A'
# (2) ATA
# (3) LU decomposition
# (4) matrix inverse operation

# (1) A matrix
def data_to_matrix(data, base):
    n = base
    A = []
    
    for i, x in enumerate(data[0]):
        row = []
        for j in range(n):
            row.append(pow(x, j))
        A.append(row)
    A = np.array(A)
    return A
    
    
# (2) ATA
# ATA = np.matmul(A.T, A)
# ATA = np.dot...
# def ATA(A):
#     A_transpose = []
#     for j in range(len(A[0])):
#         row = []
#         for i in range(len(A)):
#             row.append(A[i][j])
#         A_transpose.append(row)
    
#     result = []
#     for i in range(len(A_transpose)):
#         row = []
#         for j in range(len(A_transpose)):
#             sum = 0
#             for k in range(len(A)):
#                 sum += A_transpose[i][k] * A[k][j]
#             row.append(sum)
#         result.append(row)
    
#     return result
    
# (3) LU decomposition
def LU_decomp(A):
    n = len(A)
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    for i in range(n):
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:, ] -= factor[:, np.newaxis] * U[i] # increase dimension on row
    return L, U

# matrix inverse with Gauss-Jordan
def tri_inverse(A):
    n = len(A)
    inv_A = np.eye(n)
    for i in range(n):
        for j in range (i + 1): 
            inv_A[i][j] /= A[i][i]
        
        for j in range(i + 1, n):   #column above diagonal element
            temp = A[j][i]
            
            for k in range(j):
                inv_A[j][k] -= temp * inv_A[i][k]
    
    return inv_A

# closed-form LSE approach
def LSE(A, base, Lambda):
    ATA = np.matmul(A.T, A)
    ATA += Lambda * np.eye(base)
    
    L, U = LU_decomp(ATA)
    inv_L = tri_inverse(L)
    inv_U = tri_inverse(U.T).T
    
    # x = np.matmul(A.T, df[1])
    # x = np.matmul(inv_L, x)
    # x = np.matmul(inv_ U, x)
    x = np.matmul(inv_U, np.matmul(inv_L, np.matmul(A.T, df[1])))    
    return x


# Steepest descent method
def steepest_descent(A, lr, base, iterations=5000):

    # x = np.zeros((A.shape[1]))
    x = np.zeros((base, 1))
    b = df[1].values.reshape(-1, 1)
    
    for _ in range(iterations):
        r = b - np.matmul(A, x)
        gradient = -np.matmul(A.T, r)
        x -= lr * gradient
        print(L1_norm(A, x, b))
    return x

    # r = b - np.matmul(A, x)
    # gradient = -np.matmul(A.T, r)
    # x -= lr * gradient
    # return x
    

# Newton's method
def newton_method(A, base):
    n = base
    
    # Ax = b
    b = df[1].values.reshape(-1, 1)
    ATA = np.matmul(A.T, A)
    
    # random initial point
    x = (np.random.rand(n, 1) * 2 - [[0.5] for _ in range(n)])
    
    # For testing
    # print("Shape of ATA:", ATA.shape)
    # print("Shape of b:", b.shape)
    
    # gradient = 2ATAx - 2AT * b
    gradient = np.matmul(ATA, x) - np.matmul(A.T, b)
    
    # Hessian
    Hessian = ATA
    L, U = LU_decomp(Hessian)
    inv_L = tri_inverse(L)
    inv_U = tri_inverse(U.T).T
    inv_Hessian = np.matmul(inv_U, inv_L)
    
    x -= np.matmul(inv_Hessian, gradient)
    return x.reshape(-1)

# error (by using Residual sum of squares)
def error_compute_RSS(A, x, y):
    error_vec = np.matmul(A, x) - y
    error_vec = list(map(lambda x: x ** 2, error_vec))
    return sum(error_vec)

def error_compute_MSE(A, x, y):
    error_vec = np.matmul(A, x) - y
    error_vec = list(map(lambda x : x ** 2, error_vec))
    return sum(error_vec) / len(y)

# error for L1-norm
def L1_norm(A, x, y):
    if isinstance(y, pd.Series):
        y = y.values.reshape(-1, 1)
    elif isinstance(y, np.ndarray) and y.ndim == 1:
        y = y.reshape(-1, 1)
    error_vec = np.matmul(A, x) - y
    error_vec = np.abs(error_vec)
    return sum(error_vec)


if __name__ == '__main__':
    base = int(input("The bases of polynomial : "))
    Lambda = int(input("Lambda : "))
    lr = float(input("Learning_rate for steepest descent : "))
    
    df = pd.read_csv('C:/Users/ChenHan/Desktop/Allen_Lin/112/112_下/ML/ML_HW/HW01/test.txt', sep=',', header=None)
    A = data_to_matrix(df, base)

    
    coef_of_LSE = LSE(A, base, Lambda)
    coef_of_Steepest = steepest_descent(A, lr, base)
    coef_of_Newton = newton_method(A, base)
    
    error_of_LSE = error_compute_RSS(A, coef_of_LSE, df[1])
    error_of_Steepest = L1_norm(A, coef_of_Steepest, df[1])
    error_of_Newton = error_compute_RSS(A, coef_of_Newton, df[1])
    
    # LSE
    print("LSE: ")
    output_of_LSE = "Fitting line: "
    
    for degree in range(base - 1, -1, -1):
        if degree != (base - 1):
            if coef_of_LSE[degree] >= 0:
                output_of_LSE += ' + '
            else:
                output_of_LSE += ' '
        if degree != 0:
            output_of_LSE += f'{coef_of_LSE[degree]: .11f}X^{degree}'
        else:
            output_of_LSE += f'{coef_of_LSE[degree]: .11f}'
    
    print(output_of_LSE)
    print(f'Total error: {error_of_LSE: .11f}')
    print()
    
    # Steepest descent
    print("Steepest Descent Method: ")
    output_of_steepest = "Fitting line: "
    
    for degree in range(base - 1, -1, -1):
        coef = coef_of_Steepest[degree][0]
        if degree != (base - 1):
            if coef >= 0:
                output_of_steepest += ' + '
            else:
                output_of_steepest += ' '
        if degree != 0:
            output_of_steepest += f'{coef: .11f}X^{degree}'
        else:
            output_of_steepest += f'{coef: .11f}'
    
    print(output_of_steepest)
    print(f'Total error: {np.sum(error_of_Steepest): .11f}')
    print()
    
    
    # Newton's method
    print("Newton's Method: ")
    output_of_newton = "Fitting line: "
    
    for degree in range(base - 1, -1, -1):
        if degree != (base - 1):
            if coef_of_Newton[degree] >= 0:
                output_of_newton += ' + '
            else:
                output_of_newton += ' '
        if degree != 0:
            output_of_newton += f'{coef_of_Newton[degree]: .11f}X^{degree}'
        else:
            output_of_newton += f'{coef_of_Newton[degree]: .11f}'
    
    print(output_of_newton)
    print(f'Total error: {error_of_Newton: .11f}')
    print()
    
    
    # plot
    LSE_func = 0
    steepest_func = 0
    newton_func = 0
    for degree in range(base-1, -1, -1):
        LSE_func += coef_of_LSE[degree] * (df[0].values ** degree)
        steepest_func += coef_of_Steepest[degree] * (df[0].values ** degree)
        newton_func += coef_of_Newton[degree] * (df[0].values ** degree) 
    
    # # plt.figure(1)
    # plt.figure(figsize=(8,6))
    
    plt.subplot(311)
    plt.scatter(df[0].values, df[1].values, c='r')
    plt.plot(df[0].values, LSE_func, c='black')
    
    plt.subplot(312)
    plt.scatter(df[0].values, df[1].values, c='r')
    plt.plot(df[0].values, steepest_func, c='black')
    
    plt.subplot(313)
    plt.scatter(df[0].values, df[1].values, c='r')
    plt.plot(df[0].values, newton_func, c='black')
    
    plt.tight_layout()
    plt.show()