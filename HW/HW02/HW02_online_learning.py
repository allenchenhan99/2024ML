import numpy as np
import argparse
import os

# Input file.txt and parameter "a" & "b"
    # a : 1
    # b : 0
    
# Beta-Binomial conjugation to perform online learning
# Function : Beta-Binomial conjugation
# Likelihood : Binomial
  
  
# Factorial 
def factorial(n):
    fact = 1
    for i in range(1, n + 1):
        fact = fact * i
    return fact

# Combination
def combination(n, m):
    return factorial(n) / (factorial(m) * factorial(n - m))

# Likelihood
def Likelihood(n, m, prob):
    return combination(n, m) * pow(prob, m) * pow(1 - prob, n - m)
    
# Beta-Binomial
def beta_binomial(a, b, data):
    for i, x in enumerate(data):
        one = x.count('1')
        zero = x.count('0')
     
        prob = float(one / (one + zero))
        likelihood = Likelihood(len(x), one, prob)
        
        posterior_a = a + one
        posterior_b = b + zero
        
        print(f"case {i + 1}: {x}")
        print(f"Likelihood: {likelihood}")
        print(f"Beta prior:\ta = {a}\tb = {b}")
        print(f"Beta posterior:\ta = {posterior_a}\tb = {posterior_b}")
        
        a = posterior_a
        b = posterior_b

# ArgumentParser(prog=None, usage=None, description=None, epilog=None)
def get_parser():
    
    parser = argparse.ArgumentParser(description= 'HW02_online_learning')
    parser.add_argument('--a', default=0, type=int, help='Input a')
    parser.add_argument('--b', default=0, type=int, help='Input b')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    a = args.a
    b = args.b
    
    # load data
    # data = ['0101', '...', ...]
    with open("." + os.sep + "test.csv", 'r') as f:
        data = f.read().splitlines()
    
    # python .\HW02_online_learning.py --a 0 --b 0
    beta_binomial(a, b, data)