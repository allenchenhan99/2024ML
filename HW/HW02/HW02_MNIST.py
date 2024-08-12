import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
import os, codecs

# Function
    # Discrete mode
    # Continuous mode

# For training ()
    # train-images.idx3-ubyte : 28 * 28 pixels
    # train-labels.idx1-ubyte : true value
# For testing
    # t10k-images.idx3-ubyte : 28 * 28 pixels
    # t10k-labels.idx1-ubyte : true value
    
# if torch.cuda.is_available():
#     device_name = torch.cuda.get_device_name(0)
#     print("Using GPU:", device_name)
# else:
#     print("Using CPU")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file = [
    "t10k-images.idx3-ubyte",
    "t10k-labels.idx1-ubyte",
    "train-images.idx3-ubyte",
    "train-labels.idx1-ubyte",]


# label : 0~4-> 2049   4~8-> 60000
def b2i(b):
    return int(codecs.encode(b, 'hex'), 16)


# Prior
def get_prior(label):
    # 10 labels
    prior = np.zeros(10, dtype=float)
    # prior = torch.tensor(prior).to(device)
    
    for i in range(len(label)):
        prior[label[i]] += 1
    
    return prior / len(label)


# Likelihood
def get_likelihood(data, label):
    # 10 labels, data 28 * 28, 32 gray level
    likelihood = np.zeros((10, len(data[0]), 32), dtype=float)
    # likelihood = torch.tensor(likelihood).to(device)
    
    for i in range(len(data)):
        for pixel in range(len(data[0])):
            # pixel index is from 0 ~ 28*28
            likelihood[label[i], pixel, data[i][pixel] // 8] += 1
            
    # Frequency , for normalization
    total_num = np.sum(likelihood, axis=2)
    # total_num = torch.tensor(total_num).to(device)

    for l in range(10):
        for pixel in range(len(data[0])):
            likelihood[l, pixel, :] /= total_num[l, pixel]
    
    # pseudo count
    # improve accuracy
    likelihood[likelihood == 0] = 0.00001
    
    return likelihood

def discrete_classifier(x_train, y_train, x_test, y_test):
    prior = get_prior(y_train)
    likelihood = get_likelihood(x_train, y_train)
    
    wrong_predict = 0
    posteriors = []
    
    # Posterior
    for i in range(len(x_test)):
        posterior = np.log(prior)
        
        for l in range(10):
            for pixel in range(len(x_test[0])):
                posterior[l] += np.log(likelihood[l, pixel, x_test[i][pixel] // 8])
                
        posterior /= np.sum(posterior)
        posteriors.append(posterior)
        
        # Maximize a Posterior -> find min because posterior is positive
        predict = np.argmin(posterior)
        if predict != y_test[i]:
            wrong_predict += 1
    
    return posteriors, likelihood, float(wrong_predict / len(x_test))


def MLE_Gaussian(data, label, weight):
    # compute the mean and variance of each pixel in each class
    label_num = weight * len(data)
    
    # mean
    mean = np.zeros((10, len(data[0])), dtype=float)
    
    for i in range(len(data)):
        mean[label[i], :] += data[i, :]
    for l in range(10):
        mean[l, :] /= label_num[l]
    
    
    # variance
    variance = np.zeros((10, len(data[0])), dtype=float)
    
    for i in range(len(data)):
        variance[label[i], :] += np.square(data[i, :] - mean[label[i], :])
    for l in range(10):
        variance[l, :] /= label_num[l]
    
    return mean, variance


def continuous_classifier(x_train, y_train, x_test, y_test):
    # using the mean and variance of the Gaussian distribution to compute posterior
    
    prior = get_prior(y_train)
    
    # get mean and variance
    mean, variance = MLE_Gaussian(x_train, y_train, prior)
    
    # compute the posterior
    wrong_predict = 0
    posteriors = []
    
    for i in range(len(x_test)):
        posterior = np.log(prior)
        
        for l in range(10):
            for pixel in range(len(x_test[0])):
                if variance[l, pixel] == 0:
                    continue
                posterior[l] -= np.log(variance[l, pixel]) / 2.0
                posterior[l] -= np.square(x_test[i, pixel] - mean[l, pixel]) / variance[l, pixel]
                
        posterior /= np.sum(posterior)
        posteriors.append(posterior)
        
        # Maximize a Posterior -> find min because posterior is positive
        predict = np.argmin(posterior)
        if predict != y_test[i] :
            wrong_predict += 1
    
    return posteriors, mean, float(wrong_predict) / len(x_test)


def get_parser():
    parser = argparse.ArgumentParser(description="HW02_MNIST")
    parser.add_argument("--mode", default=0, type=int, help="discrete=0, continuous=1")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    mode = args.mode
    
    # load data
    # x_train, y_train, x_test, y_test respectively
    
    # x_train
    with open("." + os.sep + "train-images.idx3-ubyte_", 'rb') as f:
        data = f.read()
        type = b2i(data[0:4])
        dataLen = b2i(data[4:8])
        train_row = b2i(data[8:12])
        train_col = b2i(data[12:16])
        
        # data start from pos=16
        x_train = np.frombuffer(data, dtype=np.uint8, offset=16)
        x_train = x_train.reshape(dataLen, train_row * train_col)
        # x_train = torch.tensor(x_train).to(device)

    # y_train
    with open("." + os.sep + "train-labels.idx1-ubyte_", 'rb') as f:
        data = f.read()
        type = b2i(data[0:4])
        dataLen = b2i(data[4:8])

        y_train = np.frombuffer(data, dtype=np.uint8, offset=8)
        y_train = y_train.reshape(dataLen)
        # y_train = torch.tensor(y_train).to(device)
    
    # x_test
    with open("." + os.sep + "t10k-images.idx3-ubyte_", 'rb') as f:
        data = f.read()
        type = b2i(data[0:4])
        dataLen = b2i(data[4:8])
        test_row = b2i(data[8:12])
        test_col = b2i(data[12:16])

        x_test = np.frombuffer(data, dtype=np.uint8, offset=16)
        x_test = x_test.reshape(dataLen, test_row * test_col)
        # x_test = torch.tensor(x_test).to(device)
    
    # y_test
    with open("." + os.sep + "t10k-labels.idx1-ubyte_", 'rb') as f:
        data = f.read()
        type = b2i(data[0:4])
        dataLen = b2i(data[4:8])

        y_test = np.frombuffer(data, dtype=np.uint8, offset=8)
        y_test = y_test.reshape(dataLen)
        # y_test = torch.tensor(y_test).to(device)

    
    if mode == 0:
        # discrete mode
        posteriors, likelihood, wrong_rate = discrete_classifier(x_train, y_train, x_test, y_test)
        
        # displsay results
        # Posteriors
        for i in range(len(posteriors)):
            print("Posterior (in log scale):")
            for l in range(10):
                print(f"{l}: {posteriors[i][l]}")
            print(f"Prediction: {np.argmin(posteriors[i])}, Ans: {y_test[i]}\n")
            
        # print graph
        one = np.sum(likelihood[:, :, 16:32], axis=2)
        zero = np.sum(likelihood[:, :, 0:16], axis=2)
        graph = (one >= zero)
        
        for l in range(10):
            print(f'{l}:')
            for r in range(test_row):
                for c in range(test_col):
                    print("1", end=" ") if graph[l, r * test_col + c] else print("0", end=" ")
                print("")
            print("")
    else:
        posteriors, means, wrong_rate = continuous_classifier(x_train, y_train, x_test, y_test)
        
    # display results
    # Posteriors
        for i in range(len(posteriors)):
            print("Posterior (in log scale):")
            for l in range(10):
                print(f"{l}: {posteriors[i][l]}")
            print(f"Prediction: {np.argmin(posteriors[i])}, Ans: {y_test[i]}\n")
            
            
        # print graph
        graph = (means >= 128)

        for l in range(10):
            print(f"{l}:")
            for r in range(test_row):
                for c in range(test_col):
                    print("1", end=" ") if graph[l, r * test_col + c] else print("0", end=" ")
                print("")
            print("")
    print(f"Error rate: {wrong_rate}")