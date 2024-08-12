import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.spatial.distance import cdist
from PIL import Image
  
    
def simple_PCA(num_image, images):
    image_transpose = images.T
    mean = np.mean(image_transpose, axis=1)
    mean = np.tile(mean.T, (num_image, 1)).T
    difference = image_transpose - mean
    covariance = difference.dot(difference.T) / num_image
    return covariance
    
def kernel_PCA(images, kernel_type, gamma):
    if kernel_type == 'linear':
        kernel = images.T.dot(images)
    elif kernel_type == 'rbf':
        kernel = np.exp(-gamma * cdist(images.T, images.T, 'sqeuclidean'))
    else:
        raise BaseException(f'Invalid kernel type. The kernel type should be linear or rbf')
    
    matrix_n = np.ones((29 * 24, 29 * 24), dtype=float) / (29 * 24)
    matrix = kernel - matrix_n.dot(kernel) - kernel.dot(matrix_n) + matrix_n.dot(kernel).dot(matrix_n)
    return matrix
    
def PCA(train_images, train_labels, test_images, test_labels, mode, k_neighbors, kernel_type, gamma):
    num_train = len(train_images)
    if mode == 'simple':
        matrix = simple_PCA(num_train, train_images)
    elif mode == 'kernel':
        matrix = kernel_PCA(train_images, kernel_type, gamma)
    else:
        raise BaseException(f'Invalid mode. The mode should be simple or kernel')
    
    eigenvec = find_eigenvector(matrix)
    eigenface(eigenvec, 0)
    reconstruction(num_train, train_images, eigenvec)
    classify(num_train, len(test_images), train_images, train_labels, test_images, test_labels, eigenvec, k_neighbors)
    plt.tight_layout()
    plt.show()
         
         
def simple_LDA(num_of_each_class, images, labels):
    overall_mean = np.mean(images, axis=0)
    n_class = len(num_of_each_class)
    class_mean = np.zeros((n_class, 29 * 24))
    for label in range(n_class):
        class_mean[label, :] = np.mean(images[labels == label + 1], axis=0)
    
    scatter_b = np.zeros((29 * 24, 29 * 24), dtype=float)
    for idx, num in enumerate(num_of_each_class):
        difference = (class_mean[idx] - overall_mean).reshape((29 * 24, 1))
        scatter_b += num * difference.dot(difference.T)
    
    scatter_w = np.zeros((29 * 24, 29 * 24), dtype=float)
    for idx, mean in enumerate(class_mean):
        difference = images[labels == idx + 1] - mean
        scatter_w += difference.T.dot(difference)
    
    matrix = np.linalg.pinv(scatter_w).dot(scatter_b)
    return matrix
    
def kernel_LDA(num_of_each_class, images, labels, kernel_type, gamma):
    n_class = len(num_of_each_class)
    n_image = len(images)
    
    if kernel_type == 'linear':
        kernel_of_each_class = np.zeros((n_class, 29 * 24, 29 * 24))
        for idx in range(n_class):
            image = images[labels == idx + 1]
            kernel_of_each_class[idx] = image.T.dot(image)
        kernel_of_all = images.T.dot(images)
    elif kernel_type == 'rbf':
        kernel_of_each_class = np.zeros((n_class, 29 * 24, 29 * 24))
        for idx in range(n_class):
            image = images[labels == idx + 1]
            kernel_of_each_class[idx] = np.exp(-gamma * cdist(image.T, image.T, 'sqeuclidean'))
        kernel_of_all = np.exp(-gamma * cdist(images.T, images.T, 'sqeuclidean'))
    else:
        raise BaseException(f'Invalid kernel type. The kernel type should be linear or rbf')
    
    matrix_n = np.zeros((29 * 24, 29 * 24))
    identity_matrix = np.eye(29 * 24)
    for idx, num in enumerate(num_of_each_class):
        matrix_n += kernel_of_each_class[idx].dot(identity_matrix - num * identity_matrix).dot(kernel_of_each_class[idx].T)
    
    matrix_m_i = np.zeros((n_class, 29 * 24))
    for idx, kernel in enumerate(kernel_of_each_class):
        for row_idx, row in enumerate(kernel):
            matrix_m_i[idx, row_idx] = np.sum(row) / num_of_each_class[idx]
    
    matrix_m_star = np.zeros(29 * 24)
    for idx, row in enumerate(kernel_of_all):
        matrix_m_star[idx] = np.sum(row) / n_image
    
    matrix_m = np.zeros((29 * 24, 29 * 24))
    for idx, num in enumerate(num_of_each_class):
        difference = (matrix_m_i[idx] - matrix_m_star).reshape((29 * 24, 1))
        matrix_m += num * difference.dot(difference.T)
    
    matrix = np.linalg.pinv(matrix_n).dot(matrix_m)
    return matrix
    
def LDA(train_images, train_labels, test_images, test_labels, mode, k_neighbors, kernel_type, gamma):
    n_train = len(train_images)
    _, num_of_each_class = np.unique(train_labels, return_counts=True)
    
    if mode == 'simple':
        matrix = simple_LDA(num_of_each_class, train_images, train_labels)
    elif mode == 'kernel':
        matrix = kernel_LDA(num_of_each_class, train_images, train_labels, kernel_type, gamma)
    else:
        raise BaseException(f'Invalid mode. The mode should be simple or kernel')
    
    target_eigenvectors = find_eigenvector(matrix)
    eigenface(target_eigenvectors, 1)
    reconstruction(n_train, train_images, target_eigenvectors)
    classify(n_train, len(test_images), train_images, train_labels, test_images, test_labels, target_eigenvectors, k_neighbors)
    plt.tight_layout()
    plt.show()


def find_eigenvector(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    idx = np.argsort(eigenvalues)[::-1][:25]
    eigenvec = eigenvectors[:, idx].real
    return eigenvec

def eigenface(eigenvectors, mode):
    faces = eigenvectors.T.reshape((25, 29, 24))
    fig = plt.figure(1)
    # fig.canvas.set_window_title(f'{"Eigenfaces" if mode == 0 else "Fisherfaces"}')
    fig.suptitle(f'{"Eigenfaces" if mode == 0 else "Fisherfaces"}')
    for idx in range(25):
        plt.subplot(5, 5, idx + 1)
        plt.axis('off')
        plt.imshow(faces[idx, :, :], cmap='gray')
    

def reconstruction(num_image, images, eigenvec):
    reconstruct_image = np.zeros((10, 29 * 24))
    choice = np.random.choice(num_image, 10)
    for idx in range(10):
        reconstruct_image[idx, :] = images[choice[idx], :].dot(eigenvec).dot(eigenvec.T)
    
    fig = plt.figure(2)
    # fig.canvas.set_window_title('Reconstructed faces')
    fig.suptitle('Reconstructed faces')
    for idx in range(10):
        plt.subplot(10, 2, idx * 2 + 1)
        plt.axis('off')
        plt.imshow(images[choice[idx], :].reshape(29, 24), cmap='gray')
        
        plt.subplot(10, 2, idx * 2 + 2)
        plt.axis('off')
        plt.imshow(reconstruct_image[idx, :].reshape(29, 24), cmap='gray')

def decorrelate(num_image, images, eigenvec):
    decorrlated_image = np.zeros((num_image, 25))
    for idx, image in enumerate(images):
        decorrlated_image[idx, :] = image.dot(eigenvec)
    return decorrlated_image

def classify(num_train, num_test, train_images, train_labels, test_images, test_labels, eigenvec, k_neighbors):
    decorrelate_train = decorrelate(num_train, train_images, eigenvec)
    decorrelate_test = decorrelate(num_test, test_images, eigenvec)
    
    error = 0
    distance = np.zeros(num_train)
    
    for test_idx, test_image in enumerate(decorrelate_test):
        for train_idx, train_image in enumerate(decorrelate_train):
            distance[train_idx] = np.linalg.norm(test_image - train_image)
        
        min_distance = np.argsort(distance)[:k_neighbors]
        predict = np.argmax(np.bincount(train_labels[min_distance]))
        if predict != test_labels[test_idx]:
            error += 1
    print(f'Error count: {error}\nError rate: {float(error) / num_test}')


def read_images_from_directory(directory_path, image_size=(24, 29)):
    """
    讀取指定目錄中的圖像和標籤。
    
    參數：
    directory_path (str): 圖像目錄的路徑。
    image_size (tuple): 圖像大小，預設為 (24, 29)。
    
    返回：
    images (numpy.ndarray): 讀取的圖像數據。
    labels (numpy.ndarray): 對應的標籤數據。
    """
    n_files = 0
    with os.scandir(directory_path) as directory:
        n_files = len([file for file in directory if file.is_file()])

    images = np.zeros((n_files, image_size[0] * image_size[1]))
    labels = np.zeros(n_files, dtype=int)

    with os.scandir(directory_path) as directory:
        for idx, file in enumerate(directory):
            if file.path.endswith('.pgm') and file.is_file():
                face = np.asarray(Image.open(file.path).resize(image_size)).reshape(1, -1)
                images[idx, :] = face
                labels[idx] = int(file.name[7:9])
    
    # images: 二維numpy數組，形狀為(n_files, images[0]*images[1])，每一列儲存flatten後的圖像數據
    # labels: 一為numpy數組，長度為n_files，對應的labels
    return images, labels


def parse_arguments():
    parser = ArgumentParser(description='kernel_eigenfaces')
    parser.add_argument('--algo', default='PCA', help='PCA, LDA', type=str)
    parser.add_argument('--mode', default='kernel', help='simple, kernel', type=str)
    parser.add_argument('--k_neighbors', default=10, type=int)
    parser.add_argument('--type_kernel', default='rbf', help='linear, rbf', type=str)
    parser.add_argument('--gamma', default=1e-6, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    filename = './Yale_Face_Database/Yale_Face_Database'
    algo = args.algo
    mode = args.mode
    k_neighbors = args.k_neighbors
    kernel_type = args.type_kernel
    gamma = args.gamma
    
    train_images, train_labels = read_images_from_directory(f'{filename}/Training', image_size=(24, 29))
    test_images, test_labels = read_images_from_directory(f'{filename}/Testing', image_size=(24, 29))
    
    if algo == 'PCA':
        PCA(train_images, train_labels, test_images, test_labels, mode, k_neighbors, kernel_type, gamma)
    else:
        LDA(train_images, train_labels, test_images, test_labels, mode, k_neighbors, kernel_type, gamma)