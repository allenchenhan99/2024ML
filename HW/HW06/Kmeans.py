import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cdist
import os




def compute_kernel(image, gamma_s, gamma_c):
    n_rows, n_cols, _ = image.shape
    color_distance = cdist(image.reshape(n_rows * n_cols, 3), image.reshape(n_rows * n_cols, 3), 'sqeuclidean')
    grid = np.indices((n_rows, n_cols))
    row_indices, col_indices = grid[0], grid[1]
    indices = np.hstack((row_indices.reshape(-1, 1), col_indices.reshape(-1, 1)))
    spatial_distance = cdist(indices, indices, 'sqeuclidean')
    return np.multiply(np.exp(-gamma_s * spatial_distance), np.exp(-gamma_c * color_distance))




def choose_center(n_rows, n_cols, n_clusters, mode):
    if not mode:
        return np.random.choice(100, (n_clusters, 2))
    else:
        grid = np.indices((n_rows, n_cols))
        row_indices, col_indices = grid[0], grid[1]
        indices = np.hstack((row_indices.reshape(-1, 1), col_indices.reshape(-1, 1)))
        n_points = n_rows * n_cols
        centers = [indices[np.random.choice(n_points, 1)[0]].tolist()]
        for _ in range(n_clusters - 1):
            distance = np.zeros(n_points)
            for idx, point in enumerate(indices):
                min_distance = np.Inf
                for center in centers:
                    dist = np.linalg.norm(point - center)
                    min_distance = dist if dist < min_distance else min_distance
                distance[idx] = min_distance
            distance /= np.sum(distance)
            centers.append(indices[np.random.choice(n_points, 1, p=distance)[0]].tolist())
        return np.array(centers)

def init_clustering(n_rows, n_cols, n_clusters, kernel, mode):
    centers = choose_center(n_rows, n_cols, n_clusters, mode)
    n_points = n_rows * n_cols
    cluster = np.zeros(n_points, dtype=int)
    for p in range(n_points):
        distance = np.zeros(n_clusters)
        for idx, center in enumerate(centers):
            seq_center = center[0] * n_rows + center[1]
            distance[idx] = kernel[p, p] + kernel[seq_center, seq_center] - 2 * kernel[p, seq_center]
        cluster[p] = np.argmin(distance)
    return clusters



def get_sum_of_pairwise_distance(n_points, n_clusters, n_members, kernel, cluster):
    pairwise_distance = np.zeros(n_clusters)
    for c in range(n_clusters):
        tmp_kernel = kernel.copy()
        for p in range(n_points):
            if cluster[p] != c:
                tmp_kernel[p, :] = 0
                tmp_kernel[:, p] = 0
        pairwise_distance[c] = np.sum(tmp_kernel)
    n_members[n_members == 0] = 1
    return pairwise_distance / n_members ** 2

def kernel_clustering(n_points, n_clusters, kernel, cluster):
    n_members = np.array([np.sum(np.where(cluster == c, 1, 0)) for c in range(n_clusters)])
    pairwise_distance = get_sum_of_pairwise_distance(n_points, n_clusters, n_members, kernel, cluster)
    new_cluster = np.zeros(n_points, dtype=int)
    for p in range(n_points):
        distance = np.zeros(n_clusters)
        for c in range(n_clusters):
            distance[c] += kernel[p, p] + pairwise_distance[c]
            distance2others = np.sum(kernel[p, :][np.where(cluster == c)])
            distance[c] -= 2.0 / n_members[c] * distance2others
        new_cluster[p] = np.argmin(distance)
    return new_cluster

def capture_current_state(n_rows, n_cols, cluster, colors):
    state = np.zeros((n_rows * n_cols, 3))
    for p in range(n_rows * n_cols):
        state[p, :] = colors[cluster[p], :]
    state = state.reshape((n_rows, n_cols, 3))
    return Image.fromarray(np.uint8(state))

def kernel_kmeans(n_rows, n_cols, n_clusters, cluster, kernel, mode, index):
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    if n_clusters > 3:
        colors = np.append(colors, np.random.choice(256, (n_clusters - 3, 3)), axis=0)
    img = [capture_current_state(n_rows, n_cols, cluster, colors)]
    current_cluster = cluster.copy()
    count = 0
    iteration = 1000
    while True:
        new_cluster = kernel_clustering(n_rows * n_cols, n_clusters, kernel, current_cluster)
        img.append(capture_current_state(n_rows, n_cols, new_cluster, colors))
        if np.linalg.norm((new_cluster - current_cluster), ord=2) < 0.001 or count >= iteration:
            break
        current_cluster = new_cluster.copy()
        count += 1
    filename = f'./gifs/kernel_kmeans/image{index+1}_cluster={n_clusters}_{"kmeans" if mode else "random"}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img[0].save(filename, save_all=True, append_images=img[1:], optimize=False, loop=0, duration=100)





def parse_arguments():
    parser = argparse.ArgumentParser(description='kernel k-means')
    parser.add_argument('--n_cluster', default=3, type=int, help='how many group do we want?')
    parser.add_argument('--mode', default=0, type=int, help='0: randomly pick, 1: k-means++')
    parser.add_argument('--gamma_s', default=0.0001, type=float, help='Spatial similarity hyperparameter')
    parser.add_argument('--gamma_c', default=0.001, type=float, help='Color similarity hyperparameter')
    
    return parser.parse_args()





if __name__ == "__main__":
    args = parse_arguments()
    images = [Image.open('image1.png'), Image.open('image2.png')]
    images[0] = np.asarray(images[0])
    images[1] = np.asarray(images[1])
    n_cluster = args.n_cluster
    mode = args.mode
    gamma_s = args.gamma_s
    gamma_c = args.gamma_c
    
    for i, image in enumerate(images):
        gram_matrix = compute_kernel(image, gamma_s, gamma_c)
        rows, cols, _ = image.shape
        clusters = init_clustering(rows, cols, n_cluster, gram_matrix, mode)
        kernel_kmeans(rows, cols, n_cluster, clusters, gram_matrix, mode, i)