import argparse
import numpy as np
from skimage import io
from skimage.transform import resize
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags, csr_matrix
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

class GraphParams:
    def __init__(self, radius=5, sigma_x=5, sigma_i=10):
        self.radius = radius
        self.sigma_x = sigma_x
        self.sigma_i = sigma_i

def load_image(path, max_side=None):
    img = io.imread(path)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    elif img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]
    if max_side is not None:
        scale = max_side / max(img.shape[:2])
        img = resize(img, (int(img.shape[0]*scale), int(img.shape[1]*scale)), anti_aliasing=True)
    return img

def build_affinity(img, params):
    H, W = img.shape[:2]
    C = img.shape[2] if img.ndim == 3 else 1
    N = H * W

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords = np.column_stack((yy.ravel(), xx.ravel()))
    features = img.reshape(-1, C).astype(np.float32)

    tree = cKDTree(coords)
    neighbor_list = tree.query_ball_point(coords, params.radius)

    rows, cols = [], []
    for i, nbrs in enumerate(neighbor_list):
        for j in nbrs:
            if i < j:
                rows.append(i)
                cols.append(j)

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)

    spatial_diff = coords[rows] - coords[cols]
    spatial_dist_sq = np.sum(spatial_diff**2, axis=1)

    intensity_diff = features[rows] - features[cols]
    intensity_dist_sq = np.sum(intensity_diff**2, axis=1)

    vals = np.exp(-spatial_dist_sq / (2 * params.sigma_x**2)) * \
           np.exp(-intensity_dist_sq / (2 * params.sigma_i**2))

    Wmat = csr_matrix((vals, (rows, cols)), shape=(N, N))
    Wmat = Wmat + Wmat.T

    d = np.array(Wmat.sum(axis=1)).flatten()
    return Wmat, d

def run_2way(img, params):
    W, d = build_affinity(img, params)
    D = diags(d)
    L = D - W

    vals, vecs = eigsh(L, k=2, M=D, which='SM')
    fiedler = vecs[:, 1]

    threshold = np.median(fiedler)  # median threshold ensures balanced split
    mask = fiedler > threshold
    segmentation = mask.reshape(img.shape[:2])
    return segmentation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='path to image')
    parser.add_argument('--max_side', type=int, default=200, help='max side length')
    args = parser.parse_args()

    img = load_image(args.image, max_side=args.max_side)

    H, W = img.shape[:2]
    max_intensity = img.max() if img.max() > 0 else 1
    params = GraphParams(
        radius=max(5, max(H, W)//20),       # 5–10% of image size
        sigma_x=max(1, max(H, W)/50),       # spatial scale
        sigma_i=max(0.01, max_intensity/10) # intensity scale
    )

    seg = run_2way(img, params)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(seg, cmap='gray')
    plt.title('2-way Ncut')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
