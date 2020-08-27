import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState

n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)
dataset = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))

faces = dataset.data

def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)

    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i+1)
        vmax = max()
