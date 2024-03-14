import os
import struct
import numpy as np

def load_data(data_dir):
    train_img_dir = os.path.join(data_dir, 'train-images-idx3-ubyte')
    test_img_dir = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    train_labels_dir = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    test_labels_dir = os.path.join(data_dir, 't10k-labels-idx1-ubyte')

    with open(train_labels_dir, 'rb') as file:
        magic, num = struct.unpack(">II", file.read(8))
        train_labels = np.fromfile(file, dtype=np.uint8)

    with open(train_img_dir, 'rb') as file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        train_data = np.fromfile(file, dtype=np.uint8).reshape(len(train_labels), rows * cols).astype('float')

    with open(test_labels_dir, 'rb') as file:
        magic, num = struct.unpack(">II", file.read(8))
        test_labels = np.fromfile(file, dtype=np.uint8)

    with open(test_img_dir, 'rb') as file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        test_data = np.fromfile(file, dtype=np.uint8).reshape(len(test_labels), rows * cols).astype('float')

    return train_data, train_labels, test_data, test_labels