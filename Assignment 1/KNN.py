import numpy as np


class KNN (object):
    def __init__(self):
        pass

    def train(self, train_data, train_labels):
        self.data_train = train_data
        self.labels_train = train_labels

    def compute_distances_l1(self, data_test):
        num_test = data_test.shape[0]
        num_train = self.data_train.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                distances[i, j] = np.abs(data_test[i]-self.data_train[j]).sum()

        return distances

    def compute_distances_l2(self, data_test):
        num_test = data_test.shape[0]
        num_train = self.data_train.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                distances[i, j] = np.sqrt(np.sum(np.square(data_test[i]-self.data_train[j])))
        return distances

    def predict_labels(self, distances, k=1):
        num_test = distances.shape[0]
        label_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_label = []
            closest_label = self.labels_train[np.argsort(distances[i])[:k]]
            label_pred[i] = np.argmax(np.bincount(closest_label))
        return label_pred
    def test(self, data, k=1, model='l1'):
        if  model == 'l1':
            distances = self.compute_distances_l1(data)

        elif model == 'l2':
            distances = self.compute_distances_l2(data)




