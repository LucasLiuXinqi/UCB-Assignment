import matplotlib.pyplot as plt
import numpy as np
import os
from DataLoader import load_data
from KNN import KNN

current_dir = os.path.dirname(__file__)
mnist_dir = os.path.join(current_dir, 'DataSet', 'mnist')
train_data, train_labels, test_data, test_labels = load_data(mnist_dir)

# samples = train_data[:16]
# fig, axes = plt.subplots(4, 4, figsize=(8, 8))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(samples[i].reshape(28, 28), cmap='gray')
#     ax.axis('off')
# plt.show()

num_training = 5000
mask = list(range(num_training))
train_data = train_data[mask]
train_labels = train_labels[mask]

num_test = 500
mask = list(range(num_test))
test_data = test_data[mask]
test_labels = test_labels[mask]

train_data = train_data.reshape(train_data.shape[0], -1)
test_data = test_data.reshape(test_data.shape[0], -1)

classifier = KNN()
classifier.train(train_data, train_labels)

# k = 5
# model = 'l2'
# predicted_labels = classifier.test(test_data, k, model)
#
# num_correct = np.sum(predicted_labels == test_labels)
# accuracy = float(num_correct) / num_test
# print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

l1_x_axis = []
l1_y_axis = []
l2_x_axis = []
l2_y_axis = []

for k in range(1, 10):
    predicted_labels = classifier.test(test_data, k, 'l1')
    num_correct = np.sum(predicted_labels == test_labels)
    accuracy = num_correct / num_test
    l1_x_axis.append(k)
    l1_y_axis.append(accuracy)

    predicted_labels = classifier.test(test_data, k, 'l2')
    num_correct = np.sum(predicted_labels == test_labels)
    accuracy = num_correct / num_test
    l2_x_axis.append(k)
    l2_y_axis.append(accuracy)



plt.plot(l1_x_axis, l1_y_axis, label='l1', color='red')
plt.plot(l2_x_axis, l2_y_axis, label='l2', color='blue')
plt.legend()
plt.show()

