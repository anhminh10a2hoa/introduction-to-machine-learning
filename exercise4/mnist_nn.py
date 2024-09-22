import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sys

# Load MNIST dataset using TensorFlow
mnist = tf.keras.datasets.mnist

# Argument to choose between original MNIST and Fashion MNIST
dataset_type = sys.argv[1] if len(sys.argv) > 1 else 'original'

if dataset_type == 'fashion':
    mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data from 28x28 to 784
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Initialize 1-NN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

# Predict on test set
y_pred = knn.predict(x_test)

# Compute and print classification accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification accuracy is {accuracy * 100:.2f}%")
