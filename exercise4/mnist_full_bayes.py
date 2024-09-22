import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal
import sys

# Load MNIST dataset using TensorFlow
mnist = tf.keras.datasets.mnist

# Argument to choose between original MNIST and Fashion MNIST
dataset_type = sys.argv[1] if len(sys.argv) > 1 else 'original'

if dataset_type == 'fashion':
    mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data to vectors of size 784
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Add small noise to prevent zero variance
x_train = x_train + np.random.normal(loc=0.0, scale=0.1, size=x_train.shape)

# Compute the covariance matrices and means
covariances = np.zeros((10, x_train.shape[1], x_train.shape[1]))
means = np.zeros((10, x_train.shape[1]))

for i in range(10):
    class_i = x_train[y_train == i]
    means[i, :] = np.mean(class_i, axis=0)
    covariances[i] = np.cov(class_i, rowvar=False) + np.eye(x_train.shape[1]) * 0.001  # Add small noise

# Full Bayes classification
def bayes_predict(x):
    log_probs = np.zeros(10)
    for i in range(10):
        log_probs[i] = multivariate_normal.logpdf(x, mean=means[i], cov=covariances[i])
    return np.argmax(log_probs)

# Predict and evaluate accuracy
y_pred = np.array([bayes_predict(x) for x in x_test])
accuracy = np.mean(y_pred == y_test)
print(f"Full Bayes Classification accuracy is {accuracy * 100:.2f}%")
