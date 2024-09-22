import numpy as np
import tensorflow as tf
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

# Calculate mean and variance for each class
means = np.zeros((10, x_train.shape[1]))
variances = np.zeros((10, x_train.shape[1]))

for i in range(10):
    class_i = x_train[y_train == i]
    means[i, :] = np.mean(class_i, axis=0)
    variances[i, :] = np.var(class_i, axis=0) + 0.001  # Add small constant to variance

# Naive Bayes classification
def naive_bayes_predict(x):
    log_probs = np.zeros(10)
    for i in range(10):
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * variances[i])) \
                   - 0.5 * np.sum(((x - means[i]) ** 2) / variances[i], axis=1)
        log_probs[i] = log_prob
    return np.argmax(log_probs, axis=0)

# Predict and evaluate accuracy
y_pred = np.array([naive_bayes_predict(x) for x in x_test])
accuracy = np.mean(y_pred == y_test)
print(f"Naive Bayes Classification accuracy is {accuracy * 100:.2f}%")
