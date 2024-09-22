import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal

# Function to compute classification accuracy
def class_acc(pred, gt):
    correct = np.sum(pred == gt)
    total = len(gt)
    accuracy = correct / total
    return accuracy

# Load MNIST dataset
def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
    return (x_train, y_train), (x_test, y_test)

# Function to reshape the data
def flatten_data(x):
    return x.reshape(len(x), -1)  # Flatten 28x28 into 784

# Full Bayes training (computes mean and covariance matrix)
def bayes_train_full(x_train, y_train):
    classes = np.unique(y_train)
    mean, cov = [], []
    
    for c in classes:
        x_class = x_train[y_train == c]
        mean.append(np.mean(x_class, axis=0))
        cov_matrix = np.cov(x_class.T) + 0.001 * np.eye(x_class.shape[1])  # Add small value to diagonal
        cov.append(cov_matrix)
    
    return np.array(mean), np.array(cov)

# Full Bayes prediction using multivariate normal distribution
def bayes_predict_full(x_test, mean, cov):
    log_likelihoods = []
    for c in range(10):
        log_likelihood = multivariate_normal.logpdf(x_test, mean=mean[c], cov=cov[c])
        log_likelihoods.append(log_likelihood)
    
    return np.argmax(log_likelihoods, axis=0)

# Function to add noise to the data
def add_noise(x_train, noise_level=0.1):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=x_train.shape)
    return x_train + noise

# Main function to train and evaluate the Full Bayes classifier
def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train_flat, x_test_flat = flatten_data(x_train), flatten_data(x_test)
    
    # Optionally, add noise to training data
    x_train_noisy = add_noise(x_train_flat, noise_level=0.1)
    
    # Train Full Bayes classifier
    mean, cov = bayes_train_full(x_train_noisy, y_train)
    
    # Predict on test data
    y_pred = bayes_predict_full(x_test_flat, mean, cov)
    
    # Compute accuracy using the class_acc function
    accuracy = class_acc(y_pred, y_test)
    print(f"Full Bayes Classifier accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
