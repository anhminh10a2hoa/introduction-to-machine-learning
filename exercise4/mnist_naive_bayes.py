import numpy as np
import tensorflow as tf

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

# Naive Bayes training (computes mean and variance)
def naive_bayes_train(x_train, y_train):
    classes = np.unique(y_train)
    mean, var = [], []
    
    for c in classes:
        x_class = x_train[y_train == c]
        mean.append(np.mean(x_class, axis=0))
        var.append(np.var(x_class, axis=0) + 0.001)  # Avoid zero variance by adding 0.001
    
    return np.array(mean), np.array(var)

# Naive Bayes prediction (log likelihood calculation)
def naive_bayes_predict(x_test, mean, var):
    log_likelihoods = []
    for c in range(10):
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var[c]))
        log_likelihood -= 0.5 * np.sum(((x_test - mean[c]) ** 2) / var[c], axis=1)
        log_likelihoods.append(log_likelihood)
    
    return np.argmax(log_likelihoods, axis=0)

# Function to add noise to the data
def add_noise(x_train, noise_level=0.1):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=x_train.shape)
    return x_train + noise

# Main function to train and evaluate the Naive Bayes classifier
def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train_flat, x_test_flat = flatten_data(x_train), flatten_data(x_test)
    
    # Optionally, add noise to training data
    x_train_noisy = add_noise(x_train_flat, noise_level=0.1)
    
    # Train Naive Bayes classifier
    mean, var = naive_bayes_train(x_train_noisy, y_train)
    
    # Predict on test data
    y_pred = naive_bayes_predict(x_test_flat, mean, var)
    
    # Compute accuracy using the class_acc function
    accuracy = class_acc(y_pred, y_test)
    print(f"Naive Bayes Classifier accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
