import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier

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
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the pixel values
    return (x_train, y_train), (x_test, y_test)

# Function to reshape the data
def flatten_data(x):
    return x.reshape(len(x), -1)  # Flatten the 28x28 images into 1x784 vectors

# Main function to train and evaluate the 1-NN classifier
def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    # Flatten the images
    x_train_flat = flatten_data(x_train)
    x_test_flat = flatten_data(x_test)
    
    # Create a 1-NN classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    
    # Train the classifier
    print("Training the 1-NN classifier...")
    knn.fit(x_train_flat, y_train)
    
    # Predict on the test data
    print("Predicting on the test data...")
    y_pred = knn.predict(x_test_flat)
    
    # Calculate the accuracy using the class_acc function
    accuracy = class_acc(y_pred, y_test)
    print(f"1-NN Classifier accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
