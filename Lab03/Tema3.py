import numpy as np
from torchvision.datasets import MNIST
import time


def download_mnist(is_train: bool):
    dataset = MNIST(
        root='./data',
        transform=lambda x: np.array(x).flatten(),
        download=True,
        train=is_train
    )
    mnist_data=[]
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)


train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)
train_X= train_X / 255.0
test_X=test_X / 255.0

def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    for index, label in enumerate(labels):
        one_hot[index, label] = 1
    return one_hot


train_Y = one_hot_encode(train_Y)
test_Y = one_hot_encode(test_Y)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


input_size=784
hidden_size=100
output_size = 10
np.random.seed(42)

# Xavier Initialization slide 35 curs 6
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
b2 = np.zeros((1, output_size))


def forward(X):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def cross_entropy_loss(y_true, y_pred):
    cross_entropy_loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return cross_entropy_loss


# Backpropagation
def backward(X, y_true, z1, a1, z2, a2, learning_rate, lambd=0.01):
    global W1, b1, W2, b2
    m = X.shape[0]

    delta2 = a2 - y_true
    dW2 = np.dot(a1.T, delta2) / m + (lambd * W2 / m)
    db2 = np.sum(delta2, axis=0, keepdims=True) / m

    delta1 = np.dot(delta2, W2.T) * a1 * (1 - a1)
    dW1 = np.dot(X.T, delta1) / m + (lambd * W1 / m)
    db1 = np.sum(delta1, axis=0, keepdims=True) / m

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


def evaluate_accuracy(X, y):
    _, _, _, a2 = forward(X)
    predictions = np.argmax(a2, axis=1)
    true_classes = np.argmax(y, axis=1)
    return np.mean(predictions == true_classes)


def train_model(X, Y, X_val, Y_val, epochs=300, batch_size=64, learning_rate=0.02, target_accuracy=0.95):
    patience =5
    decay_factor = 0.8
    best_val_accuracy = 0
    epochs_no_improvement = 0
    m = X.shape[0]
    start_time = time.time()


    for epoch in range(epochs):
        indices = np.arange(m)
        np.random.shuffle(indices)
        X= X[indices]
        Y= Y[indices]

        for i in range(0, m, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = Y[i:i + batch_size]
            z1, a1, z2, a2 = forward(X_batch)
            backward(X_batch, y_batch, z1, a1, z2, a2, learning_rate)

        train_loss = cross_entropy_loss(Y, forward(X)[3])
        train_acc = evaluate_accuracy(X, Y)
        val_acc = evaluate_accuracy(X_val, Y_val)
        elapsed_time = time.time() - start_time

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Acc: {train_acc * 100:.2f}% - Val Acc: {val_acc * 100:.2f}% - Loss: {train_loss:.4f}")

       # if val_acc >= target_accuracy:
        #    print("Stopping training: Target validation accuracy achieved.")
         #   break
        if elapsed_time >= 300:
            print("Stop due to time limit")
            break

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= patience:
            learning_rate *= decay_factor
            epochs_no_improvement = 0
            print(f"Learning rate adjusted to {learning_rate}")

    return W1, b1, W2, b2



W1_trained, b1_trained, W2_trained, b2_trained = train_model(train_X, train_Y, test_X, test_Y)

test_accuracy = evaluate_accuracy(test_X, test_Y)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
