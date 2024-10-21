import numpy as np
from torchvision.datasets import MNIST

def download_mnist(is_train: bool):
    dataset = MNIST(
        root='./data',
        transform=lambda x: np.array(x).flatten(),
        download=True,
        train=is_train
    )
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)





train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

train_X = train_X /255.0
test_X = test_X /255.0

def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    for idex, label in enumerate(labels):
        one_hot[idex, label] = 1  
    return one_hot
train_Y = one_hot_encode(train_Y)
test_Y = one_hot_encode(test_Y)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(predictions, y_true):
    loss = -np.sum(y_true * np.log(predictions))
    return loss / y_true.shape[0]  

input= 784 
output= 10  
np.random.seed(20)
W = np.random.randn(input, output) * 0.01  
b = np.zeros((output,)) 

learning_rate = 0.01
num_epochs = 100
batch_size = 100

def update_weights_and_biases(X, y_true, W, b, learning_rate):
    m = X.shape[0] 
    z = np.dot(X, W) + b  
    predictions = softmax(z) 
    error = y_true - predictions  

    loss = cross_entropy_loss(predictions, y_true)

    W += learning_rate * np.dot(X.T, error) / m
    b += learning_rate * np.mean(error, axis=0)

    return W, b, loss 
    
def evaluate_accuracy(X, y, W, b):
    z = np.dot(X, W) + b
    predictions = softmax(z)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    return accuracy

initial_accuracy = evaluate_accuracy(test_X, test_Y, W, b)
print(f'Initial Accuracy: {initial_accuracy * 100:.2f}%')

for epoch in range(num_epochs):
    indices = np.arange(train_X.shape[0])
    np.random.shuffle(indices)
    train_X_shuffled = train_X[indices]
    train_Y_shuffled = train_Y[indices]

    for i in range(0, train_X.shape[0], batch_size):
        X_batch = train_X_shuffled[i:i + batch_size]
        y_batch = train_Y_shuffled[i:i + batch_size]

        W, b, loss = update_weights_and_biases(X_batch, y_batch, W, b, learning_rate)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}')



final_accuracy = evaluate_accuracy(test_X, test_Y, W, b)
print(f'Final Accuracy: {final_accuracy * 100:.2f}%')
