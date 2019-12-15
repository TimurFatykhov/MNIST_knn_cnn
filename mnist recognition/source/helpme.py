import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import torchvision
import os

def load_mnist_8():
    """
    Загружает данные с цифрами
    
    
    Возвращает две переменные:
    -------------------------
    - X: матрица 1797x64
        картинки в виде векторов длинной 64
        
    - y: вектор длинной 1797
        true вектор, хранящий класс каждой картинки
        
    
    Пример использования:
    ---------------------
    >>> X, y = load_data()
    >>>
    """
    X, y = load_digits(return_X_y=True)
    return X, y

def load_fmnist():
    local = os.getcwd() # ../ in Linux
    root = os.path.join(local,'data', 'fmnist')
    if not os.path.exists(root):
        os.makedirs(root)
    fmnist = torchvision.datasets.FashionMNIST(root, train=True, download=True)
    x_train = fmnist.train_data.numpy()
    y_train = fmnist.train_labels.numpy()
    
    fmnist = torchvision.datasets.FashionMNIST(root, train=False, download=True)
    x_test = fmnist.test_data.numpy()
    y_test = fmnist.test_labels.numpy()
    
    x = np.concatenate((x_train, x_test))
    x = np.expand_dims(x, 1)
    y = np.concatenate((y_train, y_test))
    return x, y


def load_mnist():
    local = os.getcwd() # ../ in Linux
    root = os.path.join(local,'data', 'mnist')
    if not os.path.exists(root):
        os.makedirs(root)
    mnist = torchvision.datasets.MNIST(root, train=True, download=True)
    x_train = mnist.train_data.numpy()
    y_train = mnist.train_labels.numpy()
    
    mnist = torchvision.datasets.MNIST(root, train=False, download=True)
    x_test = mnist.test_data.numpy()
    y_test = mnist.test_labels.numpy()
    
    x = np.concatenate((x_train, x_test))
    x = np.expand_dims(x, 1)
    y = np.concatenate((y_train, y_test))
    return x, y


def calculate_pad(input_size, kernel_size, stride, output_size):
    pad = output_size * stride - input_size + kernel_size - 1
    pad /= 2
    if int(pad) != pad:
        print('С такими параметрами нереально подобрать размер pad-а!')
    else:
        return int(pad)


def show_image(img, figsize=(5,5)):
    """
    Показывает изображение
    
    Параметры:
    - img: numpy.array
        массив numpy, с тремя или одним каналом (цветное или ч/б фото)
    """
    if len(img.shape) < 2:
        s = np.sqrt(len(img)).astype(int)
        img = img.reshape((s,s))
        
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
def show_history(train_history, valid_history=None, hide_left=0, figsize=None, fontsize=30, title=None, width=4):
    if figsize is not None:
        plt.figure(figsize=figsize)
        
    N = len(train_history)
    plt.plot(np.arange(hide_left, N), train_history[hide_left:], color='blue', label='train', linewidth=width)
    
    if valid_history is not None:
        plt.plot(np.arange(hide_left, N), valid_history[hide_left:], color='green', label='val', linewidth=width)
        
    plt.title(title)
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.show()
        
    
        
    
    
    
    
    
    