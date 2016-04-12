from controlnet import ControlNet
from tensornet import TensorNet
import numpy as np
import cv2
from inputdata import MNISTData
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
np.set_printoptions(threshold=np.nan)

print len(mnist.train.next_batch(20)[0][0])
im = mnist.train.next_batch(20)[0][9].reshape((28, 28, 1))
cv2.imwrite("hi.jpg", im * 255.0)

data = MNISTData()
batch = data.next_train_batch(20)
batch = data.next_train_batch(20)

cv2.imwrite("hi2.jpg", batch[0][9] * 255.0)
