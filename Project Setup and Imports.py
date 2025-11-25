
#1. Project Setup and Imports

#First, we import the necessary libraries. TensorFlow/Keras is the core deep learning framework, and Matplotlib is used for visualization.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os

# Set hyperparameters
NUM_CLASSES = 10
EPOCHS = 8  # Increased epochs for better accuracy (feel free to experiment)
BATCH_SIZE = 128
IMG_SHAPE = (28, 28, 1)

print(f"TensorFlow Version: {tf.__version__}")
# Disable TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
