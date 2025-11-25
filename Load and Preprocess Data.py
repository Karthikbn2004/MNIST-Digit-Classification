
#2. Load and Preprocess Data

#The MNIST dataset is built into Keras, making it easy to load. We must normalize the pixel values and reshape the images for the CNN.

# Load the dataset
print("Loading and preparing MNIST data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 1. Normalize Images (Scale pixel values from [0-255] to [0.0-1.0])
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 2. Reshape for CNN Input (Add the channel dimension: 28x28 -> 28x28x1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 3. One-Hot Encode Labels (Convert integer labels to vectors)
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

print(f"x_train shape after preprocessing: {x_train.shape}")
print(f"y_train shape after preprocessing: {y_train.shape}")
