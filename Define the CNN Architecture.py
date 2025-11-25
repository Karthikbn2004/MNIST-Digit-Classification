#4. Define the CNN Architecture

#We use a standard, yet effective, architecture for image classification: two sets of Convolutional and Pooling layers, followed by a Flatten layer and Dense layers for classification.

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        # Layer 1: Convolution + ReLU Activation
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        # Layer 2: Max Pooling (downsamples feature maps)
        MaxPooling2D(pool_size=(2, 2)),

        # Layer 3: Second Convolution + ReLU
        Conv2D(64, (3, 3), activation='relu'),
        # Layer 4: Second Max Pooling
        MaxPooling2D(pool_size=(2, 2)),

        # Layer 5: Flatten for Dense layers
        Flatten(),

        # Layer 6: Fully Connected (Dense) hidden layer
        Dense(128, activation='relu'),

        # Layer 7: Output layer (10 classes) with Softmax for probability scores
        Dense(num_classes, activation='softmax')
    ])
    return model

model = create_cnn_model(IMG_SHAPE, NUM_CLASSES)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
