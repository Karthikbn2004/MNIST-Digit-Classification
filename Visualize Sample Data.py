#3. Visualize Sample Data

#A good practice in any ML project is to visually inspect the data. Let's plot the first 9 images from the training set.

# Visualize the first 9 images
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    # The image is 28x28x1, so we drop the channel dimension for plotting
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    # Get the true label by reversing the one-hot encoding
    plt.title(f"Label: {np.argmax(y_train[i])}")
    plt.axis('off')
plt.show()
