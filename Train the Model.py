#5. Train the Model

#We fit the model to the training data and use the test data for validation during training. The results are stored in the history object.

print("\nStarting model training...")
# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(x_test, y_test)
)
print("Model training complete.")
