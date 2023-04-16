import tensorflow as tf
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv("wine_quality.csv")

# Split into train and test sets
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

# Split the features and target variable
train_features = train_data.drop(columns=["quality"])
train_target = train_data["quality"]
test_features = test_data.drop(columns=["quality"])
test_target = test_data["quality"]

# Scale the features
train_mean = train_features.mean(axis=0)
train_std = train_features.std(axis=0)
train_features = (train_features - train_mean) / train_std
test_features = (test_features - train_mean) / train_std

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[len(train_features.keys())]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.RMSprop(0.001), metrics=['mean_absolute_error', 'mean_squared_error'])

# Train the model
history = model.fit(train_features, train_target, epochs=100, validation_split=0.2, verbose=0)

# Evaluate the model on the test set
test_loss, test_mae, test_mse = model.evaluate(test_features, test_target, verbose=0)

# Make predictions on the test set
predictions = model.predict(test_features).flatten()

# Calculate the mean absolute error
mae = np.mean(np.abs(predictions - test_target))

print("Test Mean Absolute Error: {:.2f} Quality points".format(mae))
