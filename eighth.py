import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json

# Load the training history
with open('road_condition_training_history.json', 'r') as f:
    history = json.load(f)

# Plot training & validation accuracy values
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.savefig('training_validation_plots.png')
plt.show()

# Load your pre-trained road condition model
road_condition_model = tf.keras.models.load_model('road_condition_detection_model.h5')

# Create an example image (256x256x4)
example_image = np.random.rand(256, 256, 4)

# Predict road conditions
def predict_road_conditions(model, image):
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    predictions = model.predict(image)
    return predictions[0]

# Predict road condition
road_condition = predict_road_conditions(road_condition_model, example_image)
print(f"Predicted Road Condition Probabilities: {road_condition}")

# Visualize road condition predictions
classes = ['Snow', 'Rain', 'Fog','Lowlight']  # Adjusted to match the number of classes in the model
predicted_class = np.argmax(road_condition)
predicted_probabilities = road_condition

plt.figure()
plt.bar(classes, predicted_probabilities)
plt.title(f'Predicted Road Condition: {classes[predicted_class]}')
plt.ylabel('Probability')
plt.show()

# Example trajectory data (for LSTM model)
timesteps = 10
features = 5
example_trajectory_data = np.random.rand(1, timesteps, features)

# Placeholder function for trajectory prediction (replace with actual LSTM model)
def predict_trajectory(trajectory_data):
    # Simulated prediction output
    return np.random.rand(timesteps, 3)  # Example trajectory points

# Predict trajectory
predicted_trajectory = predict_trajectory(example_trajectory_data)
print(f"Predicted Trajectory: {predicted_trajectory}")

# Visualize predicted trajectory
plt.figure()
plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], marker='o')
plt.title('Predicted Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Placeholder function for path planning (replace with actual function)
def plan_path(image, current_trajectory, start, goal, grid):
    # Simulated path output
    return [(start[0] + i, start[1] + i) for i in range(11)]

# Plan path
start = (0, 0)
goal = (10, 10)
grid = np.zeros((20, 20))  # Example grid
path = plan_path(example_image, example_trajectory_data, start, goal, grid)

# Visualize planned path
plt.figure()
plt.imshow(grid, cmap='gray')
plt.plot([p[0] for p in path], [p[1] for p in path], marker='o')
plt.title('Planned Path')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
