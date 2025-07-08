import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

# Image pairs for each condition
images = [
    {"rgb": r"D:\iddaw\IDDAW\train\FOG\rgb\48\00000001_rgb.png", 
     "nir": r"D:\iddaw\IDDAW\train\FOG\nir\48\00000001_nir.png", "name": "Fog"},
    
    {"rgb": r"D:\iddaw\IDDAW\train\RAIN\rgb\3\00000002_rgb.png", 
     "nir": r"D:\iddaw\IDDAW\train\RAIN\nir\3\00000002_nir.png", "name": "Rain"},
    
    {"rgb": r"D:\iddaw\IDDAW\train\SNOW\rgb\180\00000001_rgb.png", 
     "nir": r"D:\iddaw\IDDAW\train\SNOW\nir\180\00000001_nir.png", "name": "Snow"},
    
    {"rgb": r"D:\iddaw\IDDAW\train\LOWLIGHT\rgb\110\00000000_rgb.png", 
     "nir": r"D:\iddaw\IDDAW\train\LOWLIGHT\nir\110\00000000_nir.png", "name": "Low Light"}


]

def load_image(path):
    """Load an image from a file path."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img

def predict_condition(weather_type):
    """Return the weather type as the condition."""
    return weather_type

def astar_path(start, goal):
    """Sample A* path."""
    return [(start[0], start[1]), (3, 3), (goal[0], goal[1])]

def rrt_path(start, goal):
    """Sample RRT path."""
    return [(start[0], start[1]), (random.randint(1, 5), random.randint(1, 5)), (goal[0], goal[1])]

def trajectory_prediction(condition, start):
    """Generate unique trajectories based on weather conditions."""
    trajectory = [start]
    
    if condition == "Fog":
        for _ in range(3):
            x = trajectory[-1][0] + random.uniform(0.5, 1.0)
            y = trajectory[-1][1] + random.uniform(-0.5, 0.5)
            trajectory.append((x, y))
    elif condition == "Rain":
        for _ in range(3):
            x = trajectory[-1][0] + 1
            y = trajectory[-1][1] + 0.5
            trajectory.append((x, y))
    elif condition == "Snow":
        for _ in range(3):
            x = trajectory[-1][0] + random.uniform(0.5, 1.0)
            y = trajectory[-1][1] + random.uniform(0.5, 1.5)
            trajectory.append((x, y))
    elif condition == "Low Light":
        for _ in range(3):
            x = trajectory[-1][0] + 1
            y = trajectory[-1][1] + random.uniform(-1, 1)
            trajectory.append((x, y))
    
    return trajectory

def plot_accuracy_loss(history, condition):
    """Plot accuracy and loss graphs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy')
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'Accuracy - {condition}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    # Plot Loss
    axes[1].plot(history['loss'], label='Training Loss')
    axes[1].plot(history['val_loss'], label='Validation Loss')
    axes[1].set_title(f'Loss - {condition}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.show()

def display_results():
    """Display images, path planning, and metrics for each image pair."""
    # Mock accuracy and loss history for demonstration
    mock_history = {
        'accuracy': [0.7, 0.8, 0.85, 0.9, 0.92],
        'val_accuracy': [0.68, 0.77, 0.83, 0.87, 0.9],
        'loss': [0.6, 0.5, 0.4, 0.35, 0.3],
        'val_loss': [0.65, 0.55, 0.45, 0.38, 0.32]
    }
    
    for pair in images:
        try:
            rgb_img = load_image(pair["rgb"])
            nir_img = load_image(pair["nir"])

            condition = predict_condition(pair["name"])
            print(f"Predicted Condition: {condition}")

            grid = np.zeros((10, 10))
            start, goal = (0, 0), (9, 9)

            astar = astar_path(start, goal)
            rrt = rrt_path(start, goal)
            trajectory = trajectory_prediction(condition, start)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Display RGB Image
            axes[0].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"RGB Image - Condition: {condition}")
            axes[0].axis('off')

            # Display NIR Image
            axes[1].imshow(nir_img, cmap='gray')
            axes[1].set_title("NIR Image")
            axes[1].axis('off')

            # Combined Path Planning
            axes[2].set_title("Combined Path Planning (A* + RRT)")
            for path, color, label in [(astar, 'blue', 'A* Path'), (rrt, 'green', 'RRT Path')]:
                if path:
                    axes[2].plot(*zip(*path), marker='o', color=color, label=label)
            axes[2].plot(start[1], start[0], 'r^', markersize=15, label='Start (Car)')
            axes[2].plot(goal[1], goal[0], 'bs', markersize=15, label='Goal')
            axes[2].legend()
            axes[2].grid(True)

            plt.show()

            # Trajectory Prediction Visualization
            plt.figure(figsize=(6, 6))
            plt.plot(*zip(*trajectory), marker='o', color='purple', label='Trajectory')
            plt.plot(start[1], start[0], 'r^', markersize=15, label='Start (Car)')
            plt.title(f"Trajectory Prediction - Condition: {condition}")
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot Accuracy and Loss Graphs
            plot_accuracy_loss(mock_history, condition)

        except FileNotFoundError as e:
            print(e)

# Run the display function
display_results()