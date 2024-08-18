
import numpy as np
import matplotlib.pyplot as plt

def print_progress_bar(percentage):
    """Pretty printing percentage level of completion"""
    # Ensure percentage is within the valid range
    if percentage < 0:
        percentage = 0
    elif percentage > 100:
        percentage = 100

    # Define the length of the progress bar
    bar_length = 50  # You can adjust this length as needed

    # Calculate the number of filled and empty segments
    filled_length = int(bar_length * percentage // 100)
    empty_length = bar_length - filled_length

    # Create the progress bar string
    bar = '█' * filled_length + '-' * empty_length

    # Print the progress bar with the percentage
    print(f'[{bar}] {percentage:.2f}%', end='\r')

def f(x, y):
    return np.sin(x) * np.cos(y)


def display_surface(x, y, func, labels):
    # Define the grid of points
    x, y = np.array(x), np.array(y)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')

    # Set labels
    ax.set_xlabel(labels["X"])
    ax.set_ylabel(labels["Y"])
    ax.set_zlabel(labels["Z"])

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

