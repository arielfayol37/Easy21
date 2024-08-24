
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


def display_surface(x, y, func, labels, title='title'):
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
    plt.title(title)
    if title != "title":
        plt.savefig(title + ".png")
    plt.show()

def plot_line(x_vals, y_vals, labels, title):
    # Create the line plot
    plt.figure(figsize=(10, 6))  # Optional: set the figure size
    plt.plot(x_vals, y_vals)

    # Add labels and title
    plt.xlabel(labels["X"])
    plt.ylabel(labels["Y"])
    plt.title(title)

    # Display the plot
    plt.grid(True)  # Optional: add a grid
    plt.savefig(title + ".png")  # Save the plot
    plt.show()  # Display the plot

